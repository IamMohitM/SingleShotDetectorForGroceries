import os
from collections import Counter

import torch
import ttools
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import src.data.shelf_dataset as shelf_dataset
import src.scripts.bbox_utils as bb_utils


def normalize_image(image):
    """
    Normalizes the image
    :param image:
    :return: Normalized image values in the range 0 and 1
    """
    max_rgb = image.max()
    min_rgb = image.min()
    return (image - min_rgb) / (max_rgb - min_rgb)


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def load_dataset(dataset_path, annotation_path, dataset_type, batch_size):
    dataset = shelf_dataset.ShelfImageDataset(dataset_path=dataset_path, annotation_path=annotation_path,
                                              dataset_type=dataset_type)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, collate_fn=shelf_dataset.collate_fn)

    return dataloader


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def training_setup(interface, trainer_type, training_config, training_keys, validation_keys):
    """
    Setus up training with ttools based trainers and interface
    :param interface: An adapter for training the model
    :param trainer_type: The type of trainer to used
    :param training_config: A loaded config with only training parameters
    :param training_keys: A list of training metrics to track
    :param validation_keys: A list of validation metrics to track
    :return:
    """
    train_log = f'training_log_{training_config["log_directory_name"]}'
    val_log = f'Validation_log_{training_config["log_directory_name"]}'

    writer = SummaryWriter(
        os.path.join(interface.checkpoint_dir, 'summaries',
                     train_log), flush_secs=1)
    val_writer = SummaryWriter(
        os.path.join(interface.checkpoint_dir, 'summaries',
                     val_log),
        flush_secs=1)

    trainer = trainer_type(interface)
    trainer.add_callback(
        ttools.callbacks.TensorBoardLoggingCallback(keys=training_keys, val_keys=validation_keys,
                                                    writer=writer,
                                                    val_writer=val_writer,
                                                    frequency=3))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=training_keys))

    print(f"Type `tensorboard --logdir={trainer.interface.checkpoint_dir}` in your terminal to track progress")

    return trainer


def class_predictor(inputs_num, num_anchors=1, num_classes=12):
    """
    returns a
    :param inputs_num:
    :param num_anchors:
    :param num_classes:
    :return:
    """
    return torch.nn.Conv2d(inputs_num, num_anchors * num_classes,
                           kernel_size=3, padding=1)


def bbox_predictor(inputs_num, num_anchors=1):
    return torch.nn.Conv2d(inputs_num, num_anchors * 4, kernel_size=3, padding=1)


def predict(model, X):
    """
    Returns the output predictions for X (tensor image) with model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predicted_locs, predicted_scores = model(X.to(device))
    predicted_scores = F.softmax(predicted_scores, dim=2).permute(0, 2, 1)
    print("Detecting Objects")
    output = bb_utils.multibox_detection(predicted_scores, predicted_locs, model.priors_cxcy)
    # Filtering out background
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def classification_accuracy(cls_preds, cls_labels):
    """

    :param cls_preds: (batch, num_anchors, classes) - class predictions
    :param cls_labels: (batch, num_anchors) - class labels
    :return:
    """
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    true_labels = cls_labels != 0
    preds = cls_preds.argmax(dim=-1).type(cls_labels.dtype)

    correct_predictions = torch.sum(preds[true_labels] == cls_labels[true_labels])
    return correct_predictions / true_labels.sum()
    # return float(
    #     (cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum()) / cls_labels.numel()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """

    :param bbox_preds: (batch, num_anchors, 4)
    :param bbox_labels: (batch, labels)
    :param bbox_masks:
    :return:
    """
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()) / bbox_labels.numel()


def compute_precision_recall(all_predictions, all_truths, iou_threshold=0.5):
    """

    :param all_predictions: shape (n_test_images*num_anchors, 7) - (test_image_index, class_id, confidence, box_coords * 4)
    :param all_truths: shape (total ground truths in all test_images, 6) - (test_image_index, class_id, box_coords * 4)
    :param iou_threshold: min jaccard
    :return: a dictionary containing precision and recall values
    """

    true_positives = []
    false_positives = []
    false_negatives = []

    all_predictions = all_predictions[all_predictions[:, 1] != -1]  # remove background predictions

    image_indices = torch.unique(all_predictions[:, 0])

    for image_index in image_indices:
        image_predictions = all_predictions[all_predictions[:, 0] == image_index]
        image_truths = all_truths[all_truths[:, 0] == image_index]
        total_image_predictions = image_predictions.shape[0]
        total_image_truths = image_truths.shape[0]

        iou = bb_utils.box_iou(image_predictions[:, 3:], image_truths[:, 2:])

        # anchor_map = torch.zeros(total_image_predictions, dtype=torch.long, device=image_predictions.device)

        iou_condition = iou < iou_threshold
        missed_truths = torch.sum(torch.sum(iou_condition, dim=0) == total_image_predictions).item()  # false negatives
        false_predictions = torch.sum(torch.sum(iou_condition, dim=1) == total_image_truths).item()  # false_positives

        max_ious, _ = torch.max(iou, dim=1)
        true_predictions = torch.sum(max_ious >= 0.5).item()

        true_positives.append(true_predictions)
        false_positives.append(false_predictions)
        false_negatives.append(missed_truths)

    TP = sum(true_positives)
    FP = sum(false_positives)
    FN = sum(false_negatives)

    precision = TP / (TP + FP)
    recall = TP / TP / FN

    return {'precision': precision, 'recall': recall}


def compute_MAP(all_predictions, all_truths, iou_threshold=0.5, num_classes=11):
    """

    :param all_predictions: shape (n_test_images*num_anchors, 7) - (test_image_index, class_id, confidence, box_coords * 4)
    :param all_truths: shape (total ground truths in all test_images, 6) - (test_image_index, class_id, box_coords * 4)
    :param iou_threshold: min jaccard
    :param num_classes: total classes without background
    :return: a dictionary containing average precisions of each class and map for a given threshold
    """
    average_precisions = []

    epsilon = 1e-6
    for c in range(num_classes):
        detections = all_predictions[all_predictions[:, 1] == c].tolist()
        ground_truths = all_truths[all_truths[:, 1] == c].tolist()

        amount_bboxes = Counter([int(gt[0]) for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities (classification) which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        # TODO: Optimize
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            # ground truth from same image as current detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            # number of true bboxes in the image
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                pred_box = torch.Tensor(detection[3:]).unsqueeze(0)
                true_box = torch.Tensor(gt[2:]).unsqueeze(0)
                iou = bb_utils.box_iou(
                    pred_box,
                    true_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls).item())
    return {"average_precision": average_precisions, "mAP": sum(average_precisions) / len(average_precisions)}
