import argparse

import torch
from torch.nn import functional as F
from tqdm import tqdm
import src.scripts.bbox_utils as bb_utils
import src.scripts.utils as utils


def evaluate(args):
    dataset_path = args.dataset_path
    annotation_path = args.annotation_path

    dataset = utils.load_dataset(dataset_path, annotation_path, 'test', 32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = checkpoint['model_state_dict']
    model = model.to(device)
    model.eval()
    all_preds = []
    example = 0
    true_boxes = []
    true_labels = []
    true_index = []
    with torch.no_grad():
        for images, boxes, labels in tqdm(dataset):

            images = images.to(device)
            labels = [l.to(device) for l in labels]
            for i, b in enumerate(boxes):
                boxes[i] = b.to(device)
                true_index.extend(torch.full((b.shape[0],), example))
                example += 1

            true_boxes.extend(boxes)
            true_labels.extend(labels)
            predicted_locs, predicted_scores = model(images)
            predicted_scores = F.softmax(predicted_scores, dim=2).permute(0, 2, 1)
            output = bb_utils.multibox_detection(predicted_scores, predicted_locs, model.priors_cxcy)
            all_preds.append(output)

    true_boxes = torch.vstack(true_boxes).to(device)
    true_labels = torch.hstack(true_labels).reshape(-1, 1).to(device)
    true_labels = true_labels - 1  # necessary because labels in dataset start with 1 but preds start with 0
    true_index = torch.vstack(true_index).to(device)
    all_truths = torch.hstack((true_index, true_labels, true_boxes))  # (test_image_index, class_id, box_coords * 4)

    all_preds = torch.vstack(all_preds).to(device)
    pred_index = []
    for example_idx in range(all_preds.shape[0]):
        pred_index.extend([example_idx] * all_preds.shape[1])

    all_preds = torch.hstack((torch.Tensor(pred_index).view(-1, 1).to(device),
                              all_preds.view(-1, 6)))  # (test_image_index, class_id, confidence, box_coords * 4)

    print("Computing MAP")
    vals = utils.compute_MAP(all_preds, all_truths)
    print("Computing precision and recall")
    pr_dict = utils.compute_precision_recall(all_preds, all_truths)
    print(vals)
    print(pr_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluates an object detection network\n"
                                     "usage: python evaluate_detector.py --checkpoint path_model_checkpoint --dataset_path path_to_dataset --annotation_path path_to_annotations(csv)")
    parser.add_argument('--checkpoint_path', required=True, help="Path to Model Checkpoint")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to dataset")
    parser.add_argument("--annotation_path", required=False, type=str, help="Path to Annotation")
    args = parser.parse_args()
    evaluate(args)
