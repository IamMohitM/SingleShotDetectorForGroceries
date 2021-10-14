import os

import torch
from ttools.training import ModelInterface

import src.scripts.bbox_utils as bb_utils
import src.scripts.utils as utils


class TrainingInterface(ModelInterface):
    # noinspection PyMissingConstructor
    def __init__(self, model, training_config, best_val_loss, epoch=0):
        self.model = model
        self.epoch_num = epoch
        lr = training_config["learning_rate"]
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.checkpoint_dir = os.path.join(training_config['checkpoint_dir'], training_config['log_directory_name'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = best_val_loss

        biases = list()
        not_biases = list()
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        if training_config['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                              lr=lr)
        else:
            self.optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                             lr=lr, momentum=training_config["momentum"],
                                             weight_decay=training_config["weight_decay"])
        self.neg_pos_ratio = 3
        class_weights = torch.tensor([0.05, 0.0065, 0.2218, 0.0676, 1.0065, 0.1637, 0.5915, 0.3549, 0.2168, 0.3458,
                                      0.8645, 0.8991])
        self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights).to(self.device)
        self.bbox_loss = torch.nn.L1Loss().to(self.device)

    def calc_loss(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, n_priors, num_classes = cls_preds.shape
        # cls_labels = cls_labels.view(-1)
        # cls_preds = cls_preds.view(-1, num_classes)
        cls = self.cls_loss(cls_preds.view(-1, num_classes),
                            cls_labels.view(-1)).view(batch_size, n_priors)

        # Hard negative mining
        positive_labels = cls_labels != 0
        n_positives = torch.sum(positive_labels, dim=1)  # positives examples for each item in batch (batch_size)
        n_hard_negatives = self.neg_pos_ratio * torch.sum(n_positives)
        positive_loss = cls[positive_labels]
        conf_loss_neg = cls.clone()
        conf_loss_neg[positive_labels] = 0
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        # cls = self.cls_loss(cls_preds.reshape(-1, num_classes),
        #                     cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        # hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
        #     self.device)  # (N, n_priors)
        # hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, n_priors)
        conf_loss_hard_neg = conf_loss_neg[:, :n_hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + positive_loss.sum()) / n_positives.sum().float()  # (), scalar

        bbox = self.bbox_loss(bbox_preds * bbox_masks,
                              bbox_labels * bbox_masks)
        return conf_loss + bbox

    def training_step(self, batch):

        self.model.train()
        self.optimizer.zero_grad()

        images, boxes, labels = batch

        images = images.to(self.device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(self.device) for b in boxes]
        labels = [l.to(self.device) for l in labels]

        predicted_locs, predicted_scores = self.model(images)
        bbox_labels, bbox_masks, cls_labels = bb_utils.multibox_target(self.model.priors_cxcy, labels, boxes)

        loss = self.calc_loss(predicted_scores, cls_labels, predicted_locs, bbox_labels,
                              bbox_masks)

        acc = utils.classification_accuracy(predicted_scores, cls_labels)
        mae = utils.bbox_eval(predicted_locs, bbox_labels, bbox_masks)

        loss.mean().backward()
        self.optimizer.step()

        # exponential moving average
        return {'training_loss': loss.mean().item(), 'batch_accuracy': acc.item(), 'mean_absolute_error': mae}

    def init_validation(self):
        metrics = ['validation_loss', 'validation_accuracy', 'mean_absolute_error', 'count']
        return {met: 0 for met in metrics}

    def validation_step(self, batch, running_val_data):
        self.model.eval()

        images, boxes, labels = batch

        images = images.to(self.device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(self.device) for b in boxes]
        labels = [l.to(self.device) for l in labels]
        count = running_val_data['count']
        n = images.shape[0]
        predicted_locs, predicted_scores = self.model(images)
        bbox_labels, bbox_masks, cls_labels = bb_utils.multibox_target(self.model.priors_cxcy, labels, boxes)
        loss = self.calc_loss(predicted_scores, cls_labels, predicted_locs, bbox_labels,
                              bbox_masks)
        acc = utils.classification_accuracy(predicted_scores, cls_labels)
        mae = utils.bbox_eval(predicted_locs, bbox_labels, bbox_masks)

        return {
            'validation_loss': ((running_val_data['validation_loss'] * count) + (loss.mean().item() * n)) / (
                    count + n),
            'validation_accuracy': ((running_val_data['validation_accuracy'] * count) + (acc.item() * n)) / (
                    count + n),
            'mean_absolute_error': ((running_val_data['mean_absolute_error'] * count) + (mae * n)) / (count + n),
            'count': count + n
        }
