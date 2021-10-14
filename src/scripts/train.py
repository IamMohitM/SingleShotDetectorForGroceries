import argparse
import os
import random

import torch
import yaml

import src.scripts.utils as utils
from src.model.interface import TrainingInterface
from src.model.single_shot_detector import SSD300
from src.scripts.model_trainer import GeneralTrainer

N_CLASSES = 11 + 1

torch.manual_seed(20)
random.seed(20)


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    training_dataset = utils.load_dataset(config['dataset']['train_dataset_dir'],
                                          config['dataset']['training_annotation'],
                                          'train', config['dataset']['batch_size'])
    if config['training_parameters']['use_validation']:
        validation_dataset = utils.load_dataset(config['dataset']['val_dataset_dir'],
                                                config['dataset']['validation_annotation'],
                                                'test', config['dataset']['batch_size'])
    else:
        validation_dataset = None

    # validation_dataset = utils.load_dataset(config['dataset']['Validation_dir'], config['dataset']['validation'])
    checkpoint_path = os.path.join(config['training_parameters']['checkpoint_dir'],
                                   config['training_parameters']['log_directory_name'],
                                   'best_model.pth')
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = checkpoint['model_state_dict']
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_validation_loss']
        print("Checkpoint Loaded")
    else:
        model = SSD300(n_classes=N_CLASSES)
    model = model.to(device)

    interface = TrainingInterface(model, config['training_parameters'], best_val_loss, start_epoch)

    keys = ['training_loss', 'batch_accuracy', 'mean_absolute_error']
    validation_keys = ['validation_loss', 'validation_accuracy', 'mean_absolute_error']
    trainer = utils.training_setup(interface, GeneralTrainer, config['training_parameters'], keys, validation_keys)

    trainer.train(training_dataset, num_epochs=config['training_parameters']['epochs'] - start_epoch,
                  starting_epoch=start_epoch, val_dataloader=validation_dataset)

    # torch.save({"model_state_dict": trainer.interface.model.state_dict()},
    #            os.path.join(trainer.interface.checkpoint_dir, 'last_epoch.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Starts training the CNN for image classfication\n"
                                     "usage: python train.py --config_file path_to_config_file")
    parser.add_argument("--config_file", required=True, type=str, help="Path to config File")
    parser.add_argument("--model", required=False, type=str, default='transfer', help="Model to use (vgg16, siamese)")
    args = parser.parse_args()
    train(args)
