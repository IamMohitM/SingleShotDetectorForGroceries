import os

import PIL.Image as Image
import pandas as pd
import torch

import src.data.transformation_utils as tu


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels  # tensor (N, 3, 300, 300), 2 lists of N tensors each


class ShelfImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, annotation_path, dataset_type='train', image_transformations=None,
                 transformation=None):
        assert dataset_type.lower() in ['train', 'test']
        self.dataset_type = dataset_type
        # self._MAX_BOXES = 200
        self.dataset_path = dataset_path
        self.transforms = transformation
        # self.dataset_images = [image for image in os.listdir(dataset_path)
        #                        if image.endswith('.JPG')]
        self.data = pd.read_csv(annotation_path, index_col=False)
        self.dataset_images = self.data['image_name'].unique().tolist()


    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, index):
        image_name = self.dataset_images[index]
        coordinate_keys = ['b_i', 'x_1', 'y_1', 'x_2', 'y_2']
        targets = torch.tensor(self.data.loc[self.data['image_name'] == image_name, coordinate_keys].values)
        labels = targets[:, 0]
        boxes = targets[:, 1:]
        image_path = os.path.join(self.dataset_path, image_name)
        image = Image.open(image_path)

        image, boxes, labels = tu.transform(image, boxes.float(), labels.long(), self.dataset_type)
        return image, boxes, labels


if __name__ == "__main__":
    a = torch.utils.data.DataLoader(ShelfImageDataset(
        dataset_path='/Users/mo/Projects/InfilectObjectDetection/dataset/GroceryDataset_part1.tar-2/ShelfImages/test',
        annotation_path='/Users/mo/Projects/InfilectObjectDetection/dataset/GroceryDataset_part1.tar-2/ShelfImages/testing_annotations.csv',
        dataset_type='test'),
        batch_size=3, collate_fn=collate_fn)
    for i, b, l in a:
        break
