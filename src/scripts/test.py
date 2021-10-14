import torch
import random
import matplotlib.pyplot as plt
import src.data.shelf_dataset as shelf_dataset
import src.scripts.utils as utils
import src.scripts.bbox_utils as bb_utils

if __name__ == "__main__":

    a = torch.utils.data.DataLoader(shelf_dataset.ShelfImageDataset(
        dataset_path='/Users/mo/Projects/InfilectObjectDetection/dataset/GroceryDataset_part1.tar-2/ShelfImages/test',
        annotation_path='/Users/mo/Projects/InfilectObjectDetection/dataset/GroceryDataset_part1.tar-2/ShelfImages/testing_annotations.csv',
    dataset_type='train'),
        batch_size=8, collate_fn = shelf_dataset.collate_fn)
    for images, boxes, labels in a:
        # im = images[0]
        # fig, axes = plt.subplots(4, 2, figsize=(15, 25))
        # axes = axes.flatten()

        # for i, ax in enumerate(axes):
        #     image = im[i]
        #     image = image.permute(1, 2, 0)
        #     image = utils.normalize_image(image)
        #     ax.imshow(image)
        #     bboxes = b[i]
        #     labels = l[i]
        #     # bboxes = bboxes[labels != -1, :]
        #     # print(bboxes.shape)
        #     bb_utils.show_bboxes(ax, b[i] * torch.tensor((300, 300, 300, 300)),  labels=labels[labels != -1].tolist())
            #
        if random.random() > 0.3:
            rand_int = random.randint(0, 7)

            # print(im.shape)
            image = images[rand_int]
            print(image.shape)
            image = image.permute(1, 2, 0)
            image = utils.normalize_image(image)
            f = plt.imshow(image)
            bboxes = boxes[rand_int]
            labels = labels[rand_int]
            # bboxes = bboxes[labels != -1, :]
            # print(bboxes.shape)
            bb_utils.show_bboxes(f.axes, boxes[rand_int] * torch.tensor((300, 300, 300, 300)),  labels=labels.tolist())
            break
        # break

    plt.show()
