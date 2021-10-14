import argparse

import torch
import torchvision

import matplotlib.pyplot as plt
from src.scripts.utils import predict
from src.scripts.bbox_utils import display_detections


def detect_object(args):
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model = checkpoint['model_state_dict']
        model.eval()
        print(args.image)
        X = torchvision.io.read_image(args.image).unsqueeze(0).float()
        X = torchvision.transforms.Resize((300, 300))(X)
        print(X.shape)
        img = X.squeeze(0).permute(1, 2, 0).long()
        plt.imshow(img)
        print("Computing Predictions")
        output = predict(model, X)
        print("Displaying Predictions")

        count = display_detections(img, output.cpu(), args.min_threshold, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Displays product detections in the image\n"
                                     "usage: python detect_objects.py --checkpoint path_model_checkpoint ----image path_to_dataset --min_threshold 0.7")
    parser.add_argument('--checkpoint_path', required=True, help="Path to Model Checkpoint")
    parser.add_argument("--image", required=True, type=str, help="Path to dataset")
    parser.add_argument("--min_threshold", required=False, type=float, help="Minimum threshold", default=0.5)
    parser.add_argument("--save", default=False, help='Save the image as out.png')
    args = parser.parse_args()
    detect_object(args)
