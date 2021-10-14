import os

import torch
import torchvision
from src.scripts.utils import predict
from src.scripts.bbox_utils import display_detections

path = '/Users/mo/Projects/InfilectObjectDetection/dataset/GroceryDataset_part1.tar-2/ShelfImages/test'

files = [os.path.join(path, image) for image in os.listdir(path)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('checkpoints/object_detector_adam/best_model.pth', map_location=device)
model = checkpoint['model_state_dict']
min_threshold = 0.5

image_products = {}
with torch.no_grad():
    for file in files:
        print(file)
        model.eval()
        X = torchvision.io.read_image(file).unsqueeze(0).float()
        X = torchvision.transforms.Resize((300, 300))(X)
        img = X.squeeze(0).permute(1, 2, 0).long()
        output = predict(model, X)
        filename = os.path.basename(file).split('.')[0]
        count = display_detections(img, output.cpu(), min_threshold, save_image=True, output_file=os.path.join('detection_images', filename + "_detect.png"))
        image_products[os.path.basename(file)] = count

    print(image_products)