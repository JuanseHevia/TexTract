from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
from torchvision.transforms import v2 as T



def get_transform():
    transforms = []
    # resize images
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class StackedFormulasDataset(torch.utils.data.Dataset):
    """
    Dataset class for the stacked formulas.
    """

    def __init__(self, root, transforms, filepath, device="cpu"):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(filepath)
        self.device = device

        self.images = self.data["stacked_fname"].values
        self.boxes = self.data["boxes"].apply(lambda v: np.array(ast.literal_eval(v)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = read_image(img_path)
        boxes = self.boxes.iloc[idx]
        
        # num objs is the number of bounding boxes
        num_objs = len(boxes)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64, device=self.device)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img, device=self.device)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img), device=self.device)
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64, device=self.device)
        target["image_id"] = torch.tensor([image_id], device=self.device)
        target["area"] = area
        target["iscrowd"] = iscrowd
        # insert mock masks to comply with dataset structure
        target["masks"] = torch.zeros((num_objs, img.shape[-2], img.shape[-1]), dtype=torch.uint8, device=self.device)


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


    def plot_img_and_boxes(self, img, target):
        boxes = target["boxes"]
        img = F.to_pil_image(img)
        plt.imshow(img)
        for box in boxes:
            plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'r-')
        # plt.axis('off')
        plt.show()