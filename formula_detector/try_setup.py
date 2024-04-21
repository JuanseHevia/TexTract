import torchvision
import ast
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.io import read_image
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2 as T


def get_transform():
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class StackedFormulasDataset(torch.utils.data.Dataset):
    """
    Dataset class for the stacked formulas.
    """

    def __init__(self, root, transforms, filepath):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(filepath)

        self.images = self.data["stacked_fname"].values
        self.boxes = self.data["boxes"].apply(
            lambda v: np.array(ast.literal_eval(v)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = read_image(img_path)
        boxes = self.boxes.iloc[idx]

        # num objs is the number of bounding boxes
        num_objs = len(boxes)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([image_id])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = torch.zeros((num_objs, img.shape[-2], img.shape[-1]), dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def collate_fn(batch):
    return tuple(zip(*batch))

def run_test():
    pretrained_model = get_model_instance_segmentation(2)
    dataset = StackedFormulasDataset("stacked_formulae/train",
                                    get_transform(), "stacked_formulae/stacked_data_train.csv")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = pretrained_model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    pretrained_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = pretrained_model(x)  # Returns predictions
    print(predictions[0])

if __name__ == "__main__":
    run_test()
    print("Done running test!")
