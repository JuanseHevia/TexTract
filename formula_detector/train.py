import wandb
from model import get_model_instance_segmentation
from dataset import StackedFormulasDataset, get_transform
import os
import torch
import torchvision.ops as tv_ops
import torchvision.models as tv_models

def collate_fn(batch):
    return tuple(zip(*batch))


NUM_CLASSES = 2
# DEVICE = torch.device("mps")
DEVICE="cpu"

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = 0


def train(config):
    """Run training for the formula detector model."""

    if config["wandb"]:
        wandb.init(project="formula-detector",
                config=config)

    # model = get_model_instance_segmentation(NUM_CLASSES)
    model = tv_models.detection.fasterrcnn_resnet50_fpn_v2()
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = tv_models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(DEVICE)

    # use our dataset and defined transformations
    dataset = StackedFormulasDataset(
        root="stacked_formulae/train", transforms=get_transform(), filepath="stacked_formulae/stacked_data_train.csv", device=DEVICE)
    dataset_test = StackedFormulasDataset(
        root="stacked_formulae/val", transforms=get_transform(), filepath="stacked_formulae/stacked_data_val.csv", device=DEVICE)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        # shuffle=True,
        # num_workers=4,
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config["test_batch_size"],
        # shuffle=False,
        # num_workers=4,
        collate_fn=collate_fn
    )

    # construct an optimizer
    params = [p.to(DEVICE) for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config["lr"],
        momentum=0.9,
        # weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it for 10 epochs
    num_epochs = 5
    try:
        for ep in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            model.train()
            for images, targets in data_loader:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: torch.Tensor(v).to(DEVICE)
                            for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss = losses.item()
                if config["wandb"]:
                    wandb.log({"train/loss": total_loss})

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            model.eval()
            for images, targets in data_loader_test:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: torch.Tensor(v).to(DEVICE)
                            for k, v in t.items()} for t in targets]

                outputs = model(images)
                # evaluate model output
                res = {target["image_id"]: output for target,
                    output in zip(targets, outputs)}

                # Compute metrics
                iou = tv_ops.box_iou([t["boxes"]
                                    for t in targets], [r["boxes"] for r in res])
                if config["wandb"]:
                    wandb.log({"test/iou": iou})

            # save model checkpoint
            torch.save(model.state_dict(), f"formula_detector_weights_{ep}.pth")
    except KeyboardInterrupt:
        # allow for clean exit on command line interrupt
        torch.save(model.state_dict(), "formula_detector_weights.pth")

    if config["wandb"]:
        wandb.finish()

if __name__ == "__main__":
    config = {
        "lr": 0.005,
        "train_batch_size": 8,
        "test_batch_size": 4,
        "wandb": True
    }
    train(config)
