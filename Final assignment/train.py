"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn.functional as Fnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ColorJitter,
    RandomResizedCrop,
    Resize,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    InterpolationMode
)
from torchvision.transforms.v2 import functional as F

from model import Model


def boundary_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    boundary_weight: float = 4.0,
    boundary_kernel_size: int = 5,
) -> torch.Tensor:
    """Cross-entropy with higher weight on class boundaries.

    Boundary pixels are detected with a local max/min label disagreement test.
    """
    if boundary_kernel_size % 2 == 0:
        raise ValueError("boundary_kernel_size must be odd")
    if boundary_weight < 1.0:
        raise ValueError("boundary_weight must be >= 1.0")

    # Per-pixel CE to apply custom spatial weights.
    ce_loss = Fnn.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        reduction="none",
    )

    valid_mask = (targets != ignore_index)
    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0

    label_map = safe_targets.unsqueeze(1).float()
    pad = boundary_kernel_size // 2
    local_max = Fnn.max_pool2d(label_map, kernel_size=boundary_kernel_size, stride=1, padding=pad)
    local_min = -Fnn.max_pool2d(-label_map, kernel_size=boundary_kernel_size, stride=1, padding=pad)

    boundary_mask = (local_max != local_min).squeeze(1) & valid_mask
    pixel_weights = torch.ones_like(ce_loss)
    pixel_weights[boundary_mask] = boundary_weight

    weighted_loss = ce_loss * pixel_weights
    weighted_loss = weighted_loss * valid_mask.float()

    normalizer = valid_mask.float().sum().clamp_min(1.0)
    return weighted_loss.sum() / normalizer


class SegmentationTrainTransforms:
    def __init__(self, size: int):
        self.size = size
        self.to_image = ToImage()
        self.to_float = ToDtype(torch.float32, scale=True)
        self.to_long = ToDtype(torch.int64)
        self.normalize = Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.hflip = RandomHorizontalFlip(p=1.0)  # We will apply horizontal flip with 50% probability in __call__

    def __call__(self, image, target):
        image = self.to_image(image)
        target = self.to_image(target)

        # Keep image and target aligned by sampling one crop and reusing it.
        top, left, height, width = RandomResizedCrop.get_params(
            image,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.3333333333333333),
        )

        image = F.resized_crop(
            image,
            top=top,
            left=left,
            height=height,
            width=width,
            size=(self.size, self.size),
            interpolation=InterpolationMode.BILINEAR,
        )
        target = F.resized_crop(
            target,
            top=top,
            left=left,
            height=height,
            width=width,
            size=(self.size, self.size),
            interpolation=InterpolationMode.NEAREST,
        )

        if torch.rand(1).item() < 0.5:
            image = self.hflip(image)
            target = self.hflip(target)

        if torch.rand(1).item() < 0.8:
            image = self.color_jitter(image)

        image = self.to_float(image)
        image = self.normalize(image)
        target = self.to_long(target)
        return image, target


class SegmentationEvalTransforms:
    def __init__(self, size: int):
        self.image_transform = Compose([
            ToImage(),
            Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            ToDtype(torch.float32, scale=True),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self.target_transform = Compose([
            ToImage(),
            Resize((size, size), interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),
        ])

    def __call__(self, image, target):
        return self.image_transform(image), self.target_transform(target)


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch Dinov3 model with a linear segmentation head")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="dinov3-training", help="Experiment ID for Weights & Biases")
    parser.add_argument(
        "--load-pretrained-backbone",
        action="store_true",
        default=False,
        help="Load local DINOv3 backbone checkpoint before training",
    )

    return parser


def main(args):
    patch_size = Model.PATCH_SIZE
    patch_nr = Model.RESOLUTION // patch_size
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = patch_nr * patch_size
    train_transforms = SegmentationTrainTransforms(size=image_size)
    eval_transforms = SegmentationEvalTransforms(size=image_size)

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
    args.data_dir,
    split="train",
    mode="fine",
    target_type="semantic",
    transforms=train_transforms,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transforms=eval_transforms,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        load_backbone_for_training=True 
    ).to(device)

    # Define boundary-aware loss to emphasize class borders.
    boundary_weight = 4.0
    boundary_kernel_size = 5

    # Define the optimizer
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )

    lmbda = lambda epoch: 0.9
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs
    # )

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = boundary_weighted_cross_entropy(
                outputs,
                labels,
                ignore_index=255,
                boundary_weight=boundary_weight,
                boundary_kernel_size=boundary_kernel_size,
            )
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = boundary_weighted_cross_entropy(
                    outputs,
                    labels,
                    ignore_index=255,
                    boundary_weight=boundary_weight,
                    boundary_kernel_size=boundary_kernel_size,
                )
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)

        if epoch > 10:
            scheduler.step()
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
