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
import torch.nn as nn
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
        "--backbone-train-last-n",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Number of last DINO backbone blocks to fine-tune (0 disables backbone fine-tuning).",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen backbone blocks.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=("auto", "fp32", "fp16", "bf16"),
        help="Training precision. auto picks bf16/fp16 on CUDA, otherwise fp32.",
    )


    return parser


def _resolve_precision(precision_arg: str) -> tuple[bool, torch.dtype | None, str]:
    if precision_arg == "fp32":
        return False, None, "fp32"
    if precision_arg == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, "bf16"
        print("Warning: bf16 requested but not supported. Falling back to fp16.")
        return True, torch.float16, "fp16"
    if precision_arg == "fp16":
        return True, torch.float16, "fp16"

    # auto mode: prefer bf16 (typically more stable), then fp16
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bf16"
    return True, torch.float16, "fp16"


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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training in this script.")

    # Define the device
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    use_amp, amp_dtype, resolved_precision = _resolve_precision(args.precision)
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    print(f"Using precision: {resolved_precision}")

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
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        load_backbone_for_training=True,
        head_hidden_channels=256,
        head_num_layers=3,
        use_multidepth_decoder=True,  # All-MLP head
        multidepth_feature_levels=8,
        
    ).to(device)

    trainable_backbone_params = model.enable_backbone_finetune(args.backbone_train_last_n)
    print(
        f"Backbone fine-tuning: last_n_blocks={args.backbone_train_last_n}, "
        f"trainable_backbone_params={trainable_backbone_params:,}, backbone_lr={args.backbone_lr:g}"
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {trainable_params:,}")

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    optimizer_param_groups = []
    if head_params:
        optimizer_param_groups.append({"params": head_params, "lr": args.lr})
    if backbone_params:
        optimizer_param_groups.append({"params": backbone_params, "lr": args.backbone_lr})

    if not optimizer_param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer = AdamW(optimizer_param_groups)

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
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
