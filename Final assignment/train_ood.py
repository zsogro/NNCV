"""Train a normalizing-flow OOD detector on DINOv3 patch tokens."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)

import wandb

from model import Model

# Ensure local normalizing-flows package is importable by ood_model.py.
_THIS_DIR = Path(__file__).resolve().parent
_NF_LOCAL_REPO = _THIS_DIR / "normalizing-flows"
if _NF_LOCAL_REPO.exists() and str(_NF_LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(_NF_LOCAL_REPO))

from ood_model import OOD_Detector


class OODTrainTransforms:
    def __init__(self, size: int):
        self.image_transform = Compose(
            [
                ToImage(),
                Resize((size, size), interpolation=InterpolationMode.BILINEAR),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ToDtype(torch.float32, scale=True),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image, target):
        # Return a dummy tensor target to keep DataLoader collation valid.
        return self.image_transform(image), torch.tensor(0, dtype=torch.int64)


class OODevalTransforms:
    def __init__(self, size: int):
        self.image_transform = Compose(
            [
                ToImage(),
                Resize((size, size), interpolation=InterpolationMode.BILINEAR),
                ToDtype(torch.float32, scale=True),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image, target):
        # Return a dummy tensor target to keep DataLoader collation valid.
        return self.image_transform(image), torch.tensor(0, dtype=torch.int64)


def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser("Train OOD detector on DINOv3 patch tokens")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--flow-dim", type=int, default=128, help="Projected token dimensionality for flow")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for projector/flow MLPs")
    parser.add_argument("--num-flow-layers", type=int, default=8, help="Number of flow coupling layers")
    parser.add_argument(
        "--token-sample-size",
        type=int,
        default=4096,
        help="Number of tokens sampled per step for likelihood loss",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Validation ID-score quantile used as OOD threshold",
    )

    parser.add_argument("--experiment-id", type=str, default="dinov3-ood-flow", help="Run name")
    parser.add_argument("--output-root", type=str, default="checkpoints", help="Checkpoint root directory")
    parser.add_argument("--wandb-project", type=str, default="5lsm0-cityscapes-ood", help="wandb project name")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="If > 0, limit train batches per epoch (for debugging)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="If > 0, limit validation batches per epoch (for debugging)",
    )

    return parser


def _extract_patch_tokens(backbone_model: Model, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return backbone_model._forward_backbone_patch_tokens(images)


def _build_backbone(device: torch.device) -> Model:
    backbone_model = Model(
        in_channels=3,
        n_classes=19,
        load_backbone_for_training=True,
    ).to(device)
    for parameter in backbone_model.parameters():
        parameter.requires_grad = False
    backbone_model.eval()
    return backbone_model


def _save_checkpoint(
    output_path: str,
    detector: OOD_Detector,
    args,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    threshold: float,
) -> None:
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "threshold": threshold,
        "detector_state_dict": detector.state_dict(),
        "detector_config": {
            "token_dim": detector.token_dim,
            "flow_dim": detector.flow_dim,
            "hidden_dim": detector.hidden_dim,
            "num_flow_layers": detector.num_flow_layers,
            "token_sample_size": detector.token_sample_size,
        },
        "args": vars(args),
    }
    torch.save(payload, output_path)


def main(args) -> None:
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, name=args.experiment_id, config=vars(args))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(args.output_root, args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    image_size = (Model.RESOLUTION // Model.PATCH_SIZE) * Model.PATCH_SIZE
    train_transforms = OODTrainTransforms(size=image_size)
    eval_transforms = OODevalTransforms(size=image_size)

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
        pin_memory=torch.cuda.is_available(),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone_model = _build_backbone(device)
    detector = OOD_Detector(
        token_dim=backbone_model.embed_dim,
        flow_dim=args.flow_dim,
        hidden_dim=args.hidden_dim,
        num_flow_layers=args.num_flow_layers,
        token_sample_size=args.token_sample_size,
    ).to(device)

    optimizer = AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_loss = float("inf")
    best_path: str | None = None

    for epoch in range(args.epochs):
        detector.train()
        train_losses: list[float] = []

        for step, (images, _) in enumerate(train_dataloader, start=1):
            images = images.to(device, non_blocking=True)
            tokens = _extract_patch_tokens(backbone_model, images)

            optimizer.zero_grad(set_to_none=True)
            loss = detector.loss(tokens)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

            if not args.disable_wandb:
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=epoch * len(train_dataloader) + (step - 1),
                )

            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break

        train_loss = sum(train_losses) / max(len(train_losses), 1)

        detector.eval()
        valid_losses: list[float] = []
        valid_scores: list[torch.Tensor] = []
        with torch.no_grad():
            for step, (images, _) in enumerate(valid_dataloader, start=1):
                images = images.to(device, non_blocking=True)
                tokens = _extract_patch_tokens(backbone_model, images)

                v_loss = detector.loss(tokens)
                valid_losses.append(float(v_loss.item()))

                scores = detector(tokens).detach().cpu()
                valid_scores.append(scores)

                if args.max_val_batches > 0 and step >= args.max_val_batches:
                    break

        valid_loss = sum(valid_losses) / max(len(valid_losses), 1)

        score_tensor = torch.cat(valid_scores, dim=0)
        threshold = float(torch.quantile(score_tensor, args.threshold_quantile).item())
        detector.threshold = threshold

        if not args.disable_wandb:
            wandb.log(
                {
                    "train/loss_epoch": train_loss,
                    "valid/loss_epoch": valid_loss,
                    "valid/threshold": threshold,
                    "epoch": epoch + 1,
                },
                step=(epoch + 1) * len(train_dataloader) - 1,
            )

        print(
            f"Epoch {epoch + 1:04}/{args.epochs:04} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} threshold={threshold:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if best_path is not None and os.path.exists(best_path):
                os.remove(best_path)
            best_path = os.path.join(
                output_dir,
                f"best_ood_detector-epoch={epoch + 1:04}-val_loss={valid_loss:.6f}.pt",
            )
            _save_checkpoint(
                output_path=best_path,
                detector=detector,
                args=args,
                epoch=epoch + 1,
                train_loss=train_loss,
                valid_loss=valid_loss,
                threshold=threshold,
            )

    final_path = os.path.join(output_dir, "final_ood_detector.pt")
    _save_checkpoint(
        output_path=final_path,
        detector=detector,
        args=args,
        epoch=args.epochs,
        train_loss=train_loss,
        valid_loss=valid_loss,
        threshold=detector.threshold if detector.threshold is not None else float("nan"),
    )

    print(f"Training complete. Final checkpoint: {final_path}")
    if best_path is not None:
        print(f"Best checkpoint: {best_path}")

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    print("Training Out of Ordinary detector.")
    parser = get_args_parser()
    parsed_args = parser.parse_args()

    if parsed_args.threshold_quantile <= 0.0 or parsed_args.threshold_quantile >= 1.0:
        raise ValueError("--threshold-quantile must be in (0, 1)")

    if parsed_args.token_sample_size is not None and parsed_args.token_sample_size <= 0:
        raise ValueError("--token-sample-size must be > 0")

    main(parsed_args)
