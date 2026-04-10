"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model
import os
import csv

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

PATCH_SIZE = Model.PATCH_SIZE 
PATCH_NR = Model.RESOLUTION // PATCH_SIZE


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def _infer_head_config(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    use_multidepth = any(key.startswith("seg_head.level_mlps.") for key in keys)

    if use_multidepth:
        level_ids = {
            int(match.group(1))
            for key in keys
            for match in [re.match(r"seg_head\.level_mlps\.(\d+)\.", key)]
            if match is not None
        }
        multidepth_feature_levels = max(len(level_ids), 2)

        fuse_conv_keys = [
            key
            for key, value in state_dict.items()
            if key.startswith("seg_head.fuse_mlp.") and key.endswith(".weight") and value.ndim == 4
        ]
        head_num_layers = len(fuse_conv_keys) + 1

        if "seg_head.fuse_mlp.0.weight" in state_dict:
            head_hidden_channels = int(state_dict["seg_head.fuse_mlp.0.weight"].shape[0])
        else:
            head_hidden_channels = int(state_dict["seg_head.out_conv.weight"].shape[1])

        return {
            "use_multidepth_decoder": True,
            "multidepth_feature_levels": multidepth_feature_levels,
            "head_num_layers": head_num_layers,
            "head_hidden_channels": head_hidden_channels,
        }

    hidden_conv_keys = [
        key
        for key, value in state_dict.items()
        if key.startswith("seg_head.hidden_layers.") and key.endswith(".weight") and value.ndim == 4
    ]
    head_num_layers = len(hidden_conv_keys) + 1

    if "seg_head.hidden_layers.0.weight" in state_dict:
        head_hidden_channels = int(state_dict["seg_head.hidden_layers.0.weight"].shape[0])
    else:
        head_hidden_channels = 512

    return {
        "use_multidepth_decoder": False,
        "multidepth_feature_levels": 4,
        "head_num_layers": head_num_layers,
        "head_hidden_channels": head_hidden_channels,
    }


def _load_non_ood_weights_strict(model: Model, state_dict: dict) -> None:
    """Load checkpoint while allowing external OOD detector weights.

    The OOD detector is initialized and loaded from its own checkpoint in Model,
    so submission model checkpoints may legitimately omit ``ood_detector.*`` keys.
    """
    load_info = model.load_state_dict(state_dict, strict=False)

    missing_non_ood = [k for k in load_info.missing_keys if not k.startswith("ood_detector.")]
    if missing_non_ood:
        raise RuntimeError(
            "Checkpoint is missing non-OOD model keys: " + ", ".join(missing_non_ood)
        )

    if load_info.unexpected_keys:
        raise RuntimeError(
            "Checkpoint has unexpected keys: " + ", ".join(load_info.unexpected_keys)
        )

def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=(PATCH_NR*PATCH_SIZE, PATCH_NR*PATCH_SIZE), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model state_dict")

    head_config = _infer_head_config(state_dict)

    # model.pt contains full model weights (backbone + segmentation head),
    # so loading the separate local backbone checkpoint is not required here.
    model = Model(
        load_backbone_for_training=False,
        use_multidepth_decoder=head_config["use_multidepth_decoder"],
        multidepth_feature_levels=head_config["multidepth_feature_levels"],
        head_num_layers=head_config["head_num_layers"],
        head_hidden_channels=head_config["head_hidden_channels"],
        ood=True,
    )

    _load_non_ood_weights_strict(model, state_dict)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))

    print(f"Found {len(image_files)} images to process.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open CSV file for predictions
    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass - model should return segmentation and inclusion decision
            seg_pred, include_decision, probability = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(seg_pred, original_shape)

            # Create mirrored output folder structure
            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            seg_pred_img = Image.fromarray(seg_pred.astype(np.uint8))
            seg_pred_img.save(out_path)

            # Record prediction
            predictions.append({
                'image_name': str(relative_path).replace('\\', '/'),
                'include': bool(include_decision),
            })

            print(f"[{img_path.name}] p={float(probability):.3f}: ", "ID" if bool(include_decision) else "OOD")

    # Write predictions to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")

if __name__ == "__main__":
    main()