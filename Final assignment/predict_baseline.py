"""
This script provides and example implementation of a prediction pipeline 
for a DINOv2 model with a segmentation head. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path
import time

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

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

PATCH_SIZE = Model.PATCH_SIZE  # ViT-S patch size, used for postprocessing to ensure correct resizing of output masks
PATCH_NR = Model.RESOLUTION // PATCH_SIZE  # Number of patches along one dimension, used for postprocessing to ensure correct resizing of output masks


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=(PATCH_NR*PATCH_SIZE, PATCH_NR*PATCH_SIZE), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5,), std=(0.5,)),
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


def _synchronize_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model.pt contains full model weights (backbone + segmentation head),
    # so loading the separate local backbone checkpoint is not required here.
    model = Model(load_pretrained_backbone=False)
    total_params, trainable_params = _count_parameters(model)
    print(f"Baseline Model Params: total={total_params:,}, trainable={trainable_params:,}")

    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict, 
        strict=True,
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    total_start = time.perf_counter()
    total_forward_time = 0.0
    processed_images = 0

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            _synchronize_if_cuda(device)
            forward_start = time.perf_counter()
            pred = model(img_tensor)
            _synchronize_if_cuda(device)
            total_forward_time += time.perf_counter() - forward_start
            processed_images += 1

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

    total_runtime = time.perf_counter() - total_start
    if processed_images > 0:
        avg_forward_ms = (total_forward_time / processed_images) * 1000.0
        forward_fps = processed_images / total_forward_time if total_forward_time > 0 else float("inf")
        end_to_end_fps = processed_images / total_runtime if total_runtime > 0 else float("inf")
        print(f"Inference speed (forward only): {avg_forward_ms:.2f} ms/image ({forward_fps:.2f} img/s)")
        print(f"Inference speed (end-to-end): {total_runtime:.2f} s total ({end_to_end_fps:.2f} img/s)")


if __name__ == "__main__":
    main()
