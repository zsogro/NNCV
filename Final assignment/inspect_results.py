from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Cityscapes color palette (trainId → color)
CITYSCAPES_COLORS = {
    0: (128, 64,128),   # road
    1: (244, 35,232),   # sidewalk
    2: ( 70, 70, 70),   # building
    3: (102,102,156),   # wall
    4: (190,153,153),   # fence
    5: (153,153,153),   # pole
    6: (250,170, 30),   # traffic light
    7: (220,220,  0),   # traffic sign
    8: (107,142, 35),   # vegetation
    9: (152,251,152),   # terrain
    10:( 70,130,180),   # sky
    11:(220, 20, 60),   # person
    12:(255,  0,  0),   # rider
    13:(  0,  0,142),   # car
    14:(  0,  0, 70),   # truck
    15:(  0, 60,100),   # bus
    16:(  0, 80,100),   # train
    17:(  0,  0,230),   # motorcycle
    18:(119, 11, 32),   # bicycle
}

def label_to_color_image(label_img):
    """
    Convert a 2D label image (H, W) to a color RGB image (H, W, 3)
    """
    h, w = label_img.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in CITYSCAPES_COLORS.items():
        mask = label_img == label
        color_img[mask] = color[::-1]  # Convert RGB to BGR for OpenCV

    return color_img

import cv2



import os
import math
import matplotlib.pyplot as plt
from PIL import Image

def plot_all_pairs_grid(folder_path, cols=2):
    # 1. Match the pairs
    image_pairs = {}
    files = os.listdir(folder_path)
    for f in files:
        if f.endswith('_colorized.png'):
            img_id = f.replace('_colorized.png', '')
            image_pairs.setdefault(img_id, {})['color'] = f
        elif f.endswith('_gtFine.png'):
            img_id = f.replace('_gtFine.png', '')
            image_pairs.setdefault(img_id, {})['gt'] = f

    # Filter to only complete pairs
    valid_pairs = [v for k, v in image_pairs.items() if 'color' in v and 'gt' in v]
    num_pairs = len(valid_pairs)

    if num_pairs == 0:
        print("No pairs found!")
        return

    # 2. Setup the grid (2 columns per pair: Color | GT)
    # We will treat each pair as a single "unit" in a grid
    rows = num_pairs
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))

    # If there's only one pair, axes is a 1D array; make it 2D for consistency
    if rows == 1:
        axes = [axes]

    # 3. Plotting
    for i, paths in enumerate(valid_pairs):
        img_color = Image.open(os.path.join(folder_path, paths['color']))
        img_gt = Image.open(os.path.join(folder_path, paths['gt']))

        # Left side: Colorized
        axes[i][0].imshow(img_color)
        axes[i][0].set_title(f"Pair {i+1}: Colorized")
        axes[i][0].axis('off')

        # Right side: gtFine
        axes[i][1].imshow(img_gt)
        axes[i][1].set_title(f"Pair {i+1}: gtFine")
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()


def colorize_images():
    for images in Path("local_output").glob("*.png"):
        label = cv2.imread(str(images), cv2.IMREAD_GRAYSCALE)
        color = label_to_color_image(label)
        out_path = f"colorized_output/{images.stem}_colorized.png"
        cv2.imwrite(out_path, color)


if __name__ == "__main__":
    colorize_images()    
    plot_all_pairs_grid(Path("colorized_output") )