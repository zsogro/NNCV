from pathlib import Path
import numpy as np

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


for images in Path("local_output").glob("*.png"):
    print(f"Processing {images}...")
    label = cv2.imread(str(images), cv2.IMREAD_GRAYSCALE)
    color = label_to_color_image(label)
    out_path = f"colorized_output/{images.stem}_colorized.png"
    cv2.imwrite(out_path, color)

