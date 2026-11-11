"""
Generate a fixed camera-like validation set from the original 120 validation images.
Applies realistic smartphone camera variations: random crops, rotations, color jitter, and flips.
Generates the images once with a fixed seed (42) and saves them to task4/camera_val/.
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def generate_camera_val():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    source_dir = Path("ex08_leaves_images/leaves/val")
    dest_dir = Path("task4/camera_val")
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    print(f"Generating camera-like validation set from: {source_dir}")
    print(f"Target directory: {dest_dir}")
    print(f"Classes: {classes}")

    # Camera perturbation transform (outputs a PIL Image resized to 96x96)
    camera_sim_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    total_generated = 0
    for cls in classes:
        src_cls_dir = source_dir / cls
        dst_cls_dir = dest_dir / cls
        dst_cls_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(src_cls_dir.glob("*.jpg")) + list(src_cls_dir.glob("*.jpeg")) + list(src_cls_dir.glob("*.png")))
        for f in files:
            img = Image.open(f).convert("RGB")
            perturbed_img = camera_sim_transform(img)
            dst_path = dst_cls_dir / f.name
            perturbed_img.save(dst_path, quality=95)
            total_generated += 1

    print(f"Successfully generated {total_generated} camera-like validation images in {dest_dir}.")

if __name__ == "__main__":
    generate_camera_val()
