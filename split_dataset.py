# src/split_dataset.py

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, train_ratio=0.8):
    """
    Splits images in source_dir/<class-name>/ into train/ and val/ 
    under dest_dir, using the given train_ratio.
    """
    class_names = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    for cls in tqdm(class_names, desc="Splitting classes"):
        cls_path = os.path.join(source_dir, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images   = images[split_idx:]

        for phase, phase_images in zip(["train", "val"], [train_images, val_images]):
            phase_dir = os.path.join(dest_dir, phase, cls)
            os.makedirs(phase_dir, exist_ok=True)
            for img in phase_images:
                src = os.path.join(cls_path, img)
                dst = os.path.join(phase_dir, img)
                shutil.copy2(src, dst)
