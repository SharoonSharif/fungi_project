# src/data_loader.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def get_transforms(train: bool = False):
    """
    If train=True, return augmentation transforms.
    If train=False, return only resize + normalize.
    """
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


class FungiDataset(Dataset):
    def __init__(self, root_dir, transforms=None, class_to_idx=None):
        """
        root_dir: path to train/ or val/ folder that contains class subfolders.
        transforms: torchvision transforms to apply.
        class_to_idx: a pre‐built mapping (class_name → index) to enforce consistent labeling.
        """
        self.root_dir = root_dir
        self.transforms = transforms

        if class_to_idx is None:
            # Build mapping from scratch (train time)
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            # Reuse mapping (val/test time)
            self.class_to_idx = class_to_idx
            self.classes = [None] * len(class_to_idx)
            for cls_name, idx in class_to_idx.items():
                self.classes[idx] = cls_name

        # Collect (image_path, label_idx) pairs
        self.samples = []
        for cls_name, idx in self.class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_folder, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label
