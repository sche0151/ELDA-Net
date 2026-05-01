"""
ELDA-Net Dataset Loader with Albumentations Augmentation
=========================================================
Thesis Section 2.6 (augmentation) and Section 5.2.1 (dataset splits).

Augmentation pipeline (12 types, thesis Section 2.6):
  horizontal_flip, brightness_contrast, gaussian_blur, motion_blur,
  synthetic_rain, synthetic_fog, nighttime_gamma, random_erasing,
  elastic_distortion, grid_distortion, perspective_transform, hsv_jitter

ImageNet normalization applied after augmentation (thesis Section 3.2).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Optional Albumentations import (graceful fallback if not installed)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


def _build_augmentation_pipeline(aug_cfg: dict, input_size: tuple, train: bool):
    """
    Build Albumentations pipeline from config (thesis Section 2.6).
    input_size: (height, width)
    """
    if not _HAS_ALBUMENTATIONS or not train:
        return None

    h, w = input_size
    p = aug_cfg  # shorthand

    transforms = [
        A.Resize(h, w),
        A.HorizontalFlip(p=p.get('horizontal_flip', 0.5)),
        A.RandomBrightnessContrast(p=p.get('brightness_contrast', 0.4)),
        A.GaussianBlur(p=p.get('gaussian_blur', 0.3)),
        A.MotionBlur(p=p.get('motion_blur', 0.3)),
        A.RandomRain(p=p.get('synthetic_rain', 0.3)),
        A.RandomFog(p=p.get('synthetic_fog', 0.2)),
        A.RandomGamma(gamma_limit=(40, 80), p=p.get('nighttime_gamma', 0.3)),
        A.CoarseDropout(p=p.get('random_erasing', 0.2)),
        A.ElasticTransform(p=p.get('elastic_distortion', 0.2)),
        A.GridDistortion(p=p.get('grid_distortion', 0.15)),
        A.Perspective(p=p.get('perspective_transform', 0.3)),
        A.HueSaturationValue(p=p.get('hsv_jitter', 0.4)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


class LaneDataset(Dataset):
    """
    Lane segmentation dataset loader.
    Expects image_dir and label_dir with matching filenames.
    Labels are grayscale binary masks (255 = lane, 0 = background).

    input_size: (height, width) -- note: H x W order, unlike cv2.resize (W x H).
    """

    def __init__(self, image_dir: str, label_dir: str,
                 input_size,
                 train: bool = True,
                 aug_cfg: dict | None = None,
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225)):
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        # input_size stored as (H, W)
        if isinstance(input_size, (list, tuple)):
            self.h, self.w = int(input_size[0]), int(input_size[1])
        else:
            self.h = self.w = int(input_size)
        self.train = train
        self.mean = np.array(normalize_mean, dtype=np.float32)
        self.std  = np.array(normalize_std,  dtype=np.float32)

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        # Build augmentation pipeline
        self.aug_pipeline = None
        if aug_cfg and _HAS_ALBUMENTATIONS and train:
            self.aug_pipeline = _build_augmentation_pipeline(
                aug_cfg, (self.h, self.w), train)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        img_path   = os.path.join(self.image_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label not found: {label_path}")

        if self.aug_pipeline is not None:
            augmented = self.aug_pipeline(image=image, mask=label)
            image_tensor = augmented['image']           # already ToTensorV2
            mask_tensor  = augmented['mask'].float() / 255.0
            mask_tensor  = mask_tensor.unsqueeze(0)
        else:
            # Manual resize + normalize (inference / validation)
            image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
            mask_tensor  = torch.from_numpy((label / 255.0).astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor
