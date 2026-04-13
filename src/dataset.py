"""
Dataset module for loading and augmenting fabric defect images.

Dataset: 6432 images across 12 defect categories (536 per class).
Split: 60% train (3859), 40% test (2573), 11% val (717)

Expected directory structure:
  data/raw/
    broken_end/
    broken_yarn/
    broken_pick/
    ...
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.preprocessing import preprocess_for_dcnn

logger = logging.getLogger(__name__)

DEFECT_CLASSES = [
    "broken_end", "broken_yarn", "broken_pick", "weft_curling",
    "fuzzy_ball", "cut_selvage", "crease", "warp_ball",
    "knots", "contamination", "nep", "weft_crack",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(DEFECT_CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class FabricDefectDataset(Dataset):
    """PyTorch dataset for fabric defect images."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 150,
        preprocess: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.preprocess = preprocess
        self.samples = []  # (image_path, label_idx)

        self._load_samples()

    def _load_samples(self):
        """Scan directory structure and collect image paths with labels."""
        if not self.root_dir.exists():
            logger.warning("Dataset directory not found: %s", self.root_dir)
            return

        for class_name in DEFECT_CLASSES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                # Try alternative naming conventions
                for alt in [class_name.replace("_", " "), class_name.replace("_", "-")]:
                    alt_dir = self.root_dir / alt
                    if alt_dir.exists():
                        class_dir = alt_dir
                        break

            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue

            label_idx = CLASS_TO_IDX[class_name]
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    self.samples.append((str(img_file), label_idx))

        logger.info("Loaded %d samples from %s", len(self.samples), self.root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        if self.preprocess:
            image = preprocess_for_dcnn(image, self.image_size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            from PIL import Image as PILImage
            image = PILImage.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_transforms(image_size: int = 150, is_train: bool = True) -> transforms.Compose:
    """Get data augmentation transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_data_loaders(
    root_dir: str,
    image_size: int = 150,
    batch_size: int = 32,
    train_split: float = 0.6,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.

    Returns (train_loader, val_loader, test_loader)
    """
    # Create full dataset without augmentation first for splitting
    full_dataset = FabricDefectDataset(
        root_dir=root_dir,
        transform=None,
        image_size=image_size,
        preprocess=True,
    )

    total = len(full_dataset)
    if total == 0:
        raise ValueError(f"No images found in {root_dir}")

    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply transforms
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    # Wrap with transforms
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info("Data split — Train: %d, Val: %d, Test: %d", n_train, n_val, n_test)
    return train_loader, val_loader, test_loader


def generate_synthetic_dataset(output_dir: str, num_per_class: int = 50):
    """
    Generate a synthetic dataset for demo/testing purposes.
    Creates simple fabric-like texture images with artificial defects.
    """
    output_path = Path(output_dir)
    rng = np.random.RandomState(42)

    for class_name in DEFECT_CLASSES:
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_per_class):
            # Generate base fabric texture
            img = rng.randint(100, 180, (150, 150), dtype=np.uint8)

            # Add horizontal/vertical line patterns (fabric weave)
            for y in range(0, 150, 3):
                img[y, :] = np.clip(img[y, :].astype(int) + rng.randint(-20, 20), 0, 255)
            for x in range(0, 150, 4):
                img[:, x] = np.clip(img[:, x].astype(int) + rng.randint(-15, 15), 0, 255)

            # Add class-specific defect patterns
            if class_name == "broken_end":
                y = rng.randint(30, 120)
                img[y:y+2, 20:130] = rng.randint(40, 80)
            elif class_name == "broken_yarn":
                x = rng.randint(30, 120)
                img[20:130, x:x+2] = rng.randint(40, 80)
            elif class_name == "broken_pick":
                y = rng.randint(40, 110)
                img[y:y+3, 10:140] = rng.randint(50, 90)
            elif class_name == "weft_curling":
                for offset in range(5):
                    y = 60 + int(15 * np.sin(np.linspace(0, 4*np.pi, 150))) + offset
                    for x in range(150):
                        yi = min(149, max(0, y))
                        img[yi, x] = rng.randint(60, 100)
            elif class_name == "fuzzy_ball":
                cx, cy = rng.randint(40, 110, 2)
                cv2.circle(img, (int(cx), int(cy)), rng.randint(5, 15), int(rng.randint(50, 90)), -1)
            elif class_name == "cut_selvage":
                y = rng.randint(20, 130)
                img[y:y+4, :] = rng.randint(30, 70)
            elif class_name == "crease":
                for offset in range(-2, 3):
                    x = 75 + offset
                    if 0 <= x < 150:
                        img[:, x] = np.clip(img[:, x].astype(int) - 40, 0, 255)
            elif class_name == "warp_ball":
                cx, cy = rng.randint(30, 120, 2)
                cv2.ellipse(img, (int(cx), int(cy)), (rng.randint(8, 20), rng.randint(3, 8)), 0, 0, 360, int(rng.randint(60, 100)), -1)
            elif class_name == "knots":
                for _ in range(rng.randint(1, 4)):
                    kx, ky = rng.randint(20, 130, 2)
                    cv2.circle(img, (int(kx), int(ky)), rng.randint(2, 6), int(rng.randint(40, 80)), -1)
            elif class_name == "contamination":
                cx, cy = rng.randint(30, 120, 2)
                size = rng.randint(10, 30)
                img[cy:cy+size, cx:cx+size] = rng.randint(30, 70, (min(size, 150-cy), min(size, 150-cx)))
            elif class_name == "nep":
                for _ in range(rng.randint(3, 8)):
                    nx, ny = rng.randint(10, 140, 2)
                    cv2.circle(img, (int(nx), int(ny)), rng.randint(1, 3), int(rng.randint(200, 255)), -1)
            elif class_name == "weft_crack":
                y = rng.randint(30, 120)
                length = rng.randint(40, 120)
                start_x = rng.randint(10, 150 - length)
                img[y:y+1, start_x:start_x+length] = rng.randint(30, 60)

            # Add slight noise
            noise = rng.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Save as RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(class_dir / f"{class_name}_{i:04d}.jpg"), img_rgb)

    logger.info("Generated synthetic dataset: %d classes x %d images in %s",
                len(DEFECT_CLASSES), num_per_class, output_dir)
