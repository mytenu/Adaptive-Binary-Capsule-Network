"""
Dataset utilities for the ABCN experiments.

Supports:
  - Ocular disease dataset (Glaucoma + Cataract, sourced from Kaggle)
  - CIFAR-10 benchmark

The ocular dataset is expected in the following folder structure:

    data/ocular/
        glaucoma_positive/   (glaucoma-affected images)
        glaucoma_negative/   (healthy retinal images)
        cataract_positive/   (cataract-affected images)
        cataract_negative/   (healthy eye images)

Class mapping:
    0 → Glaucoma-Positive
    1 → Glaucoma-Negative
    2 → Cataract-Positive
    3 → Cataract-Negative
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────
def get_transforms(image_size: int = 28, augment: bool = True):
    """
    Returns train / val-test transform pairs.

    Args:
        image_size (int): Resize target.
        augment    (bool): Whether to apply training augmentation.
    """
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalise,
    ]) if augment else transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalise,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalise,
    ])

    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────
# Ocular disease dataset  (70 : 20 : 10 split)
# ─────────────────────────────────────────────────────────────
def load_ocular_dataset(
    data_dir: str = "data/ocular",
    image_size: int = 28,
    batch_size: int = 100,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the combined glaucoma + cataract dataset and return three DataLoaders.

    The function expects an ImageFolder layout under ``data_dir`` with one
    sub-folder per class (see module docstring for the expected names).

    Args:
        data_dir   : Root folder of the ocular dataset.
        image_size : Spatial size to resize images to.
        batch_size : Mini-batch size.
        train_ratio: Fraction of data for training.
        val_ratio  : Fraction of data for validation.
        seed       : Random seed for reproducible splits.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf, val_tf = get_transforms(image_size, augment=True)

    # Full dataset with training transforms (we will override val/test below)
    full_dataset = ImageFolder(data_dir, transform=train_tf)

    n          = len(full_dataset)
    n_train    = int(n * train_ratio)
    n_val      = int(n * val_ratio)
    n_test     = n - n_train - n_val

    generator  = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply validation transform to val / test splits
    val_dataset  = ImageFolder(data_dir, transform=val_tf)
    test_dataset = ImageFolder(data_dir, transform=val_tf)

    val_ds  = Subset(val_dataset,  val_ds.indices)
    test_ds = Subset(test_dataset, test_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[OcularDataset] classes : {full_dataset.classes}")
    print(f"[OcularDataset] total   : {n}  |  train {n_train}  val {n_val}  test {n_test}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# CIFAR-10
# ─────────────────────────────────────────────────────────────
def load_cifar10(
    data_dir: str = "data/cifar10",
    image_size: int = 32,
    batch_size: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download and load CIFAR-10.

    Returns:
        (train_loader, val_loader, test_loader)
        Val loader is a 10 % subset of the training set; the remaining 90 %
        is used for training.  The original test split is used as test_loader.
    """
    train_tf, val_tf = get_transforms(image_size, augment=True)

    train_full = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds    = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    n_train = int(len(train_full) * 0.9)
    n_val   = len(train_full) - n_train
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"[CIFAR-10] train {n_train}  val {n_val}  test {len(test_ds)}")
    return train_loader, val_loader, test_loader
