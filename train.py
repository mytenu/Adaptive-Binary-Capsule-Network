"""
Training and evaluation engine for the Adaptive Binary Capsule Network.

Hyper-parameters follow the paper's experimental setup:
    - Batch size  : 100
    - Learning rate: 0.001
    - LR decay    : 0.9  (applied per epoch with ExponentialLR)
    - Epochs      : 200
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import numpy as np

from model import AdaptiveBinaryCapsNet
from losses import ABCNLoss


# ─────────────────────────────────────────────────────────────
# Training engine
# ─────────────────────────────────────────────────────────────
class Trainer:
    """
    Manages training and evaluation of the ABCN model.

    Args:
        model        : ABCN model instance.
        train_loader : DataLoader for training data.
        val_loader   : DataLoader for validation data.
        num_classes  : Number of output classes.
        lr           : Initial learning rate (default 0.001).
        lr_decay     : LR multiplicative decay per epoch (default 0.9).
        recon_weight : Weight for reconstruction loss term (default 0.0005).
        device       : 'cuda' | 'cpu' | 'mps'.
        save_dir     : Directory to save checkpoints.
    """

    def __init__(
        self,
        model: AdaptiveBinaryCapsNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 4,
        lr: float = 1e-3,
        lr_decay: float = 0.9,
        recon_weight: float = 0.0005,
        device: str = "cuda",
        save_dir: str = "checkpoints",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = ABCNLoss(recon_weight=recon_weight)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        self.best_val_acc = 0.0

    # ── Single epoch ────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                probs, recon, _ = self.model(images, labels)

                losses = self.criterion(probs, recon, labels, images)
                loss   = losses["total"]

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                preds    = probs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                total_loss += loss.item() * labels.size(0)

        return total_loss / total, correct / total

    # ── Full training loop ──────────────────────────────────────────
    def fit(self, epochs: int = 200, log_interval: int = 1) -> Dict[str, List[float]]:
        """
        Train for ``epochs`` epochs.

        Returns:
            Training history dictionary.
        """
        print(f"Training on {self.device} for {epochs} epochs")
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   training=False)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc" ].append(train_acc)
            self.history["val_loss"  ].append(val_loss)
            self.history["val_acc"   ].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    {"epoch": epoch, "state_dict": self.model.state_dict(),
                     "val_acc": val_acc},
                    self.save_dir / "best_model.pth",
                )

            if epoch % log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch:3d}/{epochs}]  "
                    f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
                    f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  "
                    f"lr {self.scheduler.get_last_lr()[0]:.6f}  "
                    f"({elapsed:.1f}s)"
                )

        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        return self.history

    # ── Load best checkpoint ────────────────────────────────────────
    def load_best(self):
        ckpt = torch.load(self.save_dir / "best_model.pth", map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']}  (val_acc={ckpt['val_acc']:.4f})")


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: AdaptiveBinaryCapsNet,
    loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute accuracy, precision, recall, F1, and AUC on a given DataLoader.

    Args:
        model       : Trained ABCN model.
        loader      : DataLoader (test / val).
        num_classes : Number of classes.
        device      : Compute device.
        class_names : Optional list of class name strings for reporting.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs, _, _ = model(images)

        all_probs .append(probs.cpu().numpy())
        all_preds .append(probs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    y_true  = np.concatenate(all_labels)
    y_pred  = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    acc  = accuracy_score (y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0)

    # Multi-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"  Accuracy  : {acc :.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec :.4f}")
    print(f"  F1-score  : {f1  :.4f}")
    print(f"  AUC (OvR) : {auc :.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    if class_names:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=class_names, zero_division=0))
    print("=" * 60)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1,
        "auc": auc,      "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────
# K-fold cross-validation
# ─────────────────────────────────────────────────────────────
def kfold_cross_validate(
    dataset,
    num_classes: int,
    image_size: int = 28,
    in_channels: int = 3,
    k: int = 5,
    epochs: int = 50,
    batch_size: int = 100,
    lr: float = 1e-3,
    lr_decay: float = 0.9,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation.

    Args:
        dataset     : A torch Dataset (full, no prior split).
        num_classes : Number of output classes.
        image_size  : Input spatial resolution.
        in_channels : Input image channels.
        k           : Number of folds (default 5).
        epochs      : Training epochs per fold.
        batch_size  : Mini-batch size.
        lr          : Learning rate.
        lr_decay    : LR decay per epoch.
        device      : Compute device.
        seed        : Random seed.

    Returns:
        Dictionary mapping metric names to per-fold lists.
    """
    from torch.utils.data import Subset, DataLoader
    from sklearn.model_selection import KFold

    results: Dict[str, List[float]] = {
        "accuracy": [], "precision": [],
        "recall":   [], "f1":        [], "auc": [],
    }

    indices = np.arange(len(dataset))
    kf      = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold + 1} / {k}")
        print(f"{'─'*50}")

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(Subset(dataset, test_idx),
                                  batch_size=batch_size, shuffle=False, num_workers=2)

        model = AdaptiveBinaryCapsNet(
            num_classes=num_classes,
            image_size=image_size,
            in_channels=in_channels,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_classes=num_classes,
            lr=lr,
            lr_decay=lr_decay,
            device=device,
            save_dir=f"checkpoints/fold_{fold}",
        )
        trainer.fit(epochs=epochs, log_interval=10)
        trainer.load_best()

        metrics = evaluate(model, test_loader, num_classes, device)
        for key in results:
            results[key].append(metrics[key])

    # Print summary
    print("\n" + "=" * 60)
    print("  Cross-Validation Summary")
    print("=" * 60)
    for key, vals in results.items():
        arr = np.array(vals)
        ci  = 1.96 * arr.std() / np.sqrt(k)
        print(f"  {key:12s}: {arr.mean():.4f} ± {arr.std():.4f}  "
              f"95% CI [{arr.mean() - ci:.4f}, {arr.mean() + ci:.4f}]")
    print("=" * 60)

    return results
