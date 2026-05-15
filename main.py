"""
Main entry point for training and evaluating the ABCN model.

Usage examples:

    # Train on ocular disease dataset (4 classes):
    python main.py --task ocular --data_dir data/ocular --epochs 200

    # Train on CIFAR-10 (10 classes):
    python main.py --task cifar10 --epochs 200

    # Run 5-fold cross-validation:
    python main.py --task ocular --data_dir data/ocular --kfold --epochs 50

    # Evaluate a saved checkpoint:
    python main.py --task ocular --data_dir data/ocular --evaluate --ckpt checkpoints/best_model.pth
"""

import argparse
import torch

from model import AdaptiveBinaryCapsNet
from losses import ABCNLoss
from train import Trainer, evaluate, kfold_cross_validate
from datasets import load_ocular_dataset, load_cifar10


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Binary Capsule Network")

    p.add_argument("--task",       default="ocular",
                   choices=["ocular", "cifar10"],
                   help="Dataset to train on.")
    p.add_argument("--data_dir",   default="data/ocular",
                   help="Root folder for the ocular dataset.")
    p.add_argument("--image_size", type=int, default=28,
                   help="Spatial resize target (default 28).")
    p.add_argument("--batch_size", type=int, default=100,
                   help="Mini-batch size (default 100).")
    p.add_argument("--epochs",     type=int, default=200,
                   help="Training epochs (default 200).")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Initial learning rate (default 0.001).")
    p.add_argument("--lr_decay",   type=float, default=0.9,
                   help="Exponential LR decay per epoch (default 0.9).")
    p.add_argument("--recon_weight", type=float, default=0.0005,
                   help="Reconstruction loss weight (default 0.0005).")
    p.add_argument("--save_dir",   default="checkpoints",
                   help="Directory to save model checkpoints.")
    p.add_argument("--kfold",      action="store_true",
                   help="Run 5-fold cross-validation instead of a single train run.")
    p.add_argument("--evaluate",   action="store_true",
                   help="Load a checkpoint and run evaluation only.")
    p.add_argument("--ckpt",       default=None,
                   help="Path to checkpoint file for --evaluate mode.")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device (default: cuda if available, else cpu).")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'='*60}")
    print(f"  Adaptive Binary Capsule Network  (ABCN)")
    print(f"  Task   : {args.task}")
    print(f"  Device : {args.device}")
    print(f"{'='*60}\n")

    # ── Resolve task-specific settings ──────────────────────────────
    if args.task == "cifar10":
        num_classes   = 10
        image_size    = args.image_size if args.image_size != 28 else 32
        class_names   = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                         "Dog", "Frog", "Horse", "Ship", "Truck"]
        train_loader, val_loader, test_loader = load_cifar10(
            image_size=image_size, batch_size=args.batch_size
        )
    else:                                                   # ocular
        num_classes   = 4
        image_size    = args.image_size
        class_names   = ["Glaucoma-Positive", "Glaucoma-Negative",
                         "Cataract-Positive", "Cataract-Negative"]
        train_loader, val_loader, test_loader = load_ocular_dataset(
            data_dir   = args.data_dir,
            image_size = image_size,
            batch_size = args.batch_size,
        )

    # ── Evaluate only ────────────────────────────────────────────────
    if args.evaluate:
        if args.ckpt is None:
            raise ValueError("Please provide --ckpt path for evaluation mode.")
        model = AdaptiveBinaryCapsNet(
            num_classes=num_classes,
            image_size=image_size,
        )
        ckpt  = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(args.device)
        print(f"Loaded checkpoint: {args.ckpt}")
        evaluate(model, test_loader, num_classes, args.device, class_names)
        return

    # ── K-fold cross-validation ───────────────────────────────────────
    if args.kfold:
        # Combine train + val for k-fold
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([
            train_loader.dataset,
            val_loader.dataset,
        ])
        kfold_cross_validate(
            dataset     = full_dataset,
            num_classes = num_classes,
            image_size  = image_size,
            k           = 5,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            lr_decay    = args.lr_decay,
            device      = args.device,
            seed        = args.seed,
        )
        return

    # ── Standard training run ─────────────────────────────────────────
    model = AdaptiveBinaryCapsNet(
        num_classes=num_classes,
        image_size=image_size,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}\n")

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_classes  = num_classes,
        lr           = args.lr,
        lr_decay     = args.lr_decay,
        recon_weight = args.recon_weight,
        device       = args.device,
        save_dir     = args.save_dir,
    )

    history = trainer.fit(epochs=args.epochs)
    trainer.load_best()

    print("\nTest set evaluation:")
    evaluate(model, test_loader, num_classes, args.device, class_names)

    # Optionally save training curves
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_acc"], label="Train")
        ax1.plot(history["val_acc"],   label="Val")
        ax1.set_title("Accuracy");  ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(history["train_loss"], label="Train")
        ax2.plot(history["val_loss"],   label="Val")
        ax2.set_title("Loss");      ax2.set_xlabel("Epoch"); ax2.legend()
        plt.tight_layout()
        plt.savefig(f"{args.save_dir}/training_curves.png")
        print(f"Training curves saved to {args.save_dir}/training_curves.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
