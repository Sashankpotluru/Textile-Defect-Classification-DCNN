"""
Training script for the fabric defect classification models.

Implements the training pipeline as described in Section 3 of the paper:
1. Load and preprocess dataset
2. Train DCNN / ResNet-50 classifier
3. Evaluate on validation set each epoch
4. Save best model checkpoint
5. Generate training curves (accuracy + loss)

Paper results: 96.29% accuracy after 100 epochs.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dcnn_classifier import DualTrackDCNN, EnhancedDCNN
from src.feature_extraction import ResNet50Classifier
from src.dataset import (
    create_data_loaders,
    generate_synthetic_dataset,
    DEFECT_CLASSES,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for fabric defect classifiers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 15,
        model_name: str = "dcnn",
    ) -> dict:
        """
        Train the model and return training history.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            model_name: Name for saving checkpoints

        Returns:
            Training history dict
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        best_val_acc = 0.0
        no_improve = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100. * train_correct / train_total:.1f}%",
                })

            scheduler.step()

            train_loss /= train_total
            train_acc = 100. * train_correct / train_total

            # Validation phase
            val_loss, val_acc = self._evaluate(val_loader, criterion)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %d/%d — Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, self.checkpoint_dir / f"best_{model_name}.pt")
                logger.info("Saved best model (val_acc=%.2f%%)", val_acc)
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        total_time = time.time() - start_time
        logger.info("Training complete in %.1fs. Best val acc: %.2f%%", total_time, best_val_acc)

        # Save training curves
        self._plot_training_curves(model_name)

        return self.history

    def _evaluate(
        self, data_loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        return avg_loss, accuracy

    def _plot_training_curves(self, model_name: str):
        """Generate training accuracy and loss plots (Figures 10 & 11)."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Accuracy plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_acc"], "b-", label="Training Accuracy")
        ax.plot(epochs, self.history["val_acc"], "r-", label="Validation Accuracy")
        ax.set_xlabel("Number of Epochs")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Train acc vs Val acc")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.results_dir / f"{model_name}_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_loss"], "b-", label="Training Loss")
        ax.plot(epochs, self.history["val_loss"], "r-", label="Validation Loss")
        ax.set_xlabel("Number of Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.results_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Training curves saved to %s", self.results_dir)


def train_dcnn(
    data_dir: str = "data/raw",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    image_size: int = 150,
    use_synthetic: bool = False,
):
    """Train the proposed DCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if use_synthetic:
        logger.info("Generating synthetic dataset for demo...")
        generate_synthetic_dataset(data_dir, num_per_class=50)

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, image_size=image_size, batch_size=batch_size, num_workers=0,
    )

    model = DualTrackDCNN(num_classes=len(DEFECT_CLASSES), input_size=image_size)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    trainer = Trainer(model, device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr, model_name="dcnn",
    )

    return history, trainer


def train_resnet50(
    data_dir: str = "data/raw",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0001,
    image_size: int = 224,
    use_synthetic: bool = False,
):
    """Train the ResNet-50 baseline model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if use_synthetic:
        generate_synthetic_dataset(data_dir, num_per_class=50)

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, image_size=image_size, batch_size=batch_size, num_workers=0,
    )

    model = ResNet50Classifier(num_classes=len(DEFECT_CLASSES), pretrained=True)
    trainer = Trainer(model, device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr, model_name="resnet50",
    )

    return history, trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Train fabric defect classifier")
    parser.add_argument("--model", choices=["dcnn", "resnet50", "enhanced"], default="dcnn")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demo")
    args = parser.parse_args()

    if args.model == "dcnn":
        train_dcnn(args.data_dir, args.epochs, args.batch_size, args.lr, use_synthetic=args.synthetic)
    elif args.model == "resnet50":
        train_resnet50(args.data_dir, args.epochs, args.batch_size, args.lr, use_synthetic=args.synthetic)
    elif args.model == "enhanced":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if args.synthetic:
            generate_synthetic_dataset(args.data_dir, num_per_class=50)
        train_loader, val_loader, _ = create_data_loaders(args.data_dir, batch_size=args.batch_size, num_workers=0)
        model = EnhancedDCNN(num_classes=len(DEFECT_CLASSES))
        trainer = Trainer(model, device)
        trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr, model_name="enhanced_dcnn")
