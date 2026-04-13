"""
Inference module for single-image defect prediction.

Full pipeline: Load image → Preprocess → Feature Extract → Detect → Classify
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.preprocessing import preprocess_for_dcnn
from src.dcnn_classifier import DualTrackDCNN
from src.dataset import DEFECT_CLASSES, IDX_TO_CLASS

logger = logging.getLogger(__name__)


class DefectPredictor:
    """End-to-end fabric defect classifier."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "dcnn",
        num_classes: int = 12,
        image_size: int = 150,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load model
        if model_type == "dcnn":
            self.model = DualTrackDCNN(num_classes=num_classes, input_size=image_size)
        else:
            from src.feature_extraction import ResNet50Classifier
            self.model = ResNet50Classifier(num_classes=num_classes)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded %s model from %s (val_acc=%.2f%%)",
            model_type, checkpoint_path,
            checkpoint.get("val_acc", 0),
        )

    def predict(self, image_path: str) -> dict:
        """
        Predict defect type for a single image.

        Returns:
            {class_name, class_idx, confidence, top_3: [(name, prob), ...]}
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess
        processed = preprocess_for_dcnn(image, self.image_size)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Transform to tensor
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        top_probs, top_indices = probs.topk(3, dim=1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()

        predicted_class = IDX_TO_CLASS[top_indices[0]]
        confidence = float(top_probs[0])

        return {
            "class_name": predicted_class,
            "class_idx": int(top_indices[0]),
            "confidence": confidence,
            "top_3": [
                (IDX_TO_CLASS[idx], float(prob))
                for idx, prob in zip(top_indices, top_probs)
            ],
        }

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        """Predict defect types for multiple images."""
        return [self.predict(path) for path in image_paths]


def run_demo():
    """Run inference demo on sample images."""
    import argparse
    parser = argparse.ArgumentParser(description="Fabric defect inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="checkpoints/best_dcnn.pt")
    parser.add_argument("--model", choices=["dcnn", "resnet50"], default="dcnn")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    predictor = DefectPredictor(args.checkpoint, args.model)
    result = predictor.predict(args.image)

    print(f"\nPrediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3:")
    for name, prob in result["top_3"]:
        print(f"  {name}: {prob:.2%}")


if __name__ == "__main__":
    run_demo()
