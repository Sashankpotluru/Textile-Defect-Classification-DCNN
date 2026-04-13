"""
YOLOv7-tiny Detection Module for Fabric Defect Localization.

Uses YOLOv7-tiny (via Ultralytics) to detect and localize defect regions
in fabric images before classification, as described in Section 3.3.

The detection pipeline:
1. Backbone (E-ELAN): 16 layers, 3x3 conv with 16 filters → feature map
2. Neck (FPN): Feature pyramid for multi-scale detection
3. Head: Bounding box prediction with class confidence

We use Ultralytics YOLOv8-nano as a modern equivalent of YOLOv7-tiny,
with the same architectural principles (E-ELAN, CSP, etc.).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class FabricDefectDetector:
    """
    YOLO-based detector for localizing defect regions in fabric images.

    For training: requires annotated dataset in YOLO format.
    For inference: loads a trained model and returns bounding boxes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "yolov8n",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info("Loaded trained YOLO model from %s", model_path)
            else:
                self.model = YOLO(f"{model_size}.pt")
                logger.info("Loaded pretrained %s model", model_size)
        except ImportError:
            logger.warning("ultralytics not installed, detector unavailable")
            self.model = None

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect defect regions in a fabric image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections: [{bbox: [x1,y1,x2,y2], confidence, class_id, class_name}]
        """
        if self.model is None:
            return []

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                })

        return detections

    def extract_roi(
        self, image: np.ndarray, bbox: list[float], padding: int = 10
    ) -> np.ndarray:
        """Extract a region of interest from detected bounding box."""
        h, w = image.shape[:2]
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        return image[y1:y2, x1:x2]

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        image_size: int = 640,
        batch_size: int = 16,
        project: str = "runs/detect",
        name: str = "fabric_defect",
    ) -> str:
        """
        Train the YOLO detector on fabric defect dataset.

        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            image_size: Training image size
            batch_size: Batch size
            project: Output project directory
            name: Experiment name

        Returns:
            Path to best model weights
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=project,
            name=name,
            patience=20,
            save=True,
            plots=True,
        )
        best_path = Path(project) / name / "weights" / "best.pt"
        return str(best_path)

    def visualize_detections(
        self, image: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Draw bounding boxes on the image."""
        vis = image.copy()
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.get('class_name', 'defect')}: {det['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis
