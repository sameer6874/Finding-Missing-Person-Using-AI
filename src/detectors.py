from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    face_chip: np.ndarray


class FaceDetector:
    """Wrapper around MTCNN with additional filtering."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.7, 0.8, config.detection_threshold],
            min_face_size=config.min_face_size,
            device=config.device if config.device != "cpu" else "cpu",
            post_process=False,
        )

    def _prepare_frame(self, frame: np.ndarray) -> Image.Image:
        # Convert BGR (OpenCV) to RGB for PIL/MTCNN
        return Image.fromarray(frame[:, :, ::-1])

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        img = self._prepare_frame(frame)
        boxes, probs = self.mtcnn.detect(img)

        if boxes is None or probs is None:
            return []

        detections: List[DetectedFace] = []
        h, w, _ = frame.shape
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self.config.detection_threshold:
                continue
            x1 = int(max(0, min(box[0], w - 1)))
            y1 = int(max(0, min(box[1], h - 1)))
            x2 = int(max(0, min(box[2], w)))
            y2 = int(max(0, min(box[3], h)))
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < self.config.min_face_size or crop.shape[1] < self.config.min_face_size:
                continue
            detections.append(DetectedFace(bbox=(x1, y1, x2, y2), score=float(prob), face_chip=crop))
        return detections

