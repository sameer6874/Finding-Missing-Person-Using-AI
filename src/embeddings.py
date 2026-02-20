from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image, ImageEnhance
from facenet_pytorch import InceptionResnetV1, MTCNN

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


def _preprocess_image(img: Image.Image, enhance: bool = True) -> Image.Image:
    if not enhance:
        return img
    img = ImageEnhance.Color(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    return img


@dataclass(slots=True)
class EmbeddingResult:
    vector: np.ndarray
    source_path: Path | None = None
    score: float | None = None


class FaceEmbedder:
    """Generates normalized embeddings using FaceNet (InceptionResnetV1)."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device != "cpu" else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.reference_aligner = MTCNN(device=self.device, keep_all=False) if config.reference_augmentation else None

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    def embed_faces(self, faces: List[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.empty((0, 512), dtype=np.float32)

        tensors = []
        for face in faces:
            img = Image.fromarray(face[:, :, ::-1])
            img = _preprocess_image(img)
            tensor = self._to_tensor(img)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed_reference_images(self, paths: Iterable[Path]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for idx, path in enumerate(paths):
            if idx >= self.config.max_reference_images:
                LOGGER.warning("Reached reference image cap (%s)", self.config.max_reference_images)
                break
            img = Image.open(path).convert("RGB")
            if self.reference_aligner is not None:
                aligned = self.reference_aligner(img)
                if aligned is not None:
                    tensor = aligned.detach().cpu()
                    img = tensor.permute(1, 2, 0).mul(255).byte().numpy()
                    img = Image.fromarray(img)
            vector = self.embed_faces([np.array(img)[:, :, ::-1]])  # convert to BGR-like layout
            if vector.size == 0:
                continue
            results.append(EmbeddingResult(vector=vector[0], source_path=path))
        return results

