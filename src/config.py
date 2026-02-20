from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class RuntimeConfig:
    """Holds tunable parameters for the detection pipeline."""

    frame_sample_rate: float = 2.0  # frames per second to sample from video
    min_face_size: int = 60
    detection_threshold: float = 0.90
    match_threshold: float = 0.82
    maybe_threshold: float = 0.75
    temporal_window: int = 6
    temporal_consensus: int = 3
    save_annotated_frames: bool = True
    max_reference_images: int = 5
    device: str = "cuda"
    reference_augmentation: bool = True
    output_stride: int = 5  # write annotated frame every Nth match
    max_video_seconds: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeConfig":
        base = asdict(cls())
        base.update(data)
        return cls(**base)


@dataclass(slots=True)
class Paths:
    video_path: Path
    reference_path: Path
    output_dir: Path

    def ensure_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ProjectMetadata:
    case_id: str
    operator: str = "unknown"
    notes: str = ""
    reference_description: str = ""
    video_source: str = ""
    timezone: str = "UTC"
    confidence_units: str = "cosine"
    extra: dict[str, str] = field(default_factory=dict)

