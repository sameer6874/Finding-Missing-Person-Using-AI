from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import pandas as pd

from .config import ProjectMetadata
from .matching import MatchEvent

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReportArtifact:
    path: Path
    description: str


@dataclass(slots=True)
class MatchRecord:
    event: MatchEvent
    artifact_path: Path | None = None


class MatchReporter:
    def __init__(
        self,
        output_dir: Path,
        save_frames: bool,
        output_stride: int = 5,
        metadata: ProjectMetadata | None = None,
    ):
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.logs_dir = output_dir / "logs"
        self.save_frames = save_frames
        self.output_stride = max(output_stride, 1)
        self.records: List[MatchRecord] = []
        self.metadata = metadata

    def _annotate_frame(self, frame, bbox, label: str, score: float) -> Path:
        x1, y1, x2, y2 = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}:{score:.2f}"
        cv2.putText(annotated, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = self.frames_dir / f"match_{len(self.records):05d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        return out_path

    def record(self, event: MatchEvent, frame) -> None:
        artifact = None
        if self.save_frames and event.label == "confirmed" and len(self.records) % self.output_stride == 0:
            artifact = self._annotate_frame(frame, event.bbox, event.label, event.score)
        self.records.append(MatchRecord(event=event, artifact_path=artifact))

    def flush(self) -> List[ReportArtifact]:
        if not self.records:
            LOGGER.warning("No matches to report.")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for rec in self.records:
            data = asdict(rec.event)
            data["artifact_path"] = str(rec.artifact_path) if rec.artifact_path else ""
            rows.append(data)

        df = pd.DataFrame(rows)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = self.logs_dir / f"matches_{timestamp}.csv"
        json_path = self.logs_dir / f"matches_{timestamp}.json"

        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)

        if self.metadata:
            meta_path = self.logs_dir / f"metadata_{timestamp}.json"
            meta_path.write_text(json.dumps(asdict(self.metadata), indent=2), encoding="utf-8")

        LOGGER.info("Report written to %s", csv_path)
        return [
            ReportArtifact(path=csv_path, description="Tabular match summary"),
            ReportArtifact(path=json_path, description="JSON match summary"),
        ]

