from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FramePacket:
    frame_idx: int
    timestamp_s: float
    frame: "cv2.Mat"


class FrameSampler:
    def __init__(self, video_path: Path, sample_rate: float, max_seconds: Optional[int] = None):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def iterate(self) -> Generator[FramePacket, None, None]:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(int(fps / self.sample_rate), 1)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_idx = 0
        sampled = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp_s = frame_idx / fps
            if self.max_seconds and timestamp_s > self.max_seconds:
                LOGGER.info("Reached max_seconds (%s), stopping.", self.max_seconds)
                break

            sampled += 1
            yield FramePacket(frame_idx=frame_idx, timestamp_s=timestamp_s, frame=frame)
            frame_idx += 1

        LOGGER.info("Processed %s/%s frames (~%.2f%%)", sampled, total_frames, sampled / max(total_frames, 1) * 100)
        capture.release()

