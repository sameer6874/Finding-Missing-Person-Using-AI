from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MatchObservation:
    score: float
    frame_idx: int
    timestamp: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class MatchEvent:
    reference_id: int
    reference_path: str
    score: float
    frame_idx: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    label: str = field(default="confirmed")


class TemporalConsensus:
    """Maintains sliding-window consensus for each reference."""

    def __init__(self, window: int, min_hits: int):
        self.window = window
        self.min_hits = min_hits
        self.buffers: Dict[int, Deque[MatchObservation]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, ref_id: int, observation: Optional[MatchObservation]) -> Optional[MatchObservation]:
        buffer = self.buffers[ref_id]
        if observation:
            buffer.append(observation)
        else:
            buffer.append(MatchObservation(score=0.0, frame_idx=-1, timestamp=-1.0, bbox=(0, 0, 0, 0)))

        hits = [obs for obs in buffer if obs.score > 0]
        if len(hits) >= self.min_hits:
            best = max(hits, key=lambda obs: obs.score)
            buffer.clear()
            return best
        return None


class FaceMatcher:
    def __init__(self, references: List[np.ndarray], reference_paths: List[str], config: RuntimeConfig):
        self.references = np.stack(references) if references else np.empty((0, 512), dtype=np.float32)
        self.reference_paths = reference_paths
        self.config = config
        self.temporal = TemporalConsensus(config.temporal_window, config.temporal_consensus)

    def _cosine_similarity(self, embedding: np.ndarray) -> np.ndarray:
        if self.references.size == 0:
            return np.array([])
        return np.dot(self.references, embedding)

    def match(
        self,
        embedding: np.ndarray,
        frame_idx: int,
        timestamp: float,
        bbox: tuple[int, int, int, int],
    ) -> List[MatchEvent]:
        scores = self._cosine_similarity(embedding)
        events: List[MatchEvent] = []

        if scores.size == 0:
            return events

        for idx, score in enumerate(scores):
            if score < self.config.maybe_threshold:
                self.temporal.update(idx, None)
                continue

            observation = MatchObservation(score=score, frame_idx=frame_idx, timestamp=timestamp, bbox=bbox)

            if score >= self.config.match_threshold:
                consensus = self.temporal.update(idx, observation)
                if consensus:
                    events.append(
                        MatchEvent(
                            reference_id=idx,
                            reference_path=self.reference_paths[idx],
                            score=consensus.score,
                            frame_idx=consensus.frame_idx,
                            timestamp=consensus.timestamp,
                            bbox=consensus.bbox,
                            label="confirmed",
                        )
                    )
            else:
                # possible match but not fully confirmed
                events.append(
                    MatchEvent(
                        reference_id=idx,
                        reference_path=self.reference_paths[idx],
                        score=score,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        bbox=bbox,
                        label="candidate",
                    )
                )
        return events

