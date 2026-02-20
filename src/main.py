from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer
from tqdm import tqdm

from .config import Paths, ProjectMetadata, RuntimeConfig
from .detectors import FaceDetector
from .embeddings import FaceEmbedder
from .matching import FaceMatcher
from .reporting import MatchReporter
from .video_processing import FrameSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("missing-person")

app = typer.Typer(add_completion=False, help="Missing person detection pipeline.")


def _load_reference_paths(reference: Path) -> List[Path]:
    if reference.is_dir():
        candidates = sorted([p for p in reference.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        candidates = [reference]
    if not candidates:
        raise FileNotFoundError(f"No reference images found at {reference}")
    return candidates


@app.command()
def run(
    video: Path = typer.Option(..., exists=True, readable=True, help="Input CCTV video file path."),
    reference: Path = typer.Option(..., exists=True, help="Reference portrait or directory of portraits."),
    output: Path = typer.Option(Path("reports/latest"), help="Directory for reports & artifacts."),
    case_id: str = typer.Option("case-001", help="Identifier for the investigation."),
    operator: str = typer.Option("analyst", help="Operator name."),
    frame_rate: float = typer.Option(2.0, help="Frames per second to sample."),
    match_threshold: float = typer.Option(0.82, help="Cosine threshold for confirmed matches."),
    maybe_threshold: float = typer.Option(0.75, help="Cosine threshold for candidate matches."),
    min_face: int = typer.Option(60, help="Minimum face size in pixels."),
    detection_threshold: float = typer.Option(0.9, help="Confidence threshold for detector."),
    device: str = typer.Option("cuda", help="Device to run models on (cuda/cpu)."),
    config_file: Optional[Path] = typer.Option(None, help="Optional JSON config override."),
):
    """Run the end-to-end face search on a video file."""

    config = RuntimeConfig(
        frame_sample_rate=frame_rate,
        min_face_size=min_face,
        detection_threshold=detection_threshold,
        match_threshold=match_threshold,
        maybe_threshold=maybe_threshold,
        device=device,
    )

    if config_file:
        overrides = json.loads(Path(config_file).read_text(encoding="utf-8"))
        config = RuntimeConfig.from_dict(overrides)

    paths = Paths(video_path=video, reference_path=reference, output_dir=output)
    paths.ensure_output()
    metadata = ProjectMetadata(case_id=case_id, operator=operator, video_source=str(video))
    LOGGER.info("Starting case %s by %s", metadata.case_id, metadata.operator)

    reference_paths = _load_reference_paths(paths.reference_path)
    embedder = FaceEmbedder(config)
    reference_embeddings = embedder.embed_reference_images(reference_paths)
    if not reference_embeddings:
        raise RuntimeError("Unable to produce embeddings for reference images.")

    matcher = FaceMatcher(
        references=[res.vector for res in reference_embeddings],
        reference_paths=[str(res.source_path) for res in reference_embeddings],
        config=config,
    )

    detector = FaceDetector(config)
    reporter = MatchReporter(
        paths.output_dir,
        config.save_annotated_frames,
        config.output_stride,
        metadata=metadata,
    )
    sampler = FrameSampler(video, sample_rate=config.frame_sample_rate, max_seconds=config.max_video_seconds)

    total_matches = 0
    for packet in tqdm(sampler.iterate(), desc="Scanning video"):
        faces = detector.detect(packet.frame)
        if not faces:
            continue
        embeddings = embedder.embed_faces([face.face_chip for face in faces])
        for face, embedding in zip(faces, embeddings):
            events = matcher.match(embedding, packet.frame_idx, packet.timestamp_s, face.bbox)
            if not events:
                continue
            for event in events:
                if event.label == "confirmed":
                    total_matches += 1
                reporter.record(event, packet.frame)

    artifacts = reporter.flush()
    LOGGER.info("Pipeline finished, %s confirmed matches.", total_matches)
    if artifacts:
        for art in artifacts:
            LOGGER.info("Artifact: %s -> %s", art.description, art.path)


if __name__ == "__main__":
    app()

