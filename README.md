## Missing Person Video Identification System

An end-to-end computer vision pipeline that ingests CCTV footage, compares each face it finds against a reference portrait of a missing person, and produces a timestamped report plus annotated frames for every high-confidence match.

### Key Capabilities
- Upload one or more reference photos for the missing individual.
- Automatically sample frames from long CCTV videos.
- Detect faces using an MTCNN detector that is robust to pose and lighting.
- Generate FaceNet embeddings and similarity scores.
- Track detections across adjacent frames to avoid duplicate reports.
- Produce CSV/JSON reports and optional annotated preview images.

### Architecture Overview
| Layer | Responsibility | Tech |
| --- | --- | --- |
| Ingestion | Read video streams, sample frames at configurable FPS, cache metadata. | `opencv-python`, `ffmpeg-python` (optional) |
| Detection | Locate faces and bounding boxes per frame. | `facenet-pytorch` MTCNN |
| Embedding | Convert faces to 512-D vectors, normalize for cosine similarity. | `facenet-pytorch` InceptionResnetV1 |
| Matching | Compare to reference embeddings, apply thresholds, temporal smoothing. | NumPy, SciPy |
| Reporting | Store match events, timestamps, bounding boxes, write CSV/JSON + overlays. | Pandas, OpenCV |
| Orchestration | CLI/SDK tying everything together. | Python 3.10+ |

```
          ┌───────────────┐
          │ reference.jpg │
          └───────┬───────┘
                  │ embeddings
        ┌─────────▼────────┐
        │ Matching Engine  │◄───────────┐
        └─────────┬────────┘            │
                  │                     │
           match events           face embeddings
                  │                     │
           ┌──────▼─────┐       ┌───────▼──────┐
           │ Reporter   │◄──────│ Detector/IAE │◄─ video frames
           └──────┬─────┘       └──────────────┘
                  │
        CSV / JSON / annotated frames
```

### Repository Layout
```
├── data/                 # user-provided videos and reference portraits
├── reports/              # generated CSV/JSON summaries & annotated frames
├── src/
│   ├── config.py         # thresholds & runtime configuration
│   ├── detectors.py      # face detection helpers (MTCNN wrapper)
│   ├── embeddings.py     # FaceNet embedding utilities
│   ├── matching.py       # similarity scoring, temporal smoothing
│   ├── reporting.py      # CSV/JSON emitters and visualization
│   ├── video_processing.py # frame sampling & batching
│   └── main.py           # CLI entry point
├── requirements.txt
└── README.md
```

### Setup
1. **Python:** install 3.10 or newer with virtualenv.
2. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data**
   - Put CCTV footage under `data/videos/`.
   - Place reference portrait(s) under `data/references/` (any filename).

### Usage
```bash
python -m src.main \
  --video data/videos/mall_cam_01.mp4 \
  --reference data/references/jane_doe.jpg \
  --output reports/run_2025-12-04
```

#### Key CLI Flags
- `--frame-rate`: sampling rate in frames per second (default 2.0).
- `--match-threshold` / `--maybe-threshold`: cosine similarity cut-offs.
- `--min-face`: ignore faces smaller than this pixel width/height.
- `--device`: `cuda` or `cpu` depending on available hardware.
- `--config-file`: optional JSON file that can override any `RuntimeConfig` field.

Example `config.json`:
```json
{
  "frame_sample_rate": 3.0,
  "match_threshold": 0.84,
  "maybe_threshold": 0.78,
  "temporal_window": 8,
  "temporal_consensus": 4,
  "save_annotated_frames": true,
  "output_stride": 3
}
```

Outputs:
- `matches.csv` – timestamp, frame index, confidence, bounding box.
- `matches.json` – same content plus per-frame metadata.
- `frames/annotated_#####.jpg` – optional annotated stills (configurable).

### Accuracy & Robustness Strategies
- Auto histogram equalization per face crop to stabilize lighting.
- Cosine similarity with dynamic threshold (default 0.82) plus margin for possible matches.
- Temporal smoothing: require N of last M frames to agree before emitting a match.
- Multi-reference support to capture different poses/expressions.
- Pluggable detector/embedding classes so you can swap models (e.g., RetinaFace + ArcFace).

### Future Enhancements
- Integrate Re-ID models for full-body cues when faces are occluded.
- Add GPU-accelerated batching and async video decoding for long recordings.
- Build web dashboard for uploading footage and reviewing matches.
- Hook into notification systems (email/SMS) for urgent alerts.

### Validation Tips
- Use short curated clips before running on multi-hour CCTV feeds.
- Verify detector performance with `--frame-rate 1` to generate quick previews.
- Compare cosine scores for true vs. false positives to fine-tune thresholds.
- Run the pipeline on both bright and dim scenes to calibrate histogram equalization parameters.

### Developer Testing Checklist
1. **Sanity check imports**
   ```bash
   python -m compileall src
   ```
2. **Dry-run CLI**
   ```bash
   python -m src.main --help
   ```
3. **Smoke test** with a short public-domain clip and two portrait photos to validate reporting.
4. Inspect generated CSV/JSON files to ensure timestamps, bounding boxes, and reference IDs are populated.

