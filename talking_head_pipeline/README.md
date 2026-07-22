# Talking-head backend benchmark suite

This directory is the reproducible, privacy-preserving evaluation protocol for the selected talking-head backend. It **does not contain portraits, speech, consent records, generated videos, or embeddings**. Run it only with a consented, held-out asset set stored in approved private storage. Do not reuse a participant's assets outside the consent scope or use any sample that was used for tuning/training.

## Comparison contract

The baseline is `ComfyUI-LivePortraitKJ`; the candidate is `HeyGem`. Every row in both CSV files must correspond to the same `sample_id` in one immutable held-out manifest and must use the pinned output settings in `config/output_settings.json`: 25 fps, 512 px height, H.264/yuv420p, 16 kHz audio, and seed `20260722`. Record the backend image/commit, workflow JSON SHA-256, GPU/driver, evaluator container digest, command line, start/end UTC timestamps, and raw profiler logs next to the private run artifacts.

1. Copy `config/held_out_manifest.example.json` to a private location and replace hashes with SHA-256 values of the approved assets. The consent record URI must resolve for the review team.
2. Run the existing **ComfyUI-LivePortraitKJ** workflow once per sample with no manual retouching. Warm up once, then measure the next run from process invocation through output muxing; sample VRAM with `nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits` every 250 ms and retain the maximum.
3. Render **HeyGem** using exactly the same portrait/audio bytes and output settings. Do not crop, re-time, enhance, or choose a different seed per backend. If HeyGem cannot accept a setting, mark that sample `not_comparable` and exclude it from both systems rather than silently substituting an output.
4. Run the pinned metric evaluators for each output and write one row per sample to `baseline.csv` and `heygem.csv`. Names and units are fixed below.
5. Generate the comparison report. A report is a gate, not a claim that either model is production-ready.

```bash
python -m unittest discover -s talking_head_pipeline/tests -v
python talking_head_pipeline/benchmarks/benchmark.py \
  --manifest /secure/held_out_manifest.json \
  --settings talking_head_pipeline/config/output_settings.json \
  --thresholds talking_head_pipeline/config/thresholds.json \
  --baseline /secure/runs/liveportraitkj/baseline.csv \
  --candidate /secure/runs/heygem/heygem.csv \
  --output /secure/runs/comparison.json
```

## CSV metric contract

Required columns are `sample_id` plus all fields below. Evaluators must be version-pinned and use the same reference assets for both runs.

| Field | Direction | Method / unit |
| --- | --- | --- |
| `lip_sync_offset_ms` | lower | absolute audio–mouth timing offset (ms), from a pinned SyncNet-style evaluator |
| `identity_similarity` | higher | cosine similarity of aligned-face embeddings to the source portrait |
| `flicker_score` | lower | temporal residual after optical-flow warping consecutive face crops |
| `pose_expression_drift` | lower | normalized temporal variance of head-pose and expression landmarks after subtracting expected speech motion |
| `detail_score` | higher | no-reference face-crop detail score (pinned MUSIQ/LPIPS-equivalent evaluator) |
| `output_width`, `output_height` | higher | decoded output dimensions in pixels |
| `inference_seconds` | lower | wall-clock seconds excluding one warm-up run |
| `peak_vram_mib` | lower | peak GPU process memory during the measured render |

The evaluator must fail a sample when no face is detected, audio duration differs by more than one frame, or a decoded video is corrupt. Never fill missing values with zero.

## Promotion decision

`config/thresholds.json` gates the candidate's aggregate means. In addition, promotion requires: no consent/data-governance exception; 100% evaluable held-out samples; no material regression versus baseline (review any change beyond 5% for quality metrics or 10% for speed/VRAM); a reviewer inspection of the worst 10% samples for artifacts; and reproducibility from a fresh environment. The generated report includes means, candidate-minus-baseline deltas, hashes, and gate results. Save immutable raw artifacts privately; commit only non-sensitive configuration and aggregate reports approved for publication.

## HeyGem training policy

**Do not begin a HeyGem training run until its supported training interface and license have been verified.** If it supports training, use the following plan before creating a checkpoint:

* Create separate, immutable `train.jsonl` and `validation.jsonl` manifests keyed by participant/session, never by individual frames. Each line must include `sample_id`, private portrait/video URI, speech URI, SHA-256 hashes, consent scope/expiry, language, duration, and split. Keep the held-out manifest completely disjoint by participant.
* Freeze and record augmentations: horizontal flip only when semantically valid; color jitter, JPEG/resample, crop, mild blur/noise, and audio gain/noise within predeclared ranges. Apply no identity-changing, lip-timing-changing, or unconsented synthetic augmentation. Seed every transform and log the augmentation policy hash.
* Split by identity/session (recommended 80/10/10 train/validation/held-out) and stratify language, pose, lighting, duration, and presentation attributes. Review split leakage using hashes and face-embedding near-duplicate checks.
* Name checkpoints `heygem-<dataset_version>-<git_sha>-step<global_step>-val<metric>.safetensors`; retain optimizer/config/state sidecars and never overwrite a promoted artifact.
* Evaluate at step 0 and every 5,000 optimizer steps (and final) on the fixed validation manifest. Run the complete held-out suite only for a promotion candidate, never to tune hyperparameters.
* Promote only if validation improves for two consecutive evaluations, all benchmark gates above pass, quality has no material regression against the LivePortraitKJ baseline, safety/consent review approves it, and an independent reviewer signs the immutable model card and artifact hashes.
