#!/usr/bin/env python3
"""Reproducible talking-head benchmark runner and report validator.

Metrics are collected by pinned evaluator containers supplied by the operator. This
module enforces the benchmark contract, records comparable runs, and computes
backend deltas without ingesting biometric source data into version control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_METRICS = (
    "lip_sync_offset_ms",
    "identity_similarity",
    "flicker_score",
    "pose_expression_drift",
    "detail_score",
    "output_width",
    "output_height",
    "inference_seconds",
    "peak_vram_mib",
)
LOWER_IS_BETTER = {
    "lip_sync_offset_ms",
    "flicker_score",
    "pose_expression_drift",
    "inference_seconds",
    "peak_vram_mib",
}


@dataclass(frozen=True)
class Threshold:
    direction: str
    value: float


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {"dataset_id", "consent_record", "split", "items"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["split"] != "held_out":
        raise ValueError("benchmark manifest must be a held_out split")
    if not manifest["consent_record"]:
        raise ValueError("manifest must point to a consent record")
    for item in manifest["items"]:
        if {"id", "portrait_sha256", "speech_sha256"} - item.keys():
            raise ValueError(f"incomplete item: {item.get('id', '<unknown>')}")


def load_thresholds(path: Path) -> dict[str, Threshold]:
    raw = load_json(path)["metrics"]
    return {name: Threshold(**spec) for name, spec in raw.items()}


def validate_rows(rows: list[dict[str, str]], manifest: dict[str, Any]) -> None:
    expected = {item["id"] for item in manifest["items"]}
    seen = {row.get("sample_id", "") for row in rows}
    if seen != expected:
        raise ValueError(
            f"sample IDs must match manifest; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    for row in rows:
        missing = [metric for metric in REQUIRED_METRICS if not row.get(metric)]
        if missing:
            raise ValueError(f"{row['sample_id']} missing metrics: {missing}")
        for metric in REQUIRED_METRICS:
            float(row[metric])


def averages(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in REQUIRED_METRICS
    }


def assess(
    values: dict[str, float], thresholds: dict[str, Threshold]
) -> dict[str, str]:
    result = {}
    for metric, threshold in thresholds.items():
        value = values[metric]
        ok = (
            value <= threshold.value
            if threshold.direction == "max"
            else value >= threshold.value
        )
        result[metric] = "pass" if ok else "fail"
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report(args: argparse.Namespace) -> None:
    manifest = load_json(Path(args.manifest))
    validate_manifest(manifest)
    thresholds = load_thresholds(Path(args.thresholds))
    baseline, candidate = read_csv(Path(args.baseline)), read_csv(Path(args.candidate))
    validate_rows(baseline, manifest)
    validate_rows(candidate, manifest)
    base_avg, cand_avg = averages(baseline), averages(candidate)
    delta = {metric: cand_avg[metric] - base_avg[metric] for metric in REQUIRED_METRICS}
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": canonical_hash(Path(args.manifest)),
        "baseline_backend": "ComfyUI-LivePortraitKJ",
        "candidate_backend": "HeyGem",
        "identical_input_contract": {
            "portrait_hashes": sorted(x["portrait_sha256"] for x in manifest["items"]),
            "speech_hashes": sorted(x["speech_sha256"] for x in manifest["items"]),
            "output_settings": load_json(Path(args.settings)),
        },
        "baseline_mean": base_avg,
        "candidate_mean": cand_avg,
        "candidate_minus_baseline": delta,
        "candidate_thresholds": assess(cand_avg, thresholds),
        "overall": "pass"
        if all(v == "pass" for v in assess(cand_avg, thresholds).values())
        else "fail",
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate comparable talking-head benchmark results."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    report(parser.parse_args())


if __name__ == "__main__":
    main()
