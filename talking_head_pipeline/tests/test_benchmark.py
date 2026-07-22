import unittest

from talking_head_pipeline.benchmarks.benchmark import (
    averages,
    validate_manifest,
    validate_rows,
)

METRICS = [
    "lip_sync_offset_ms",
    "identity_similarity",
    "flicker_score",
    "pose_expression_drift",
    "detail_score",
    "output_width",
    "output_height",
    "inference_seconds",
    "peak_vram_mib",
]


class BenchmarkContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "dataset_id": "set-v1",
            "consent_record": "private://consent/1",
            "split": "held_out",
            "items": [{"id": "one", "portrait_sha256": "p", "speech_sha256": "s"}],
        }
        self.row = {"sample_id": "one", **{name: "1" for name in METRICS}}

    def test_manifest_requires_held_out_consent_and_hashes(self):
        validate_manifest(self.manifest)
        bad = dict(self.manifest, split="train")
        with self.assertRaisesRegex(ValueError, "held_out"):
            validate_manifest(bad)

    def test_rows_require_all_metrics_and_exact_sample_set(self):
        validate_rows([self.row], self.manifest)
        missing = dict(self.row)
        del missing["detail_score"]
        with self.assertRaisesRegex(ValueError, "missing metrics"):
            validate_rows([missing], self.manifest)

    def test_average_is_numeric(self):
        self.assertEqual(averages([self.row])["peak_vram_mib"], 1.0)


if __name__ == "__main__":
    unittest.main()
