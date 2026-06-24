from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cryptomamba_ui.data import MODEL_WINDOW_SIZE, window_from_candles
from src.cryptomamba_ui.predict_artifacts import (
    OfflinePredictionError,
    load_offline_prediction,
)


def _candles(n: int = MODEL_WINDOW_SIZE) -> list[dict]:
    base = 26000.0
    return [
        {
            "date": f"2023-09-{i + 1:02d}",
            "open": base + i,
            "high": base + i + 50,
            "low": base + i - 50,
            "close": base + i + 10,
            "volume": 6_000_000_000.0 + i,
        }
        for i in range(n)
    ]


def _artifact(**overrides) -> dict:
    artifact = {
        "artifact_type": "offline_prediction",
        "warning": "Frozen offline prediction. NOT a live API call.",
        "generated_at_utc": "2026-06-21T00:00:00+00:00",
        "checkpoint_sha256": "ad5ec21bb2582e1f935837f620ed8d1ecb280e5568571783bd8c89ab717cc510",
        "source_commit": "894d0eb",
        "payload": {
            "prediction_date": "2023-10-01",
            "risk": 2.0,
            "candles": _candles(),
        },
        "response": {
            "model_id": "cmamba_v",
            "inference_type": "live",
            "checkpoint_sha256": "ad5ec21bb2582e1f935837f620ed8d1ecb280e5568571783bd8c89ab717cc510",
            "source_commit": "894d0eb",
            "prediction_date": "2023-10-01",
            "last_close": 26967.0,
            "predicted_close": 26967.7148,
            "vanilla_action": "buy",
            "smart_action": "hold",
            "smart_pct": 0.0,
        },
    }
    artifact.update(overrides)
    return artifact


class LoadOfflinePredictionTest(unittest.TestCase):
    def _write(self, artifact: object) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(artifact, tmp)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_valid_artifact_forces_offline_label(self):
        bundle = load_offline_prediction(self._write(_artifact()))
        # FE-override: a frozen 'live' response must surface as 'offline', not 'live'.
        self.assertEqual(bundle["prediction"]["inference_type"], "offline")
        self.assertEqual(bundle["prediction"]["predicted_close"], 26967.7148)
        self.assertEqual(bundle["provenance"]["source_commit"], "894d0eb")
        self.assertTrue(bundle["provenance"]["checkpoint_sha256"].startswith("ad5ec21"))
        self.assertEqual(len(bundle["window_candles"]), MODEL_WINDOW_SIZE)

    def test_window_candles_rebuild_into_dataframe(self):
        bundle = load_offline_prediction(self._write(_artifact()))
        window = window_from_candles(bundle["window_candles"])
        self.assertEqual(len(window), MODEL_WINDOW_SIZE)
        self.assertIn("close", window.columns)

    def test_missing_file_raises(self):
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(Path("/nonexistent/offline_prediction.json"))

    def test_wrong_artifact_type_raises(self):
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(self._write(_artifact(artifact_type="something_else")))

    def test_missing_response_block_raises(self):
        bad = _artifact()
        del bad["response"]
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(self._write(bad))

    def test_response_missing_required_key_raises(self):
        bad = _artifact()
        del bad["response"]["predicted_close"]
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(self._write(bad))

    def test_too_few_candles_raises(self):
        bad = _artifact()
        bad["payload"]["candles"] = _candles(MODEL_WINDOW_SIZE - 1)
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(self._write(bad))

    def test_malformed_json_raises(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp.write("{ not valid json")
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        with self.assertRaises(OfflinePredictionError):
            load_offline_prediction(path)


if __name__ == "__main__":
    unittest.main()
