from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.cryptomamba_ui.reproduce_artifacts import load_reproduce_artifacts


class ReproduceArtifactsTest(unittest.TestCase):
    def test_valid_bundle_is_ready_with_partial_baseline_and_retrained_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "READY")
            self.assertEqual(bundle.forecast_status, "PASS")
            self.assertEqual(bundle.replay_status, "COMPLETE_WITH_RETRAINED_MISMATCH")
            self.assertEqual(bundle.baseline_status, "PARTIAL")
            self.assertEqual(set(bundle.forecast_metrics["result_type"]), {"official_checkpoint", "retrained_checkpoint"})
            self.assertFalse(bundle.errors)

    def test_missing_artifact_returns_not_ready_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            (evaluation_dir / "forecast_metrics.csv").unlink()

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(bundle.forecast_metrics.empty)
            self.assertTrue(any("forecast_metrics.csv" in error for error in bundle.errors))

    def test_malformed_csv_returns_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            (evaluation_dir / "forecast_metrics.csv").write_text('"unterminated', encoding="utf-8")

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("Cannot parse forecast_metrics.csv" in error for error in bundle.errors))

    def test_missing_required_column_returns_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            forecast_path = evaluation_dir / "forecast_metrics.csv"
            forecast = pd.read_csv(forecast_path).drop(columns=["MAPE_pct"])
            forecast.to_csv(forecast_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("MAPE_pct" in error for error in bundle.errors))

    def test_duplicate_forecast_logical_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            forecast_path = evaluation_dir / "forecast_metrics.csv"
            forecast = pd.read_csv(forecast_path)
            pd.concat([forecast, forecast.iloc[[0]]], ignore_index=True).to_csv(forecast_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("duplicate logical rows" in error for error in bundle.errors))

    def test_mape_fraction_percent_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            forecast_path = evaluation_dir / "forecast_metrics.csv"
            forecast = pd.read_csv(forecast_path)
            forecast.loc[forecast["result_type"] == "retrained_checkpoint", "MAPE_pct"] = 0.0205
            forecast.to_csv(forecast_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("MAPE_gap_pct" in error for error in bundle.errors))

    def test_forecast_gap_above_tolerance_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            forecast_path = evaluation_dir / "forecast_metrics.csv"
            forecast = pd.read_csv(forecast_path)
            retrained = forecast["result_type"] == "retrained_checkpoint"
            forecast.loc[retrained, "RMSE"] = forecast.loc[retrained, "paper_RMSE"] * 1.06
            forecast.loc[retrained, ["RMSE_gap_pct", "status"]] = [6.0, "FAIL"]
            forecast.to_csv(forecast_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertEqual(bundle.forecast_status, "FAIL")

    def test_incomplete_six_mode_replay_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            replay_path = evaluation_dir / "trading_replay_metrics.csv"
            pd.read_csv(replay_path).iloc[:-1].to_csv(replay_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertEqual(bundle.replay_status, "INCOMPLETE")

    def test_non_finite_replay_metric_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            replay_path = evaluation_dir / "trading_replay_metrics.csv"
            replay = pd.read_csv(replay_path)
            replay.loc[0, "final_balance"] = float("inf")
            replay.to_csv(replay_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("non-finite numeric values" in error for error in bundle.errors))

    def test_artifact_validation_row_counts_must_match_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            validation_path = provenance_dir / "artifact_validation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["prediction_rows"] = 999
            validation_path.write_text(json.dumps(validation), encoding="utf-8")

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("prediction_rows" in error for error in bundle.errors))

    def test_failed_model_selection_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            selection_path = evaluation_dir / "model_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["forecast_status"] = "FAIL"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("model_selection forecast_status" in error for error in bundle.errors))

    def test_fixture_checkpoint_hash_mismatch_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            fixture_path = evaluation_dir / "inference_fixture.json"
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            fixture["checkpoint_sha256"] = "0" * 64
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("inference fixture checkpoint hash" in error for error in bundle.errors))

    def test_predictions_require_dates_targets_predictions_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            predictions_path = evaluation_dir / "forecast_predictions.csv"
            predictions = pd.read_csv(predictions_path).drop(columns=["prediction_date", "source_commit"])
            predictions.to_csv(predictions_path, index=False)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.status, "NOT_READY")
            self.assertTrue(any("prediction_date" in error and "source_commit" in error for error in bundle.errors))

    def test_baseline_partial_lists_present_models_when_only_naive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.baseline_status, "PARTIAL")
            self.assertEqual(bundle.baseline_models_present, ("naive_persistence",))
            self.assertTrue(bundle.baseline_comparison.empty)
            self.assertTrue(bundle.significance.empty)

    def test_baseline_partial_with_two_of_five_from_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            self._write_comparison(evaluation_dir, ["naive_persistence", "arima"])
            self._write_significance(evaluation_dir)

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            # ARIMA present but not all five -> still PARTIAL (the previous logic wrongly returned COMPLETE)
            self.assertEqual(bundle.baseline_status, "PARTIAL")
            self.assertEqual(bundle.baseline_models_present, ("arima", "naive_persistence"))
            self.assertFalse(bundle.baseline_comparison.empty)
            self.assertFalse(bundle.significance.empty)
            self.assertEqual(bundle.status, "READY")  # PARTIAL baseline still allows READY

    def test_baseline_complete_when_all_five_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir, provenance_dir, checkpoint_path = self._write_valid_bundle(Path(tmp))
            self._write_comparison(
                evaluation_dir, ["naive_persistence", "arima", "lstm", "gru", "itransformer"]
            )

            bundle = load_reproduce_artifacts(evaluation_dir, provenance_dir, checkpoint_path)

            self.assertEqual(bundle.baseline_status, "COMPLETE")
            self.assertEqual(
                set(bundle.baseline_models_present),
                {"naive_persistence", "arima", "lstm", "gru", "itransformer"},
            )

    @staticmethod
    def _write_comparison(evaluation_dir: Path, models: list[str]) -> None:
        rows = [
            {"model": m, "split": "test", "samples": 350, "RMSE": 1600.0 + i, "MAE": 1100.0,
             "MAPE_pct": 2.0, "paper_RMSE": 1600.0, "RMSE_gap_pct": 0.5}
            for i, m in enumerate(models)
        ]
        pd.DataFrame(rows).to_csv(evaluation_dir / "baseline_metrics_comparison.csv", index=False)

    @staticmethod
    def _write_significance(evaluation_dir: Path) -> None:
        pd.DataFrame(
            [
                {"comparison": "CryptoMamba-v vs arima", "samples": 350,
                 "cm_directional_acc_pct": 56.86, "baseline_directional_acc_pct": 51.14,
                 "dm_p_value": 0.1479, "wilcoxon_p_value": 0.0758, "conclusion": "not_significant"}
            ]
        ).to_csv(evaluation_dir / "significance_tests.csv", index=False)

    @staticmethod
    def _write_valid_bundle(root: Path) -> tuple[Path, Path, Path]:
        evaluation_dir = root / "output/evaluation"
        provenance_dir = root / "provenance"
        parity_dir = provenance_dir / "source_cli_parity"
        evaluation_dir.mkdir(parents=True)
        parity_dir.mkdir(parents=True)

        checkpoint_path = root / "checkpoints/cmamba_v_best_colab_train.ckpt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"validated-checkpoint")
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        forecast_rows = []
        for result_type, rmse, mae, mape in (
            ("official_checkpoint", 1598.0924, 1120.6595, 2.0343),
            ("retrained_checkpoint", 1612.3527, 1132.5406, 2.0486),
        ):
            paper_rmse, paper_mae, paper_mape = 1598.1, 1120.7, 2.034
            forecast_rows.append(
                {
                    "result_type": result_type,
                    "model": "CryptoMamba-v",
                    "split": "test",
                    "checkpoint": f"checkpoints/{result_type}.ckpt",
                    "source_commit": "source-commit",
                    "protocol": "global test aggregation",
                    "samples": 350,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE_pct": mape,
                    "paper_RMSE": paper_rmse,
                    "paper_MAE": paper_mae,
                    "paper_MAPE_pct": paper_mape,
                    "RMSE_gap_pct": abs(rmse - paper_rmse) / paper_rmse * 100,
                    "MAE_gap_pct": abs(mae - paper_mae) / paper_mae * 100,
                    "MAPE_gap_pct": abs(mape - paper_mape) / paper_mape * 100,
                    "tolerance_pct": 5.0,
                    "status": "PASS",
                }
            )
        pd.DataFrame(forecast_rows).to_csv(evaluation_dir / "forecast_metrics.csv", index=False)

        replay_rows = []
        for result_type in ("official_checkpoint", "retrained_checkpoint"):
            for split in ("val", "test"):
                for trade_mode in ("vanilla", "smart", "smart_w_short"):
                    replay_rows.append(
                        {
                            "result_type": result_type,
                            "model": "CryptoMamba-v",
                            "checkpoint": f"checkpoints/{result_type}.ckpt",
                            "source_commit": "source-commit",
                            "split": split,
                            "trade_mode": trade_mode,
                            "initial_balance": 100.0,
                            "risk_pct": 2.0,
                            "final_balance": 120.0,
                            "max_drawdown_pct": 10.0,
                            "paper_final_balance": 121.0,
                            "paper_max_drawdown_pct": 11.0,
                            "balance_gap_pct": 0.5,
                            "mdd_gap_pp": 1.0,
                            "protocol": "released replay",
                            "status": "VERIFIED" if result_type == "official_checkpoint" else "NOT_MATCHED",
                        }
                    )
        pd.DataFrame(replay_rows).to_csv(evaluation_dir / "trading_replay_metrics.csv", index=False)

        pd.DataFrame(
            [
                {
                    "model": "naive_persistence",
                    "split": "test",
                    "samples": 350,
                    "RMSE": 1568.5,
                    "MAE": 1091.9,
                    "MAPE_pct": 1.973,
                    "protocol": "persistence",
                }
            ]
        ).to_csv(evaluation_dir / "baseline_metrics.csv", index=False)

        pd.DataFrame(
            [
                {
                    "result_type": result_type,
                    "model": "CryptoMamba-v",
                    "checkpoint": f"checkpoints/{result_type}.ckpt",
                    "source_commit": "source-commit",
                    "split": "test",
                    "sample_index": 0,
                    "window_start_date": "2023-09-17",
                    "window_end_date": "2023-09-30",
                    "prediction_date": "2023-10-01",
                    "current_close": 100.0,
                    "target_close": 101.0,
                    "predicted_close": 100.5,
                    "actual_return_pct": 1.0,
                    "predicted_return_pct": 0.5,
                    "protocol": "dated prediction",
                }
                for result_type in ("official_checkpoint", "retrained_checkpoint")
            ]
        ).to_csv(evaluation_dir / "forecast_predictions.csv", index=False)

        selection = {
            "selected_checkpoint_type": "retrained_checkpoint",
            "selected_checkpoint_path": "checkpoints/cmamba_v_retrained_best.ckpt",
            "checkpoint_sha256": checkpoint_sha,
            "source_commit": "source-commit",
            "selection_reason": "all gaps below tolerance",
            "forecast_status": "PASS",
            "fallback_checkpoint": "checkpoints/cmamba_v_official.ckpt",
        }
        (evaluation_dir / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")

        fixture = {
            "candles": [{"date": f"2023-09-{day:02d}"} for day in range(17, 31)],
            "expected_feature_order": ["Timestamp", "Open", "High", "Low", "Close", "Volume"],
            "expected_tensor_shape": [1, 6, 14],
            "expected_predicted_close": 100.5,
            "absolute_tolerance": 0.05,
            "checkpoint_sha256": checkpoint_sha,
        }
        (evaluation_dir / "inference_fixture.json").write_text(json.dumps(fixture), encoding="utf-8")

        validation = {
            "status": "PASS",
            "selected_checkpoint_type": "retrained_checkpoint",
            "selected_checkpoint_sha256": checkpoint_sha,
            "forecast_rows": 2,
            "prediction_rows": 2,
            "replay_rows": 12,
            "baseline_rows": 1,
        }
        (provenance_dir / "artifact_validation.json").write_text(json.dumps(validation), encoding="utf-8")
        (provenance_dir / "source_contract.json").write_text(
            json.dumps({"source_commit": "source-commit", "config": "cmamba_v", "seed": 23}),
            encoding="utf-8",
        )
        (parity_dir / "evaluation_parity.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        (parity_dir / "trading_parity.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        return evaluation_dir, provenance_dir, checkpoint_path


if __name__ == "__main__":
    unittest.main()
