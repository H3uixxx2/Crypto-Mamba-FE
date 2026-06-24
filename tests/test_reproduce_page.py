from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.cryptomamba_ui.pages.reproduce_page import (
    evidence_table,
    forecast_comparison_table,
    trading_balance_comparison_table,
    trading_drawdown_comparison_table,
)
from src.cryptomamba_ui.reproduce_artifacts import ReproduceArtifacts


class ReproducePageTest(unittest.TestCase):
    def test_forecast_comparison_has_one_clear_row_per_source(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "result_type": "official_checkpoint",
                    "RMSE": 1598.092,
                    "MAE": 1120.660,
                    "MAPE_pct": 2.034,
                    "paper_RMSE": 1598.100,
                    "paper_MAE": 1120.700,
                    "paper_MAPE_pct": 2.034,
                    "RMSE_gap_pct": 0.000,
                    "MAE_gap_pct": 0.004,
                    "MAPE_gap_pct": 0.016,
                    "status": "PASS",
                },
                {
                    "result_type": "retrained_checkpoint",
                    "RMSE": 1612.353,
                    "MAE": 1132.541,
                    "MAPE_pct": 2.049,
                    "paper_RMSE": 1598.100,
                    "paper_MAE": 1120.700,
                    "paper_MAPE_pct": 2.034,
                    "RMSE_gap_pct": 0.892,
                    "MAE_gap_pct": 1.057,
                    "MAPE_gap_pct": 0.718,
                    "status": "PASS",
                },
            ]
        )

        table = forecast_comparison_table(metrics)

        self.assertEqual(
            table["Source"].tolist(),
            ["Paper target", "Official checkpoint", "Our retrained checkpoint"],
        )
        self.assertEqual(table.loc[0, "Largest gap"], "—")
        self.assertEqual(table.loc[2, "Largest gap"], "1.057%")
        self.assertEqual(table.loc[2, "Verdict"], "PASS (<5%)")

    def test_trading_comparison_has_one_clear_row_per_strategy(self) -> None:
        replay = self._trading_replay()

        table = trading_balance_comparison_table(replay, "test")

        self.assertEqual(table["Strategy"].tolist(), ["Vanilla", "Smart", "Smart + short"])
        self.assertEqual(table.loc[0, "Paper"], "246.58")
        self.assertEqual(table.loc[0, "Official"], "246.58")
        self.assertEqual(table.loc[0, "Our retrain"], "174.94")
        self.assertEqual(table.loc[0, "Our gap"], "-29.05%")
        self.assertEqual(table.loc[0, "Verdict"], "NOT MATCHED")

    def test_trading_drawdown_details_use_same_strategy_order(self) -> None:
        table = trading_drawdown_comparison_table(self._trading_replay(), "test")

        self.assertEqual(table["Strategy"].tolist(), ["Vanilla", "Smart", "Smart + short"])
        self.assertEqual(table.loc[0, "Paper MDD"], "23.48%")
        self.assertEqual(table.loc[0, "Official MDD"], "23.48%")
        self.assertEqual(table.loc[0, "Our MDD"], "23.94%")
        self.assertEqual(table.loc[0, "Difference"], "+0.46 pp")

    def test_evidence_values_use_one_arrow_compatible_string_type(self) -> None:
        artifacts = ReproduceArtifacts(
            artifact_validation={
                "status": "PASS",
                "forecast_rows": 2,
                "prediction_rows": 4292,
                "replay_rows": 12,
            },
            model_selection={
                "checkpoint_sha256": "abc123",
                "selection_reason": "validated",
            },
            inference_fixture={
                "expected_tensor_shape": [1, 6, 14],
                "expected_predicted_close": 26967.716796875,
            },
        )

        table = evidence_table(artifacts, Path("/tmp/model.ckpt"))

        self.assertTrue(all(isinstance(value, str) for value in table["Value"]))

    @staticmethod
    def _trading_replay() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "result_type": "official_checkpoint",
                    "split": "test",
                    "trade_mode": "vanilla",
                    "final_balance": 246.5846,
                    "paper_final_balance": 246.580,
                    "balance_gap_pct": 0.002,
                    "max_drawdown_pct": 23.481,
                    "paper_max_drawdown_pct": 23.480,
                    "status": "VERIFIED",
                },
                {
                    "result_type": "official_checkpoint",
                    "split": "test",
                    "trade_mode": "smart",
                    "final_balance": 213.197,
                    "paper_final_balance": 213.200,
                    "balance_gap_pct": -0.001,
                    "max_drawdown_pct": 13.371,
                    "paper_max_drawdown_pct": 13.370,
                    "status": "VERIFIED",
                },
                {
                    "result_type": "official_checkpoint",
                    "split": "test",
                    "trade_mode": "smart_w_short",
                    "final_balance": 262.778,
                    "paper_final_balance": 262.780,
                    "balance_gap_pct": -0.001,
                    "max_drawdown_pct": 11.091,
                    "paper_max_drawdown_pct": 11.090,
                    "status": "VERIFIED",
                },
                {
                    "result_type": "retrained_checkpoint",
                    "split": "test",
                    "trade_mode": "vanilla",
                    "final_balance": 174.937,
                    "paper_final_balance": 246.580,
                    "balance_gap_pct": -29.055,
                    "max_drawdown_pct": 23.940,
                    "paper_max_drawdown_pct": 23.480,
                    "status": "NOT_MATCHED",
                },
                {
                    "result_type": "retrained_checkpoint",
                    "split": "test",
                    "trade_mode": "smart",
                    "final_balance": 165.251,
                    "paper_final_balance": 213.200,
                    "balance_gap_pct": -22.490,
                    "max_drawdown_pct": 10.200,
                    "paper_max_drawdown_pct": 13.370,
                    "status": "NOT_MATCHED",
                },
                {
                    "result_type": "retrained_checkpoint",
                    "split": "test",
                    "trade_mode": "smart_w_short",
                    "final_balance": 204.260,
                    "paper_final_balance": 262.780,
                    "balance_gap_pct": -22.269,
                    "max_drawdown_pct": 11.740,
                    "paper_max_drawdown_pct": 11.090,
                    "status": "NOT_MATCHED",
                },
            ]
        )


if __name__ == "__main__":
    unittest.main()
