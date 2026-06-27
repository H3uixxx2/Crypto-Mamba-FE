from __future__ import annotations

import unittest

import pandas as pd

from src.cryptomamba_ui.charts import leaderboard_bar_chart, tradeoff_scatter_chart
from src.cryptomamba_ui.pages.reproduce_page import (
    model_metrics_frame,
    trading_balance_series,
)


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "arima", "RMSE": 1564.6, "MAE": 1093.0, "MAPE_pct": 1.977,
             "directional_accuracy_strict_pct": 51.14},
            {"model": "naive_persistence", "RMSE": 1568.5, "MAE": 1091.9, "MAPE_pct": 1.973,
             "directional_accuracy_strict_pct": 0.0},
            {"model": "CryptoMamba-v (retrained)", "RMSE": 1612.4, "MAE": 1132.5, "MAPE_pct": 2.049,
             "directional_accuracy_strict_pct": None},
        ]
    )


def _significance() -> pd.DataFrame:
    return pd.DataFrame([{"cm_directional_acc_pct": 56.86}])


class ModelMetricsFrameTest(unittest.TestCase):
    def test_grafts_cm_directional_accuracy_from_significance(self) -> None:
        frame = model_metrics_frame(_comparison(), _significance())
        cm = frame[frame["model"] == "CryptoMamba-v (retrained)"].iloc[0]
        self.assertAlmostEqual(cm["dir_acc_pct"], 56.86)

    def test_keeps_baseline_own_directional_accuracy(self) -> None:
        frame = model_metrics_frame(_comparison(), _significance())
        self.assertAlmostEqual(
            frame[frame["model"] == "arima"].iloc[0]["dir_acc_pct"], 51.14
        )
        self.assertAlmostEqual(
            frame[frame["model"] == "naive_persistence"].iloc[0]["dir_acc_pct"], 0.0
        )

    def test_empty_comparison_returns_empty_frame(self) -> None:
        frame = model_metrics_frame(pd.DataFrame(), _significance())
        self.assertTrue(frame.empty)

    def test_no_significance_leaves_cm_dir_acc_missing(self) -> None:
        frame = model_metrics_frame(_comparison(), pd.DataFrame())
        cm = frame[frame["model"] == "CryptoMamba-v (retrained)"].iloc[0]
        self.assertTrue(pd.isna(cm["dir_acc_pct"]))


class LeaderboardOrderingTest(unittest.TestCase):
    def test_lower_is_better_puts_smallest_first(self) -> None:
        fig = leaderboard_bar_chart(
            ["b", "a", "c"], [3.0, 1.0, 2.0], value_label="RMSE", lower_is_better=True
        )
        self.assertEqual(list(fig.data[0].x), ["a", "c", "b"])

    def test_higher_is_better_puts_largest_first(self) -> None:
        fig = leaderboard_bar_chart(
            ["b", "a", "c"], [3.0, 1.0, 2.0], value_label="Dir acc", lower_is_better=False
        )
        self.assertEqual(list(fig.data[0].x), ["b", "c", "a"])

    def test_highlights_cryptomamba_distinctly(self) -> None:
        fig = leaderboard_bar_chart(
            ["arima", "CryptoMamba-v"], [1.0, 2.0], value_label="RMSE", lower_is_better=True
        )
        colors = list(fig.data[0].marker.color)
        # two distinct colors; CryptoMamba bar differs from the baseline bar
        self.assertEqual(len(set(colors)), 2)


class TradingBalanceSeriesTest(unittest.TestCase):
    def _replay(self) -> pd.DataFrame:
        rows = []
        for result_type, final in (("official_checkpoint", 124.9), ("retrained_checkpoint", 101.0)):
            for mode in ("vanilla", "smart", "smart_w_short"):
                rows.append(
                    {"result_type": result_type, "split": "val", "trade_mode": mode,
                     "final_balance": final, "paper_final_balance": 124.1}
                )
        return pd.DataFrame(rows)

    def test_builds_three_strategies_and_three_series(self) -> None:
        categories, series = trading_balance_series(self._replay(), "val")
        self.assertEqual(categories, ["Vanilla", "Smart", "Smart + short"])
        self.assertEqual(set(series.keys()), {"Paper", "Official", "Our retrain"})
        self.assertEqual(series["Our retrain"], [101.0, 101.0, 101.0])
        self.assertEqual(series["Paper"], [124.1, 124.1, 124.1])

    def test_missing_split_returns_empty(self) -> None:
        categories, series = trading_balance_series(self._replay(), "test")
        self.assertEqual(categories, [])
        self.assertEqual(series, {})


class TradeoffScatterTest(unittest.TestCase):
    def test_one_trace_per_model(self) -> None:
        fig = tradeoff_scatter_chart(["a", "CryptoMamba-v"], [1500.0, 1600.0], [40.0, 57.0])
        self.assertEqual(len(fig.data), 2)


if __name__ == "__main__":
    unittest.main()
