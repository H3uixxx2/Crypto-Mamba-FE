from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from src.cryptomamba_ui.data import (
    CandleDataError,
    add_chronological_split,
    add_dataset_split,
    build_predict_payload,
    daily_ohlcv,
    model_tensor_preview,
    normalize_candles,
    split_summary,
)


class DataPipelineTest(unittest.TestCase):
    def test_normalize_accepts_timestamp_seconds_and_sorts(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [self._ts("2024-01-03"), self._ts("2024-01-01"), self._ts("2024-01-02"), *[self._ts(f"2024-01-{day:02d}") for day in range(4, 16)]],
                "Open": [3, 1, 2, *range(4, 16)],
                "High": [4, 2, 3, *range(5, 17)],
                "Low": [2, 0.5, 1, *range(3, 15)],
                "Close": [3.5, 1.5, 2.5, *[day + 0.5 for day in range(4, 16)]],
                "Volume": [100.0] * 15,
            }
        )

        normalized = normalize_candles(df)

        self.assertEqual(len(normalized), 15)
        self.assertEqual(normalized["date"].iloc[0], "2024-01-01")
        self.assertEqual(normalized["date"].iloc[-1], "2024-01-15")
        self.assertEqual(normalized.columns.tolist(), ["date", "open", "high", "low", "close", "volume"])

    def test_normalize_accepts_timestamp_milliseconds(self) -> None:
        timestamps_ms = [self._ts(f"2024-01-{day:02d}") * 1000 for day in range(1, 15)]
        df = self._frame_with_dates(timestamps_ms, date_column="open_time")

        normalized = normalize_candles(df)

        self.assertEqual(normalized["date"].iloc[0], "2024-01-01")
        self.assertEqual(normalized["date"].iloc[-1], "2024-01-14")

    def test_normalize_accepts_coinmarketcap_time_open_schema(self) -> None:
        df = pd.DataFrame(
            {
                "timeOpen": [f"2026-05-{day:02d}T00:00:00.000Z" for day in range(1, 15)],
                "timeClose": [f"2026-05-{day:02d}T23:59:59.999Z" for day in range(1, 15)],
                "name": ["Ethereum"] * 14,
                "open": list(range(1, 15)),
                "high": list(range(2, 16)),
                "low": [0.5 + i for i in range(14)],
                "close": [1.5 + i for i in range(14)],
                "volume": [1000.0] * 14,
                "marketCap": [1_000_000.0] * 14,
            }
        )

        normalized = normalize_candles(df)

        self.assertEqual(normalized.columns.tolist(), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(normalized["date"].iloc[0], "2026-05-01")
        self.assertEqual(normalized["date"].iloc[-1], "2026-05-14")

    def test_normalize_accepts_canonicalized_provider_headers(self) -> None:
        df = pd.DataFrame(
            {
                "Open Time": [f"2026-06-{day:02d}T00:00:00.000Z" for day in range(1, 15)],
                "Open Price": list(range(1, 15)),
                "High Price": list(range(2, 16)),
                "Low Price": [0.5 + i for i in range(14)],
                "Close Price": [1.5 + i for i in range(14)],
                "Volume USD": [1000.0] * 14,
            }
        )

        normalized = normalize_candles(df)

        self.assertEqual(normalized.columns.tolist(), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(normalized["date"].iloc[0], "2026-06-01")
        self.assertEqual(float(normalized["volume"].iloc[-1]), 1000.0)

    def test_missing_required_columns_fails_fast(self) -> None:
        df = pd.DataFrame({"date": ["2024-01-01"] * 14, "open": [1] * 14})

        with self.assertRaisesRegex(CandleDataError, "Missing required columns"):
            normalize_candles(df)

    def test_invalid_numeric_values_fail_fast(self) -> None:
        df = self._frame_with_dates([f"2024-01-{day:02d}" for day in range(1, 15)])
        df["close"] = df["close"].astype(object)
        df.loc[3, "close"] = "bad"

        with self.assertRaisesRegex(CandleDataError, "Invalid numeric OHLCV"):
            normalize_candles(df)

    def test_fewer_than_14_rows_fails(self) -> None:
        df = self._frame_with_dates([f"2024-01-{day:02d}" for day in range(1, 14)])

        with self.assertRaisesRegex(CandleDataError, "Need at least 14"):
            normalize_candles(df)

    def test_daily_ohlcv_aggregates_intraday_rows(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", *[f"2024-01-{day:02d}" for day in range(2, 15)]],
                "open": [100, 110, *range(102, 115)],
                "high": [120, 130, *range(122, 135)],
                "low": [90, 95, *range(82, 95)],
                "close": [115, 125, *range(112, 125)],
                "volume": [10, 20, *[1] * 13],
            }
        )

        daily = daily_ohlcv(df)

        first = daily.iloc[0]
        self.assertEqual(first["open"], 100)
        self.assertEqual(first["high"], 130)
        self.assertEqual(first["low"], 90)
        self.assertEqual(first["close"], 125)
        self.assertEqual(first["volume"], 30)

    def test_split_boundaries_match_paper(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2018-09-17", "2022-09-16", "2022-09-17", "2023-09-16", "2023-09-17", "2024-09-16", "2024-09-17"],
                "open": [1] * 7,
                "high": [2] * 7,
                "low": [0.5] * 7,
                "close": [1.5] * 7,
                "volume": [100] * 7,
            }
        )

        split = add_dataset_split(df)

        self.assertEqual(split["dataset_split"].tolist(), ["train", "train", "validation", "validation", "test", "test", "out_of_scope"])
        summary = split_summary(split)
        self.assertEqual(summary.loc[summary["split"] == "train", "from"].iloc[0], "2018-09-17")
        self.assertEqual(summary.loc[summary["split"] == "test", "to"].iloc[0], "2024-09-16")

    def test_chronological_split_uses_uploaded_dataset_range(self) -> None:
        df = self._frame_with_dates([f"2026-01-{day:02d}" for day in range(1, 21)])

        split = add_chronological_split(df, train_ratio=0.70, validation_ratio=0.15)

        self.assertEqual(split["dataset_split"].value_counts().to_dict(), {"train": 14, "validation": 3, "test": 3})
        self.assertEqual(split.loc[0, "date"], "2026-01-01")
        self.assertEqual(split.loc[13, "dataset_split"], "train")
        self.assertEqual(split.loc[14, "dataset_split"], "validation")
        self.assertEqual(split.loc[17, "dataset_split"], "test")

    def test_build_predict_payload_returns_sorted_14_candles(self) -> None:
        df = self._frame_with_dates([f"2024-01-{day:02d}" for day in range(1, 17)]).sample(frac=1, random_state=7)

        payload = build_predict_payload(df, prediction_date=None, risk=2)

        self.assertEqual(payload["prediction_date"], "2024-01-17")
        self.assertEqual(payload["risk"], 2.0)
        self.assertEqual(len(payload["candles"]), 14)
        self.assertEqual(payload["candles"][0]["date"], "2024-01-03")
        self.assertEqual(payload["candles"][-1]["date"], "2024-01-16")
        self.assertTrue(all(isinstance(candle["close"], float) for candle in payload["candles"]))

    def test_build_predict_payload_past_date_uses_historical_window(self) -> None:
        # Regression: a past prediction_date must feed the model the 14 candles ending
        # strictly BEFORE it (same window the chart shows), not the latest 14 candles.
        df = self._frame_with_dates([f"2024-01-{day:02d}" for day in range(1, 21)])

        payload = build_predict_payload(df, prediction_date="2024-01-15", risk=2)

        self.assertEqual(payload["prediction_date"], "2024-01-15")
        self.assertEqual(len(payload["candles"]), 14)
        self.assertEqual(payload["candles"][0]["date"], "2024-01-01")
        self.assertEqual(payload["candles"][-1]["date"], "2024-01-14")

    def test_model_tensor_preview_contract(self) -> None:
        df = self._frame_with_dates([f"2024-01-{day:02d}" for day in range(1, 15)])

        preview, payload = model_tensor_preview(df)

        self.assertEqual(len(preview), 14)
        self.assertEqual(payload["feature_order"], ["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(payload["tensor_shape_before_batch"], [6, 14])
        self.assertEqual(payload["tensor_shape_sent_to_model"], [1, 6, 14])
        self.assertEqual(len(payload["features"]), 6)
        self.assertEqual(len(payload["features"][0]), 14)
        self.assertAlmostEqual(payload["features"][5][0], 0.000001)

    @staticmethod
    def _ts(date: str) -> int:
        return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())

    @staticmethod
    def _frame_with_dates(values: list[object], date_column: str = "date") -> pd.DataFrame:
        return pd.DataFrame(
            {
                date_column: values,
                "open": list(range(1, len(values) + 1)),
                "high": list(range(2, len(values) + 2)),
                "low": [0.5 + i for i in range(len(values))],
                "close": [1.5 + i for i in range(len(values))],
                "volume": [1000.0 for _ in values],
            }
        )


if __name__ == "__main__":
    unittest.main()
