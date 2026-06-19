from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.cryptomamba_ui.dataset_service import DatasetService


class DatasetServiceTest(unittest.TestCase):
    def test_load_uploaded_csv_processes_raw_to_daily_split_bundle(self) -> None:
        service = DatasetService(sample_path=Path("unused.csv"), display_root=Path("."))
        csv = io.StringIO(
            "\n".join(
                [
                    "date,open,high,low,close,volume",
                    "2023-09-17,100,120,90,110,10",
                    "2023-09-17,111,130,95,125,20",
                    *[f"2023-09-{day:02d},{day},{day + 1},{day - 1},{day + 0.5},1" for day in range(18, 31)],
                ]
            )
        )
        csv.name = "custom.csv"

        bundle = service.load_uploaded_csv(csv)

        self.assertEqual(bundle.source_label, "Uploaded CSV")
        self.assertEqual(bundle.source_detail, "File: custom.csv")
        self.assertEqual(len(bundle.raw_df), 15)
        self.assertEqual(len(bundle.daily_df), 14)
        first_daily = bundle.daily_df.iloc[0]
        self.assertEqual(first_daily["open"], 100)
        self.assertEqual(first_daily["high"], 130)
        self.assertEqual(first_daily["low"], 90)
        self.assertEqual(first_daily["close"], 125)
        self.assertEqual(first_daily["volume"], 30)
        self.assertIn("dataset_split", bundle.processed_df.columns)
        self.assertEqual(bundle.split_strategy, "chronological_ratio")
        self.assertEqual(bundle.processed_df["dataset_split"].iloc[0], "train")
        self.assertIn("validation", set(bundle.processed_df["dataset_split"]))
        self.assertIn("test", set(bundle.processed_df["dataset_split"]))

    def test_load_uploaded_csv_infers_semicolon_delimiter(self) -> None:
        service = DatasetService(sample_path=Path("unused.csv"), display_root=Path("."))
        csv = io.StringIO(
            "\n".join(
                [
                    "timeOpen;open;high;low;close;volume",
                    *[f"2026-05-{day:02d}T00:00:00.000Z;{day};{day + 1};0.5;{day + 0.5};1000" for day in range(1, 15)],
                ]
            )
        )
        csv.name = "semicolon.csv"

        bundle = service.load_uploaded_csv(csv)

        self.assertEqual(len(bundle.daily_df), 14)
        self.assertEqual(bundle.daily_df["date"].iloc[0], "2026-05-01")
        self.assertEqual(bundle.source_detail, "File: semicolon.csv")

    def test_load_paper_sample_uses_configured_sample_path_and_display_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_path = root / "sample.csv"
            pd.DataFrame(
                {
                    "date": [f"2024-01-{day:02d}" for day in range(1, 15)],
                    "open": list(range(1, 15)),
                    "high": list(range(2, 16)),
                    "low": [0.5 + i for i in range(14)],
                    "close": [1.5 + i for i in range(14)],
                    "volume": [1000.0] * 14,
                }
            ).to_csv(sample_path, index=False)

            bundle = DatasetService(sample_path=sample_path, display_root=root).load_paper_sample()

            self.assertEqual(bundle.source_label, "Paper dataset")
            self.assertEqual(bundle.source_detail, "File: sample.csv")
            self.assertEqual(bundle.split_strategy, "paper_date")
            self.assertEqual(len(bundle.processed_df), 14)

    def test_empty_upload_returns_empty_bundle_without_processing(self) -> None:
        bundle = DatasetService.empty_upload()

        self.assertEqual(bundle.source_label, "Uploaded CSV")
        self.assertEqual(bundle.source_detail, "No file uploaded yet")
        self.assertTrue(bundle.raw_df.empty)
        self.assertTrue(bundle.daily_df.empty)
        self.assertTrue(bundle.processed_df.empty)


if __name__ == "__main__":
    unittest.main()
