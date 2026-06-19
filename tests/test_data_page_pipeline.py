from __future__ import annotations

import io
import unittest

from src.cryptomamba_ui.pages.data_page import uploaded_pipeline_report


class DataPagePipelineReportTest(unittest.TestCase):
    def test_missing_date_column_marks_validate_fail_and_later_stages_skip(self) -> None:
        csv = "open,high,low,close,volume\n" + "\n".join(["1,2,1,2,100" for _ in range(14)])
        report, raw = uploaded_pipeline_report(io.BytesIO(csv.encode()))

        self.assertIsNotNone(raw)
        self.assertEqual(report["status"].tolist(), ["PASS", "FAIL", "SKIP", "SKIP", "SKIP"])
        self.assertEqual(report.loc[1, "stage"], "Validate schema")
        self.assertIn("Missing required columns: date", report.loc[1, "message"])

    def test_short_csv_marks_validate_fail_before_tensor(self) -> None:
        csv = "date,open,high,low,close,volume\n2024-01-01,1,2,1,2,100\n2024-01-02,2,3,2,3,100"
        report, _ = uploaded_pipeline_report(io.BytesIO(csv.encode()))

        self.assertEqual(report["status"].tolist(), ["PASS", "FAIL", "SKIP", "SKIP", "SKIP"])
        self.assertIn("Need at least 14", report.loc[1, "message"])


if __name__ == "__main__":
    unittest.main()
