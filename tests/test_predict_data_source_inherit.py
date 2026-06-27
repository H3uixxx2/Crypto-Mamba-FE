from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


def _indicator_harness() -> None:
    """Minimal app: render only the Predict-side data-source indicator and surface
    what it resolved to, so AppTest can assert the inherited dataset."""
    import streamlit as st

    from src.cryptomamba_ui.pages.data_page import render_data_source_indicator

    mode, uploaded = render_data_source_indicator()
    st.text(f"MODE={mode}")
    st.text(f"UPLOAD={'yes:' + uploaded.name if uploaded is not None else 'none'}")


def _csv_bytes(name_rows: int = 14) -> bytes:
    header = b"date,open,high,low,close,volume\n"
    rows = b"\n".join(f"2021-01-{d:02d},1,2,1,2,100".encode() for d in range(1, name_rows + 1))
    return header + rows


class PredictInheritsDataSourceTest(unittest.TestCase):
    def _run(self, **session):
        at = AppTest.from_function(_indicator_harness)
        for key, value in session.items():
            at.session_state[key] = value
        return at.run()

    def test_defaults_to_paper_when_nothing_chosen(self) -> None:
        at = self._run()
        texts = [t.value for t in at.text]
        self.assertIn("MODE=Paper dataset", texts)
        self.assertIn("UPLOAD=none", texts)

    def test_inherits_uploaded_csv_from_data_screen(self) -> None:
        # Mirrors real flow: Data screen persisted the mode + uploaded bytes in
        # shared session state; Predict must inherit them without re-asking.
        at = self._run(
            active_data_mode="Upload CSV",
            uploaded_csv_bytes=_csv_bytes(),
            uploaded_csv_name="mine.csv",
        )
        texts = [t.value for t in at.text]
        self.assertIn("MODE=Upload CSV", texts)
        self.assertIn("UPLOAD=yes:mine.csv", texts)

    def test_falls_back_to_widget_key_when_mirror_absent(self) -> None:
        # Backward-compat: if only the widget-keyed value survived, still resolve it.
        at = self._run(
            data_mode="Upload CSV",
            uploaded_csv_bytes=_csv_bytes(),
            uploaded_csv_name="legacy.csv",
        )
        texts = [t.value for t in at.text]
        self.assertIn("MODE=Upload CSV", texts)
        self.assertIn("UPLOAD=yes:legacy.csv", texts)

    def test_indicator_renders_no_picker(self) -> None:
        # The whole point: Predict shows a read-only indicator, not the radio/uploader.
        at = self._run(active_data_mode="Paper dataset")
        self.assertEqual(len(at.radio), 0)


if __name__ == "__main__":
    unittest.main()
