from __future__ import annotations

import io
from typing import Any

import streamlit as st


def remember_uploaded_csv(uploaded_file: Any | None) -> io.BytesIO | None:
    """Persist uploaded file bytes across Streamlit widget unmount/remount."""
    if uploaded_file is not None:
        st.session_state["uploaded_csv_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_csv_name"] = getattr(uploaded_file, "name", "uploaded.csv")

    data = st.session_state.get("uploaded_csv_bytes")
    if not data:
        return None

    buffer = io.BytesIO(data)
    buffer.name = st.session_state.get("uploaded_csv_name", "uploaded.csv")
    return buffer


def retained_uploaded_csv_name() -> str | None:
    if not st.session_state.get("uploaded_csv_bytes"):
        return None
    return str(st.session_state.get("uploaded_csv_name", "uploaded.csv"))


def clear_retained_upload() -> None:
    st.session_state.pop("uploaded_csv_bytes", None)
    st.session_state.pop("uploaded_csv_name", None)
    st.session_state["replace_uploaded_csv"] = False
    st.session_state["uploaded_csv_uploader_version"] = int(st.session_state.get("uploaded_csv_uploader_version", 0)) + 1
