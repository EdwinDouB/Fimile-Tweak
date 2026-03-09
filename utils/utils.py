import os
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from typing import Any

from utils.constants import * 

def read_config(name: str, default: str = "") -> str:
    """Read config from env first, then Streamlit secrets."""
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value in (None, ""):
        return default
    return str(secret_value)

def to_local_dt(ts_millis: Any) -> datetime | None:
    if ts_millis is None:
        return None
    try:
        millis = int(ts_millis)
        return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).astimezone()
    except (ValueError, TypeError, OSError):
        return None


def fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def diff_hours(end_dt: datetime | None, start_dt: datetime | None) -> str:
    if not end_dt or not start_dt:
        return ""
    return f"{(end_dt - start_dt).total_seconds() / 3600:.2f}"

def normalize_region(value: Any) -> str:
    text = str(value or "").strip().upper()

    if text in {"WE", "WEST", "W", "美西"}:
        return "WE"
    if text in {"EA", "EAST", "E", "美东"}:
        return "EA"
    return ""

def to_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[column], errors="coerce")


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def calculate_package_evaluation_weight(source_df: pd.DataFrame) -> pd.Series:
    score_cols = [col for col in source_df.columns if col.startswith("pod_score_")]
    if not score_cols:
        return pd.Series([0.0] * len(source_df), index=source_df.index, dtype="float64")

    score_df = source_df[score_cols].apply(pd.to_numeric, errors="coerce")
    return score_df.mean(axis=1, skipna=True).fillna(0.0)

def tr(key: str, **kwargs: Any) -> str:
    lang = st.session_state.get("language", "zh")
    template = I18N.get(lang, I18N["zh"]).get(key, I18N["zh"].get(key, key))
    return template.format(**kwargs)


def read_uploaded_ids(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, dtype=str)
        else:
            return []
    except Exception:
        return []

    if df.empty:
        return []

    preferred = [c for c in df.columns if str(c).lower() in {"tracking_id", "trackingid", "tracking_id"}]
    if preferred:
        series = df[preferred[0]].dropna()
        return series.astype(str).tolist()

    values: list[str] = []
    for col in df.columns:
        values.extend(df[col].dropna().astype(str).tolist())
    return values