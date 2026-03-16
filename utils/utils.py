

import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

from utils.constants import I18N


def read_config(key: str, default: str = "") -> str:
    """Read runtime config from Streamlit secrets, local secrets file, then env vars."""
    value = None
    if hasattr(st, "secrets"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

    if value is None or str(value).strip() == "":
        value = _read_local_streamlit_secret(key)

    if value is None or str(value).strip() == "":
        value = os.getenv(key, default)
    return str(value)


def _read_local_streamlit_secret(key: str) -> str | None:
    """Best-effort fallback for local runs outside `streamlit run`.

    In non-Streamlit contexts (CLI scripts/tests), `st.secrets` may be unavailable.
    We mirror Streamlit's local secrets file lookup and read `secrets.toml` directly.
    """
    candidates = (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.cwd() / "secrets.toml",
    )
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            import tomllib
            with path.open("rb") as f:
                payload = tomllib.load(f)
            if isinstance(payload, dict) and key in payload:
                raw = payload.get(key)
                if raw is not None:
                    return str(raw)
        except Exception:
            continue
    return None


def tr(key: str, **kwargs) -> str:
    lang = st.session_state.get("language", "zh")
    text = I18N.get(lang, I18N["zh"]).get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def rate(hit: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return float(hit) / float(total)


def to_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")
    return pd.to_datetime(df[column], errors="coerce")

def calculate_package_evaluation_weight(df: pd.DataFrame) -> pd.Series:
    """Return per-package evaluation weight for KPI visualizations.

    Some datasets contain an explicit weight-like column while others don't.
    When no usable weight column exists, each package contributes `1.0` so
    aggregated evaluation weight remains meaningful and charts stay available.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    weight_columns = [
        "evaluation_weight",
        "package_evaluation_weight",
        "package_weight",
        "weight",
    ]
    for column in weight_columns:
        if column not in df.columns:
            continue

        numeric_weight = pd.to_numeric(df[column], errors="coerce")
        if numeric_weight.notna().any():
            return numeric_weight.fillna(1.0).clip(lower=0)

    return pd.Series(1.0, index=df.index, dtype="float64")

def to_local_dt(ts_millis: int | float | None, local_tz=timezone.utc) -> datetime | None:
    """Convert unix timestamp in milliseconds to timezone-aware datetime."""
    if ts_millis is None:
        return None

    try:
        seconds = float(ts_millis) / 1000.0
    except (TypeError, ValueError):
        return None

    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(local_tz)
    except (OverflowError, OSError, ValueError):
        return None


def fmt_dt(value: datetime | None) -> str:
    """Format datetime for exports."""
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def diff_hours(end_time: datetime | None, start_time: datetime | None) -> str:
    """Return (end-start) hour difference as a numeric string with 2 decimals."""
    if end_time is None or start_time is None:
        return ""
    delta_hours = (end_time - start_time).total_seconds() / 3600.0
    return f"{delta_hours:.2f}"

import base64
import json
import requests
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# API configuration comes from code/env only (not exposed in UI).
API_URL_TEMPLATE = os.getenv(
    "KPI_API_URL_TEMPLATE",
    "https://isp.beans.ai/enterprise/v1/lists/status_logs"
    "?tracking_id={tracking_id}&readable=true"
    "&include_pod=true&include_item=true",
)
API_URL_TEMPLATE = read_config("KPI_API_URL_TEMPLATE", API_URL_TEMPLATE)
API_TOKEN = read_config("KPI_API_TOKEN", "")
API_AUTH_MODE = read_config("KPI_API_AUTH_MODE", "auto").strip().lower()
API_TIMEOUT_SECONDS = int(read_config("KPI_API_TIMEOUT_SECONDS", "20"))
API_MAX_WORKERS = max(1, int(read_config("KPI_API_MAX_WORKERS", "12")))
API_EXTRA_HEADERS = read_config("KPI_API_EXTRA_HEADERS", "")


def build_beans_tracking_link(tracking_id: str) -> str:
    return f"https://www.beansroute.ai/3pl-manager/tabs.html#searchTrackingId/{tracking_id}"


def build_api_url(tracking_id: str) -> str:
    if not API_URL_TEMPLATE:
        raise RuntimeError("KPI_API_URL_TEMPLATE 未配置")

    if "{tracking_id}" in API_URL_TEMPLATE:
        return API_URL_TEMPLATE.format(tracking_id=tracking_id)

    parsed = urlparse(API_URL_TEMPLATE)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["tracking_id"] = tracking_id
    return urlunparse(parsed._replace(query=urlencode(query)))


def _parse_extra_headers(raw: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}

    if text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items() if str(k).strip()}
        except json.JSONDecodeError:
            return {}

    parsed: dict[str, str] = {}
    for line in text.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue

        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue

        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value

    return parsed


def build_api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        # Some upstream WAF rules reject python-requests' default UA and return 403.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    headers.update(_parse_extra_headers(API_EXTRA_HEADERS))

    if not API_TOKEN or API_AUTH_MODE == "none":
        return headers

    token = API_TOKEN.strip()

    if API_AUTH_MODE == "raw":
        headers["Authorization"] = token
        return headers

    if API_AUTH_MODE == "basic":
        encoded = base64.b64encode(token.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
        return headers

    if API_AUTH_MODE == "bearer":
        headers["Authorization"] = f"Bearer {token}"
        return headers

    # auto mode
    if token.lower().startswith("basic ") or token.lower().startswith("bearer "):
        headers["Authorization"] = token
    elif ":" in token:
        encoded = base64.b64encode(token.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
    else:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def fetch_tracking_data(tracking_id: str, session: requests.Session, headers: dict[str, str]) -> dict[str, Any]:
    url = build_api_url(tracking_id)
    response = session.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()
