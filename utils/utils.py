from utils.utils import *
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
