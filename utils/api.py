from utils.utils import *
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
API_TIMEOUT_SECONDS = int(read_config("KPI_API_TIMEOUT_SECONDS", "20"))
API_MAX_WORKERS = max(1, int(read_config("KPI_API_MAX_WORKERS", "12")))


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


def build_api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if not API_TOKEN:
        return headers

    token = API_TOKEN.strip()
    if token.lower().startswith("basic ") or token.lower().startswith("bearer "):
        headers["Authorization"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_tracking_data(tracking_id: str, session: requests.Session, headers: dict[str, str]) -> dict[str, Any]:
    url = build_api_url(tracking_id)
    response = session.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()
