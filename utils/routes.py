from collections import Counter
from datetime import datetime, timezone
from typing import Any
import os
import io 
import re 
import json
import requests

from utils.utils import *
from utils.constants import * 

# How many POD images to export per tracking_id (each image can have its own quality.feedback/score)
POD_IMAGE_EXPORT_N = int(os.getenv("POD_IMAGE_EXPORT_N", "5"))
POD_IMAGE_EXPORT_N = int(read_config("POD_IMAGE_EXPORT_N", str(POD_IMAGE_EXPORT_N)))
APP_VERSION = read_config("APP_VERSION", "a0.0.5")

POD_COLUMNS: list[str] = []
for i in range(1, POD_IMAGE_EXPORT_N + 1):
    POD_COLUMNS += [f"pod_feedback_{i}", f"pod_score_{i}"]

LEGACY_EXPORT_COLUMNS = {"Contractors", "Drivers", "Route_names"}
EXPORT_EXCLUDED_COLUMNS = set(POD_COLUMNS) | LEGACY_EXPORT_COLUMNS


REGION_BY_HUB = {
    "EDS": "EA",
    "ATL": "EA",
    "MIA": "EA",
    "ORL": "EA",
    "ONT": "WE",
    "HOU": "WE",
    "WDR": "WE",
}

STATE_ALIAS = {
    "CALIFORNIA": "CA",
    "ILLINOIS": "IL",
    "FLORIDA": "FL",
    "NEWJERSEY": "NJ",
    "NEW JERSEY": "NJ",
    "NEWYORK": "NY",
    "NEW YORK": "NY",
    "TEXAS": "TX",
    "GEORGIA": "GA",
    "PENNSYLVANIA": "PA",
    "CONNECTICUT": "CT",
}


HUB_ALIAS = {
    "GIA": "ORL",
}

HUB_BY_STATE = {
    "CA": "ONT",
    "IL": "WDR",
    "FL": "MIA",
    "NJ": "EDS",
    "NY": "EDS",
    "TX": "HOU",
    "GA": "ATL",
    "PA": "EDS",
    "CT": "EDS"
}

KNOWN_HUBS = set(HUB_BY_STATE.values()) | set(HUB_ALIAS.values()) | {"PU"}

# Existing DSP contractors from route naming convention.
KNOWN_DSP_CONTRACTORS = [
    "CBC",
    "GT",
    "FME",
    "BR",
    "BD",
    "GW",
    "DRX",
    "SLE",
    "LXE",
    "YLL",
    "EOI",
    "FF1",
    "EFB",
    "SEL",
    "GHH",
    "MYC",
    "NWB",
    "BXI",
    "KAT",
    "FEDEX",
    "GIA",
    "MET",
    "FNM",
]

ASSIGNEE_API_URL = read_config(
    "KPI_ASSIGNEE_API_URL",
    "https://isp.beans.ai/enterprise/v1/lists/assignees",
)
ASSIGNEE_CACHE_DIR = read_config("KPI_ASSIGNEE_CACHE_DIR", ".cache")
ASSIGNEE_CACHE_FILE = read_config(
    "KPI_ASSIGNEE_CACHE_FILE",
    os.path.join(ASSIGNEE_CACHE_DIR, "assignee_list_response.json"),
)

def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[col for col in df.columns if col not in EXPORT_EXCLUDED_COLUMNS]].copy()

def normalize_tracking_ids(raw_ids: list[str], uppercase: bool = False) -> tuple[list[str], list[str], Counter]:
    cleaned: list[str] = []
    for value in raw_ids:
        item = str(value).strip()
        if not item:
            continue
        cleaned.append(item.upper() if uppercase else item)

    counter = Counter(cleaned)
    unique_ids: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            unique_ids.append(item)
    return cleaned, unique_ids, counter

def split_text_ids(text: str) -> list[str]:
    if not text:
        return []
    return [x for x in re.split(r"[\s,]+", text) if x]


def parse_route(description: Any) -> str:
    text = "" if description is None else str(description)
    match = re.search(r"\broute\b\s*[:：-]?\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("\"' \t-:：")
    return ""


def _event_containers(event: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for candidate in (event, event.get("logItem"), event.get("log")):
        if isinstance(candidate, dict):
            containers.append(candidate)
    return containers


def _first_non_empty_dict_value(container: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = container.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def extract_list_route_id(event: dict[str, Any]) -> str:
    for container in _event_containers(event):
        route_obj = container.get("route")
        if isinstance(route_obj, dict):
            route_id = _first_non_empty_dict_value(route_obj, "listRouteId", "routeId", "id")
            if route_id:
                return route_id

        route_id = _first_non_empty_dict_value(container, "listRouteId", "routeId")
        if route_id:
            return route_id
    return ""


def extract_list_assignee_id(event: dict[str, Any]) -> str:
    for container in _event_containers(event):
        assignee_obj = container.get("assignee")
        if isinstance(assignee_obj, dict):
            assignee_id = _first_non_empty_dict_value(assignee_obj, "listAssigneeId", "assigneeId", "id")
            if assignee_id:
                return assignee_id

        assignee_id = _first_non_empty_dict_value(container, "listAssigneeId", "assigneeId")
        if assignee_id:
            return assignee_id
    return ""


def extract_route_name_from_event(event: dict[str, Any]) -> str:
    for container in _event_containers(event):
        route_name = parse_route(container.get("description"))
        if route_name:
            return route_name

        route_obj = container.get("route")
        if isinstance(route_obj, dict):
            route_name = _first_non_empty_dict_value(route_obj, "name", "routeName", "listRouteName")
            if route_name:
                return route_name
        elif isinstance(route_obj, str):
            route_name = str(route_obj).strip()
            if route_name:
                return route_name

        route_name = _first_non_empty_dict_value(container, "routeName", "listRouteName")
        if route_name:
            return route_name
    return ""

def latest_route_assignment(events: list[dict[str, Any]]) -> str:
    candidates: list[tuple[int, int, str, bool]] = []
    for idx, event in enumerate(events):
        route_name = extract_route_name_from_event(event)
        if not route_name:
            continue

        route_info = parse_route_identity(route_name)
        is_readable = bool(route_info["Hub"] and route_info["Contractor"])

        ts = event_ts(event)
        sort_ts = ts if ts is not None else -1
        candidates.append((sort_ts, idx, route_name, is_readable))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (x[0], x[1]))
    for _, _, route_name, is_readable in reversed(candidates):
        if is_readable:
            return route_name

    return candidates[-1][2]


def extract_all_route_assignments(events: list[dict[str, Any]]) -> list[str]:
    routes: list[str] = []
    seen: set[str] = set()
    for event in events:
        route_name = extract_route_name_from_event(event)
        normalized = route_name.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        routes.append(normalized)
    return routes


def choose_primary_route(ofd_route: str, failed_route: str, success_route: str, fallback_route: str) -> str:
    """Pick the primary route for KPI attribution.

    Delivery-rate denominator should follow the route that actually attempted OFD first,
    instead of the latest reassignment route.
    """
    for route_name in (ofd_route, failed_route, success_route, fallback_route):
        candidate = str(route_name or "").strip()
        if candidate:
            return candidate
    return ""


def _normalize_route_meta_entry(entry: dict[str, Any] | None = None) -> dict[str, str]:
    source = entry or {}
    return {
        "route_name": str(source.get("route_name") or "").strip(),
        "listAssigneeId": str(source.get("listAssigneeId") or "").strip(),
        "driver": str(source.get("driver") or "").strip(),
        "contractor": str(source.get("contractor") or "").strip(),
        "hub": str(source.get("hub") or "").strip(),
        "listWarehouseId": str(source.get("listWarehouseId") or "").strip(),
    }


def _merge_route_meta(base: dict[str, str], incoming: dict[str, Any]) -> dict[str, str]:
    merged = _normalize_route_meta_entry(base)
    for key in ("route_name", "listAssigneeId", "driver", "contractor", "hub", "listWarehouseId"):
        value = str(incoming.get(key) or "").strip()
        if value and not merged.get(key):
            merged[key] = value
    if merged.get("listWarehouseId") and not merged.get("hub"):
        merged["hub"] = merged["listWarehouseId"]
    return merged


def _route_name_fallback_meta(route_name: str) -> dict[str, str]:
    parsed = parse_route_identity(route_name)
    return {
        "route_name": str(route_name or "").strip(),
        "driver": str(parsed.get("Driver") or "").strip(),
        "contractor": str(parsed.get("Contractor") or "").strip(),
        "hub": str(parsed.get("Hub") or "").strip(),
    }


def _extract_assignee_records(payload: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            assignee_id = _first_non_empty(
                node.get("id"),
                node.get("listAssigneeId"),
                node.get("assigneeId"),
            )
            if assignee_id:
                name = _first_non_empty(node.get("name"), node.get("assigneeName"))
                contractor = _first_non_empty(node.get("contractor"), node.get("company"), node.get("vendor"))
                warehouse_id = _first_non_empty(
                    node.get("listWarehouseId"),
                    node.get("warehouseId"),
                )
                if name or contractor or warehouse_id:
                    records.append(
                        {
                            "listAssigneeId": assignee_id,
                            "driver": name,
                            "contractor": contractor,
                            "listWarehouseId": warehouse_id,
                            "hub": warehouse_id,
                        }
                    )
            for value in node.values():
                _walk(value)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return records


def load_assignee_payload_cached(force_refresh: bool = False) -> Any:
    cache_file = str(ASSIGNEE_CACHE_FILE or "").strip()
    if not cache_file:
        return None

    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass

    if not ASSIGNEE_API_URL:
        return None

    headers = build_api_headers()
    response = requests.get(ASSIGNEE_API_URL, headers=headers, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)
    return payload


def _load_assignee_cache(assignee_ids: set[str], assignee_payload: Any | None = None) -> dict[str, dict[str, str]]:
    if not assignee_ids:
        return {}

    payload = assignee_payload
    if payload is None:
        payload = load_assignee_payload_cached()

    assignee_cache: dict[str, dict[str, str]] = {}
    for record in _extract_assignee_records(payload):
        assignee_id = str(record.get("listAssigneeId") or "").strip()
        if not assignee_id or assignee_id not in assignee_ids:
            continue
        assignee_cache[assignee_id] = _merge_route_meta({}, record)
    return assignee_cache


def build_route_metadata_map(
    router_messages_map: dict[str, Any],
    assignee_payload: Any | None = None,
) -> dict[str, dict[str, str]]:
    route_metadata_map: dict[str, dict[str, str]] = {}
    needed_assignee_ids: set[str] = set()

    for payload in router_messages_map.values():
        events = normalize_events(payload)
        if not events:
            continue

        for event in events:
            route_id = extract_list_route_id(event)
            if not route_id:
                continue

            evt_type = event_type(event)
            route_name = extract_route_name_from_event(event)
            assignee_id = extract_list_assignee_id(event)

            incoming_meta: dict[str, str] = {
                "route_name": route_name if evt_type == "out-for-delivery" else "",
                "listAssigneeId": assignee_id,
            }
            if route_name:
                incoming_meta = _merge_route_meta(incoming_meta, _route_name_fallback_meta(route_name))

            current = route_metadata_map.get(route_id, _normalize_route_meta_entry())
            route_metadata_map[route_id] = _merge_route_meta(current, incoming_meta)

            resolved_assignee_id = route_metadata_map[route_id].get("listAssigneeId", "")
            if resolved_assignee_id and (
                not route_metadata_map[route_id].get("driver")
                or not route_metadata_map[route_id].get("contractor")
                or not route_metadata_map[route_id].get("hub")
            ):
                needed_assignee_ids.add(resolved_assignee_id)

    if not needed_assignee_ids:
        return route_metadata_map

    try:
        assignee_cache = _load_assignee_cache(needed_assignee_ids, assignee_payload=assignee_payload)
    except Exception:
        assignee_cache = {}

    for route_id, route_meta in list(route_metadata_map.items()):
        assignee_id = route_meta.get("listAssigneeId", "")
        if not assignee_id:
            continue
        assignee_meta = assignee_cache.get(assignee_id)
        if not assignee_meta:
            continue
        route_metadata_map[route_id] = _merge_route_meta(route_meta, assignee_meta)

    return route_metadata_map


def resolve_route_metadata_for_event(
    event: dict[str, Any],
    route_metadata_map: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    route_id = extract_list_route_id(event)
    route_name = extract_route_name_from_event(event)
    assignee_id = extract_list_assignee_id(event)

    resolved = _normalize_route_meta_entry(route_metadata_map.get(route_id) if route_metadata_map and route_id else None)
    if route_name:
        resolved = _merge_route_meta(resolved, _route_name_fallback_meta(route_name))
        if not resolved.get("route_name"):
            resolved["route_name"] = route_name
    if assignee_id and not resolved.get("listAssigneeId"):
        resolved["listAssigneeId"] = assignee_id
    resolved["listRouteId"] = route_id
    if resolved.get("listWarehouseId") and not resolved.get("hub"):
        resolved["hub"] = resolved["listWarehouseId"]
    return resolved



def extract_route_parts(route_name: str) -> list[str]:
    text = str(route_name or "").strip()
    if not text:
        return []

    dash_parts = [p.strip() for p in re.split(r"\s*[-–—]+\s*", text) if p and p.strip()]
    if len(dash_parts) >= 2:
        return dash_parts

    return [p.strip() for p in text.split() if p and p.strip()]


def is_valid_hub_name(hub: str) -> bool:
    """Hub must be exactly 3 letters (A-Z)."""
    hub_text = str(hub or "").strip()
    return bool(re.fullmatch(r"[A-Za-z]{3}", hub_text)) or hub_text.lower() == "pu"


def is_valid_contractor_name(contractor: str) -> bool:
    """Contractor supports 2-5 alphanumerics and must contain at least one letter."""
    text = re.sub(r"[^A-Za-z0-9]", "", str(contractor or "").strip().upper())
    if not re.fullmatch(r"[A-Z0-9]{2,5}", text):
        return False
    return bool(re.search(r"[A-Z]", text))


def infer_hub_from_state(state: str) -> str:
    normalized_state = normalize_state(state)
    return HUB_BY_STATE.get(normalized_state, "")



def normalize_contractor_name(contractor: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(contractor or "").strip().upper())


def match_known_contractor(token: str) -> str:
    candidate = normalize_contractor_name(token)
    if not candidate:
        return ""

    if candidate in KNOWN_DSP_CONTRACTORS:
        return candidate
    return ""


def extract_contractor_by_keywords(route_name: str) -> str:
    route_text = str(route_name or "").upper()
    if not route_text:
        return ""
    special_patterns: list[tuple[str, str]] = [
        (r"\bGTN\b", "GT"),
        (r"\bEO\b", "EOI"),
        (r"\bDX\b", "DRX"),
        (r"\bFF\b", "FF1"),
        (r"\bMET\b", "MET"),
        (r"\bFNM\b", "FNM"),
        (r"\bFINAL\s*MILE\b", "FNM"),
        (r"\bYULIN\b", "YLL"),
    ]
    for pattern, contractor in special_patterns:
        if re.search(pattern, route_text):
            return contractor
            
    for contractor in KNOWN_DSP_CONTRACTORS:
        pattern = rf"(?<![A-Z0-9]){re.escape(contractor)}(?![A-Z0-9])"
        if re.search(pattern, route_text):
            return contractor

    compact_route_text = re.sub(r"[^A-Z0-9]", "", route_text)
    for contractor in sorted(KNOWN_DSP_CONTRACTORS, key=len, reverse=True):
        if len(contractor) < 3:
            continue
        if contractor in compact_route_text:
            return contractor

    return ""



def _is_single_adjacent_swap(source: str, target: str) -> bool:
    if len(source) != len(target):
        return False

    diffs = [idx for idx, (s_char, t_char) in enumerate(zip(source, target)) if s_char != t_char]
    if len(diffs) != 2:
        return False

    first, second = diffs
    return second == first + 1 and source[first] == target[second] and source[second] == target[first]


def normalize_hub_name(hub: str, fallback_state: str = "") -> str:
    hub_text = str(hub or "").strip().upper()
    if not hub_text:
        return infer_hub_from_state(fallback_state)

    hub_compact = re.sub(r"[^A-Z]", "", hub_text)
    hub_compact = HUB_ALIAS.get(hub_compact, hub_compact)
    if hub_compact in KNOWN_HUBS:
        return hub_compact

    # Auto-fix common 3-letter typos caused by adjacent letter swaps, e.g. ALT -> ATL.
    if len(hub_compact) == 3:
        typo_candidates = [known for known in KNOWN_HUBS if len(known) == 3 and _is_single_adjacent_swap(hub_compact, known)]
        if len(typo_candidates) == 1:
            return typo_candidates[0]

    if is_valid_hub_name(hub_compact):
        return hub_compact

    return infer_hub_from_state(hub_compact) or infer_hub_from_state(fallback_state)



def looks_like_driver_token(token: str) -> bool:
    text = str(token or "").strip().upper()
    return bool(re.fullmatch(r"[A-Z]{4,}", text))

def looks_like_route_date_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    return bool(re.fullmatch(r"\d{1,2}[/\-.]\d{1,2}", text))



def parse_route_identity(route_name: str, fallback_state: str = "") -> dict[str, str]:
    """Parse route format: HUB-路区号-日期-DSP-司机名.

    Be tolerant to mixed separators and minor format issues.
    """
    parts = extract_route_parts(route_name)
    fallback_hub = infer_hub_from_state(fallback_state)
    route_text = str(route_name or "").upper()

    is_pickup_route = bool(re.search(r"\bPU\b", route_text) or re.search(r"\bPICK\s*UP\b", route_text))
    if is_pickup_route:
        return {
            "Hub": "PU",
            "Contractor": "",
            "Driver": "",
            "Route_type": "pickup",
        }

    if len(parts) < 2:

        return {
            "Hub": fallback_hub,
            "Contractor": "",
            "Driver": "",
            "Route_type": "pickup" if fallback_hub == "PU" else "delivery",
        }

    contractor = ""
    driver = ""
    contractor_idx = -1

    date_idx = -1
    for idx, token in enumerate(parts):
        if looks_like_route_date_token(token):
            date_idx = idx
            break

    if date_idx >= 0 and date_idx + 1 < len(parts):
        raw_candidate = parts[date_idx + 1].strip()
        candidate = match_known_contractor(raw_candidate)
        if candidate:
            contractor = candidate
            contractor_idx = date_idx + 1

    if contractor_idx < 0:
        for idx in range(len(parts) - 1, 0, -1):
            raw_candidate = parts[idx].strip().upper()
            candidate = normalize_contractor_name(raw_candidate)
            if "/" in raw_candidate:
                continue
            if idx == len(parts) - 1 and len(parts) >= 3 and looks_like_driver_token(candidate):
                continue
            matched = match_known_contractor(candidate)
            if matched:
                contractor = matched
                contractor_idx = idx
                break

    if contractor_idx >= 0:
        driver_tokens = [token.strip() for token in parts[contractor_idx + 1 :] if token.strip()]
        if driver_tokens:
            driver = " ".join(driver_tokens).title()
    elif len(parts) >= 2:
        driver = parts[-1].strip().title()

    route_hub_candidate = normalize_hub_name(parts[0], fallback_state=fallback_state) if parts else ""
    hub = route_hub_candidate or fallback_hub

    if contractor and not match_known_contractor(contractor):
        contractor = ""

    if re.search(r"\bGIA\b", route_text):
        hub = "ORL"
        contractor = "GIA"
    elif re.search(r"\bMIA\b", route_text):
        hub = "MIA"
        contractor = "GT"

    keyword_contractor = extract_contractor_by_keywords(route_name)
    if keyword_contractor:
        contractor = keyword_contractor

    if normalize_state(fallback_state) == "IL" and contractor != "MET":
        contractor = contractor or "IL"

    route_type = "pickup" if hub == "PU" else "delivery"

    return {
        "Hub": hub,
        "Contractor": contractor,
        "Driver": driver,
        "Route_type": route_type,
    }


def normalize_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if not isinstance(payload, dict):
        return []

    root = payload
    for key in ("data", "result", "response"):
        if isinstance(root.get(key), dict):
            root = root[key]
            break

    candidates = [
        root.get("listItemReadableStatusLogs"),
        root.get("listItemStatusLogs"),
        root.get("status_logs"),
        root.get("statusLogs"),
        root.get("logs"),
        root.get("events"),
        root.get("trackingEvents"),
        root.get("history"),
        root.get("checkpoints"),
    ]
    for events in candidates:
        if isinstance(events, list):
            return [e for e in events if isinstance(e, dict)]
    return []


EVENT_TYPE_ALIASES = {
    "out for delivery": "out-for-delivery",
    "out-for-delivery": "out-for-delivery",
    "ofd": "out-for-delivery",
    "delivered": "success",
    "delivery success": "success",
    "success": "success",
    "failed": "fail",
    "failure": "fail",
    "delivery failed": "fail",
    "attempted": "fail",
    "cancelled": "cancel",
    "canceled": "cancel",
}


def _normalize_event_type_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return ""

    compact = re.sub(r"\s+", " ", text)
    if compact in EVENT_TYPE_ALIASES:
        return EVENT_TYPE_ALIASES[compact]

    if "out" in compact and "delivery" in compact:
        return "out-for-delivery"
    if "deliver" in compact and "fail" in compact:
        return "fail"
    if "deliver" in compact and "success" in compact:
        return "success"
    if compact in {"warehouse", "scan", "picked-up", "picked up", "pickup"}:
        return compact.replace(" ", "-")

    return compact


def event_type(event: dict[str, Any]) -> str:
    for key in (
        "type",
        "eventType",
        "status",
        "readableStatus",
        "itemReadableStatus",
        "statusName",
        "itemStatus",
        "nodeStatus",
    ):
        val = event.get(key)
        if val:
            normalized = _normalize_event_type_text(val)
            if normalized:
                return normalized

    log_item = event.get("logItem")
    if isinstance(log_item, dict):
        for key in (
            "type",
            "eventType",
            "status",
            "readableStatus",
            "itemReadableStatus",
            "statusName",
            "itemStatus",
            "nodeStatus",
        ):
            val = log_item.get(key)
            if val:
                normalized = _normalize_event_type_text(val)
                if normalized:
                    return normalized

    log_obj = event.get("log")
    if isinstance(log_obj, dict):
        for key in (
            "type",
            "eventType",
            "status",
            "readableStatus",
            "itemReadableStatus",
            "statusName",
            "itemStatus",
            "nodeStatus",
        ):
            val = log_obj.get(key)
            if val:
                normalized = _normalize_event_type_text(val)
                if normalized:
                    return normalized

    description = event_description(event).lower()
    if "out for delivery" in description:
        return "out-for-delivery"
    if "delivery failed" in description or "attempted" in description:
        return "fail"
    if "delivered" in description:
        return "success"

    return ""


def event_ts(event: dict[str, Any]) -> int | None:
    pod = event.get("pod")
    if isinstance(pod, dict) and pod.get("podTimestampEpoch") is not None:
        try:
            return int(float(pod.get("podTimestampEpoch")) * 1000)
        except (TypeError, ValueError):
            pass

    log_item = event.get("logItem")
    if isinstance(log_item, dict):
        log_item_pod = log_item.get("pod")
        if isinstance(log_item_pod, dict) and log_item_pod.get("podTimestampEpoch") is not None:
            try:
                return int(float(log_item_pod.get("podTimestampEpoch")) * 1000)
            except (TypeError, ValueError):
                pass
        for key in ("tsMillis", "timestamp", "ts", "timeMillis"):
            val = log_item.get(key)
            try:
                if val is not None:
                    return int(val)
            except (ValueError, TypeError):
                continue

    log_obj = event.get("log")
    if isinstance(log_obj, dict):
        log_pod = log_obj.get("pod")
        if isinstance(log_pod, dict) and log_pod.get("podTimestampEpoch") is not None:
            try:
                return int(float(log_pod.get("podTimestampEpoch")) * 1000)
            except (TypeError, ValueError):
                pass
        for key in ("tsMillis", "timestamp", "ts", "timeMillis"):
            val = log_obj.get(key)
            try:
                if val is not None:
                    return int(val)
            except (ValueError, TypeError):
                continue

    for key in ("tsMillis", "timestamp", "ts", "timeMillis"):
        val = event.get(key)
        try:
            if val is not None:
                return int(val)
        except (ValueError, TypeError):
            continue
    return None




def events_by_predicate(events: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    filtered = [e for e in events if predicate(e)]
    with_ts = [(event_ts(e), idx, e) for idx, e in enumerate(filtered)]
    with_ts.sort(key=lambda x: ((x[0] if x[0] is not None else -1), x[1]))
    return [item[2] for item in with_ts]
def first_event_by_predicate(events: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    filtered = [e for e in events if predicate(e)]
    if not filtered:
        return None

    with_ts = [(event_ts(e), idx, e) for idx, e in enumerate(filtered)]
    with_ts.sort(key=lambda x: (10**18 if x[0] is None else x[0], x[1]))
    return with_ts[0][2]


def count_pod_stats(row: dict[str, str] | pd.Series) -> tuple[int, int]:
    pod_count = 0
    non_zero_scored_count = 0
    for i in range(1, POD_IMAGE_EXPORT_N + 1):
        feedback = str(row.get(f"pod_feedback_{i}") or "").strip()
        score = str(row.get(f"pod_score_{i}") or "").strip()
        if feedback or score:
            pod_count += 1
        if score:
            try:
                if float(score) != 0:
                    non_zero_scored_count += 1
            except (TypeError, ValueError):
                non_zero_scored_count += 1
    return pod_count, non_zero_scored_count

def auto_is_pod_compliant(row: dict[str, str] | pd.Series) -> bool:
    pod_count, non_zero_scored_count = count_pod_stats(row)
    return pod_count >= 3 and non_zero_scored_count >= 1

def last_event_by_predicate(events: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    filtered = [e for e in events if predicate(e)]
    if not filtered:
        return None

    with_ts = [(event_ts(e), idx, e) for idx, e in enumerate(filtered)]
    with_ts.sort(key=lambda x: (-1 if x[0] is None else x[0], x[1]))
    return with_ts[-1][2]


def extract_shipper_name_from_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        item = event.get("item")
        if isinstance(item, dict):
            name = item.get("shipperName")
            if name:
                return str(name)
    return ""


def _extract_pod_images_from_container(container: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(container, dict):
        return []

    all_images: list[dict[str, Any]] = []

    def _collect_images(pod_entry: Any) -> list[dict[str, Any]]:
        if not isinstance(pod_entry, dict):
            return []
        collected: list[dict[str, Any]] = []
        for key in ("images", "image", "photos", "podImages"):
            value = pod_entry.get(key)
            if isinstance(value, list):
                collected.extend([x for x in value if isinstance(x, dict)])
            elif isinstance(value, dict):
                collected.append(value)
        return collected

    pod_obj = container.get("pod")
    if isinstance(pod_obj, dict):
        all_images.extend(_collect_images(pod_obj))
    elif isinstance(pod_obj, list):
        for pod_entry in pod_obj:
            all_images.extend(_collect_images(pod_entry))

    pods_obj = container.get("pods")
    if isinstance(pods_obj, dict):
        pod_list = pods_obj.get("pod")
        if isinstance(pod_list, list):
            for pod_entry in pod_list:
                all_images.extend(_collect_images(pod_entry))
        elif isinstance(pod_list, dict):
            all_images.extend(_collect_images(pod_list))

    return all_images


def _pod_image_quality_fields(image: dict[str, Any]) -> tuple[str, str]:
    quality = image.get("quality") if isinstance(image.get("quality"), dict) else image
    if not isinstance(quality, dict):
        return "", ""

    feedback = str(
        quality.get("feedback")
        or quality.get("qualifiedFeedback")
        or quality.get("qualityFeedback")
        or ""
    ).strip()
    score = str(
        quality.get("score")
        or quality.get("qualifiedScore")
        or quality.get("qualityScore")
        or ""
    ).strip()
    return feedback, score


def _payload_has_pod_marker(payload: Any) -> bool:
    """Check whether any node in payload carries POD presence flags."""

    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            pod_value = node.get("pod")
            if isinstance(pod_value, dict) and pod_value:
                return True
            if isinstance(pod_value, list) and any(isinstance(item, dict) for item in pod_value):
                return True
            if _truthy(pod_value):
                return True

            for flag_key in ("hasPod", "isPod", "podAvailable", "podUploaded"):
                if _truthy(node.get(flag_key)):
                    return True

            return any(_walk(value) for value in node.values())

        if isinstance(node, list):
            return any(_walk(item) for item in node)

        return False

    return _walk(payload)


def _event_has_pod_marker(event: dict[str, Any] | None, payload: Any = None) -> bool:
    if not isinstance(event, dict):
        return False

    if _payload_has_pod_marker(event) or _payload_has_pod_marker(event.get("logItem")) or _payload_has_pod_marker(event.get("log")):
        return True

    if payload is not None and _payload_has_pod_marker(payload):
        return True

    return False


def extract_pod_images_from_success_event(success_evt: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not success_evt:
        return []

    all_images: list[dict[str, Any]] = []

    for container in (success_evt, success_evt.get("logItem"), success_evt.get("log")):
        all_images.extend(_extract_pod_images_from_container(container))

    return all_images


def extract_pod_images_from_payload(payload: Any) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            images.extend(_extract_pod_images_from_container(node))
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return images


def is_pod_compliant_for_event(event: dict[str, Any] | None, payload: Any = None) -> bool:
    """Temporary POD rule: qualify when extracted POD images count is >= 3."""
    if not event:
        return False

    pod_images = extract_pod_images_from_success_event(event)
    if len(pod_images) < 3 and isinstance(payload, (dict, list)):
        fallback_images = extract_pod_images_from_payload(payload)
        seen_urls = {str(img.get("url") or "").strip() for img in pod_images if isinstance(img, dict)}
        for image in fallback_images:
            image_url = str(image.get("url") or "").strip()
            if image_url and image_url in seen_urls:
                continue
            pod_images.append(image)
            if image_url:
                seen_urls.add(image_url)

    return len(pod_images) >= 3


def legacy_is_pod_compliant_for_event(event: dict[str, Any] | None, payload: Any = None) -> bool:
    """Previous POD rule (kept for reference, not applied): image count + quality/score checks."""
    if not event:
        return False

    # Some carriers only return POD flags/timestamps without complete image arrays.
    # In that case, treat it as POD-qualified to avoid false negatives in interval output.
    if _event_has_pod_marker(event, payload=payload):
        return True

    pod_images = extract_pod_images_from_success_event(event)
    if len(pod_images) < 3 and isinstance(payload, (dict, list)):
        fallback_images = extract_pod_images_from_payload(payload)
        seen_urls = {str(img.get("url") or "").strip() for img in pod_images if isinstance(img, dict)}
        for image in fallback_images:
            image_url = str(image.get("url") or "").strip()
            if image_url and image_url in seen_urls:
                continue
            pod_images.append(image)
            if image_url:
                seen_urls.add(image_url)

    pod_count = len(pod_images)
    non_zero_scored_count = 0
    for img in pod_images:
        feedback, score = _pod_image_quality_fields(img)
        if score:
            try:
                if float(score) != 0:
                    non_zero_scored_count += 1
            except (TypeError, ValueError):
                non_zero_scored_count += 1
        elif feedback:
            non_zero_scored_count += 1
    if pod_count >= 3 and non_zero_scored_count >= 1:
        return True

    return False


def build_intervals(
    events: list[dict[str, Any]],
    payload: dict[str, Any] | None = None,
    route_metadata_map: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []

    def interval_ts(event: dict[str, Any]) -> int | None:
        for container in (event, event.get("logItem"), event.get("log")):
            if not isinstance(container, dict):
                continue
            for key in ("tsMillis", "timestamp", "ts", "timeMillis"):
                value = container.get(key)
                try:
                    if value is not None:
                        return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def find_scanned_at_description(start_idx: int) -> str:
        for candidate_event in events[start_idx:]:
            candidate_desc = event_description(candidate_event).strip()
            if "scanned at" in candidate_desc.lower():
                return candidate_desc
        return ""

    for idx, event in enumerate(events):
        ts = interval_ts(event)
        evt_type = event_type(event)
        description = event_description(event)
        if not evt_type:
            continue

        if "customer service" in description.lower():
            interval_type = "Customer Service"
        else:
            interval_type = evt_type

        node: dict[str, Any] = {
            "time": ts,
            "type": interval_type,
        }

        if evt_type == "warehouse" and description.strip():
            node["description"] = description.strip()
        elif evt_type in {"sort", "sorting"}:
            sort_desc = find_scanned_at_description(idx)
            if sort_desc:
                node["description"] = sort_desc

        route_meta = resolve_route_metadata_for_event(event, route_metadata_map=route_metadata_map)

        if evt_type in {"fail", "failed", "failure", "out-for-delivery", "ofd", "outfordelivery", "success", "delivered"}:
            route = str(route_meta.get("route_name") or "").strip()
            if route:
                node["route"] = route
            route_id = str(route_meta.get("listRouteId") or "").strip()
            if route_id:
                node["listRouteId"] = route_id
            assignee_id = str(route_meta.get("listAssigneeId") or "").strip()
            if assignee_id:
                node["listAssigneeId"] = assignee_id

        if evt_type in {"fail", "failed", "failure", "success", "delivered"}:
            node["POD"] = is_pod_compliant_for_event(event, payload=payload)

        intervals.append(node)
    return intervals


def event_description(event: dict[str, Any]) -> str:
    for container in (event, event.get("logItem"), event.get("log")):
        if isinstance(container, dict):
            desc = container.get("description")
            if desc:
                return str(desc)
    return ""


def has_customer_service_record(events: list[dict[str, Any]]) -> bool:
    for event in events:
        if "customer service" in event_description(event).lower():
            return True
    return False


def extract_hub_from_scan_description(description: str) -> str:
    desc = str(description or "").strip()
    if not desc:
        return ""

    if "g.t. miami" in desc.lower():
        return "MIA"

    match = re.search(r"\bFM[_\-\s]*([A-Za-z]{3})\b", desc, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def extract_hub_name_from_warehouse_description(description: str) -> str:
    desc = str(description or "").strip()
    if not desc:
        return ""

    match = re.search(r"(?i)\bscanned\s+at\s+(.+)$", desc)
    if not match:
        return ""

    return match.group(1).strip(" .")


def infer_hub_from_pre_ofd_warehouse(events: list[dict[str, Any]], ofd_evt: dict[str, Any] | None) -> str:
    ofd_ts = event_ts(ofd_evt) if ofd_evt else None
    latest_event: dict[str, Any] | None = None
    latest_ts: int | None = None

    for evt in events:
        if event_type(evt) != "warehouse":
            continue

        evt_ts = event_ts(evt)
        if ofd_ts is not None and evt_ts is not None and evt_ts > ofd_ts:
            continue

        description = event_description(evt)
        if not extract_hub_name_from_warehouse_description(description):
            continue

        evt_ts_for_cmp = event_ts(evt)
        if latest_event is None:
            latest_event = evt
            latest_ts = evt_ts_for_cmp
            continue

        # Keep the latest event and preserve stable ordering for equal timestamps.
        if latest_ts is None or (evt_ts_for_cmp is not None and evt_ts_for_cmp >= latest_ts):
            latest_event = evt
            latest_ts = evt_ts_for_cmp

    if latest_event is None:
        return ""

    return extract_hub_name_from_warehouse_description(event_description(latest_event))


def infer_hub_from_pre_ofd_scan(events: list[dict[str, Any]], ofd_evt: dict[str, Any] | None) -> str:
    ofd_ts = event_ts(ofd_evt) if ofd_evt else None
    target_evt: dict[str, Any] | None = None
    target_ts: int | None = None

    for evt in events:
        description = event_description(evt)
        desc_lower = description.strip().lower()
        if not (desc_lower.startswith("scan at") or desc_lower.startswith("scanned at")):
            continue

        evt_ts = event_ts(evt)
        if ofd_ts is not None and evt_ts is not None and evt_ts > ofd_ts:
            continue

        evt_ts_for_cmp = event_ts(evt)
        if target_evt is None:
            target_evt = evt
            target_ts = evt_ts_for_cmp
            continue

        # Keep the latest scan event while preserving stable ordering for equal timestamps.
        if target_ts is None or (evt_ts_for_cmp is not None and evt_ts_for_cmp >= target_ts):
            target_evt = evt
            target_ts = evt_ts_for_cmp

    if target_evt is None:
        return ""

    return extract_hub_from_scan_description(event_description(target_evt))


def build_row(
    tracking_id: str,
    payload: Any,
    route_metadata_map: dict[str, dict[str, str]] | None = None,
    include_dimensions: bool = True,
) -> dict[str, str]:
    events = normalize_events(payload)
    intervals = build_intervals(events, payload=payload, route_metadata_map=route_metadata_map)
    is_delivered = any(str(x.get("type") or "").strip().lower() in {"success", "delivered"} for x in intervals)

    first_scanned_interval = next(
        (
            item
            for item in intervals
            if str(item.get("type") or "").strip().lower() == "warehouse"
            or (
                str(item.get("type") or "").strip().lower() == "sort"
                and "scanned at" in str(item.get("description") or "").strip().lower()
            )
        ),
        None,
    )

    ofd_events = events_by_predicate(events, lambda e: event_type(e) in {"out-for-delivery", "ofd", "outfordelivery"})
    fail_events = events_by_predicate(events, lambda e: event_type(e) in {"fail", "failed", "failure"})
    success_events = events_by_predicate(events, lambda e: event_type(e) in {"success", "delivered"})

    fail_evt = fail_events[0] if fail_events else None
    success_evt = success_events[0] if success_events else None
    route_assignments = extract_all_route_assignments(events)
    
    created_time_ms = None
    if intervals:
        first_interval_time = intervals[0].get("time")
        try:
            if first_interval_time is not None:
                created_time_ms = int(first_interval_time)
        except (TypeError, ValueError):
            created_time_ms = None

    created_time = to_local_dt(created_time_ms)
    first_scanned_time = to_local_dt(first_scanned_interval.get("time") if first_scanned_interval else None)
    delivered_time = to_local_dt(event_ts(success_evt) if success_evt else None)

    structured_identity = extract_route_identity_from_payload(payload)

    route_names: list[str] = []
    contractors: list[str] = []
    drivers: list[str] = []
    hubs: list[str] = []
    seen_route_names: set[str] = set()
    seen_contractors: set[str] = set()
    seen_drivers: set[str] = set()
    seen_hubs: set[str] = set()

    for event in ofd_events:
        route_meta = resolve_route_metadata_for_event(event, route_metadata_map=route_metadata_map)
        route_name = str(route_meta.get("route_name") or "").strip()
        if route_name and route_name.lower() not in seen_route_names:
            seen_route_names.add(route_name.lower())
            route_names.append(route_name)
        contractor_name = str(route_meta.get("contractor") or "").strip()
        if contractor_name and contractor_name.lower() not in seen_contractors:
            seen_contractors.add(contractor_name.lower())
            contractors.append(contractor_name)
        driver_name = str(route_meta.get("driver") or "").strip()
        if driver_name and driver_name.lower() not in seen_drivers:
            seen_drivers.add(driver_name.lower())
            drivers.append(driver_name)
        hub_name = str(route_meta.get("hub") or "").strip()
        if hub_name and hub_name.lower() not in seen_hubs:
            seen_hubs.add(hub_name.lower())
            hubs.append(hub_name)

    if not route_names:
        for route_name in route_assignments:
            normalized_name = str(route_name or "").strip()
            if normalized_name and normalized_name.lower() not in seen_route_names:
                seen_route_names.add(normalized_name.lower())
                route_names.append(normalized_name)

    if not contractors:
        fallback_contractor = str(structured_identity.get("Contractor") or "").strip()
        if fallback_contractor:
            contractors.append(fallback_contractor)
    if not drivers:
        fallback_driver = str(structured_identity.get("Driver") or "").strip()
        if fallback_driver:
            drivers.append(fallback_driver)
    if not hubs:
        fallback_hub = str(structured_identity.get("Hub") or "").strip()
        if fallback_hub:
            hubs.append(fallback_hub)

    primary_hub = hubs[0] if hubs else ""
    route_type = str(structured_identity.get("Route_type") or "delivery").strip()
    if primary_hub.upper() == "PU":
        route_type = "pickup"

    row: dict[str, str] = {
        "tracking_id": tracking_id,
        "Hub": primary_hub,
        "Contractors": json.dumps(contractors, ensure_ascii=False),
        "Drivers": json.dumps(drivers, ensure_ascii=False),
        "Route_names": json.dumps(route_names, ensure_ascii=False),
        "Weight": _extract_weight_from_payload(payload) if include_dimensions else "",
        "Volume": _extract_volume_from_payload(payload) if include_dimensions else "",
        "created_time": fmt_dt(created_time),
        "first_scanned_time": fmt_dt(first_scanned_time),
        "last_scanned_time": fmt_dt(to_local_dt(intervals[-1].get("time") if intervals else None)),
        "delivered_time": fmt_dt(delivered_time),
        "entered_costomer_service": "",
        "beans_pod_link": build_beans_tracking_link(tracking_id),
        "Route_type": route_type,
        "Intervals": json.dumps(intervals, ensure_ascii=False),
        "Is_delivered": "true" if is_delivered else "false",
    }

    for i in range(1, POD_IMAGE_EXPORT_N + 1):
        row[f"pod_feedback_{i}"] = ""
        row[f"pod_score_{i}"] = ""

    pod_images = extract_pod_images_from_success_event(success_evt)
    for i, img in enumerate(pod_images[:POD_IMAGE_EXPORT_N], start=1):
        feedback, score = _pod_image_quality_fields(img)
        row[f"pod_feedback_{i}"] = feedback
        row[f"pod_score_{i}"] = score

    return row


def _find_values_by_key(obj: Any, key: str, limit: int = 5) -> list[Any]:
    results: list[Any] = []

    def _walk(node: Any) -> None:
        if len(results) >= limit:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and v not in (None, ""):
                    results.append(v)
                    if len(results) >= limit:
                        return
                _walk(v)
                if len(results) >= limit:
                    return
        elif isinstance(node, list):
            for item in node:
                _walk(item)
                if len(results) >= limit:
                    return

    _walk(obj)
    return results


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_numeric_dimension_from_payload(payload: Any, dimension_type: str) -> str:
    values: list[float] = []
    target_dimension_type = str(dimension_type or "").strip().upper()

    def _parse_numeric(raw_value: Any) -> float | None:
        text = str(raw_value or "").strip()
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            dims = node.get("dims")
            if isinstance(dims, list):
                for dim in dims:
                    if not isinstance(dim, dict):
                        continue
                    dim_type = str(dim.get("t") or "").strip().upper()
                    dim_value = dim.get("v")
                    if dim_type == target_dimension_type:
                        parsed = _parse_numeric(dim_value)
                        if parsed is not None:
                            values.append(parsed)
                    elif target_dimension_type == "WEIGHT" and isinstance(dim_value, str) and dim_value.lower().startswith("pw:"):
                        parsed = _parse_numeric(dim_value.split(":", 1)[-1])
                        if parsed is not None:
                            values.append(parsed)

            for value in node.values():
                _walk(value)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    if not values:
        return ""
    value = values[0]
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_weight_from_payload(payload: Any) -> str:
    return _extract_numeric_dimension_from_payload(payload, "WEIGHT")


def _extract_volume_from_payload(payload: Any) -> str:
    return _extract_numeric_dimension_from_payload(payload, "VOLUME")


def extract_route_identity_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    route_name = _first_non_empty(
        *(_find_values_by_key(payload, "routeName", limit=2)),
        *(_find_values_by_key(payload, "listRouteName", limit=2)),
    )

    fallback = parse_route_identity(route_name, fallback_state="") if route_name else {"Hub": "", "Contractor": "", "Driver": "", "Route_type": ""}
    hub = normalize_hub_name(fallback.get("Hub", ""))
    contractor = fallback.get("Contractor", "")
    driver = fallback.get("Driver", "")
    route_type = "pickup" if (hub == "PU") else (fallback.get("Route_type", "delivery") or "delivery")

    return {
        "Route_name": route_name,
        "Route_names": route_name,
        "Driver": driver,
        "Drivers": driver,
        "Hub": hub,
        "Contractor": contractor,
        "Contractors": contractor,
        "Route_type": route_type,
    }


def extract_hub_from_scanned_at_payload(payload: Any) -> str:
    scanned_candidates = [
        *(_find_values_by_key(payload, "scanned_at", limit=5)),
        *(_find_values_by_key(payload, "scannedAt", limit=5)),
        *(_find_values_by_key(payload, "scannedAtHub", limit=5)),
        *(_find_values_by_key(payload, "scanAt", limit=5)),
    ]

    for candidate in scanned_candidates:
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        extracted_from_desc = extract_hub_from_scan_description(candidate_text)
        normalized_hub = normalize_hub_name(extracted_from_desc or candidate_text)
        if normalized_hub:
            return normalized_hub

    return ""


def empty_row(tracking_id: str) -> dict[str, str]:
    row = {col: "" for col in OUTPUT_COLUMNS}
    row["tracking_id"] = tracking_id
    return row

def normalize_state(state: str) -> str:
    normalized_state = str(state or "").strip().upper()
    if not normalized_state:
        return ""
    tokens = [token for token in re.split(r"[^A-Z]+", normalized_state) if token]
    for token in tokens:
        if token in HUB_BY_STATE:
            return token
        mapped_token = STATE_ALIAS.get(token)
        if mapped_token:
            return mapped_token

    for idx in range(len(tokens) - 1):
        two_token_name = f"{tokens[idx]} {tokens[idx + 1]}"
        mapped_two_token = STATE_ALIAS.get(two_token_name)
        if mapped_two_token:
            return mapped_two_token

    compact_state = re.sub(r"[^A-Z]", "", normalized_state)
    mapped_state = STATE_ALIAS.get(normalized_state, STATE_ALIAS.get(compact_state, ""))
    if mapped_state:
        return mapped_state

    if compact_state in HUB_BY_STATE:
        return compact_state

    # Handle values like `US-PA`, `Pennsylvania, US`, etc.
    if len(compact_state) > 2 and compact_state[-2:] in HUB_BY_STATE:
        return compact_state[-2:]

    return normalized_state


def infer_region_from_state(state: str) -> str:
    hub = infer_hub_from_state(state)
    return REGION_BY_HUB.get(hub, "")

def normalize_region(region: str) -> str:
    region_text = str(region or "").strip().upper()
    if not region_text:
        return ""

    compact_region = re.sub(r"[^A-Z]", "", region_text)
    if compact_region in {"WE", "WEST", "WESTERN"}:
        return "WE"
    if compact_region in {"EA", "EAST", "EASTERN"}:
        return "EA"
    return region_text



def fill_route_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "Route_type" not in df.columns:
        df["Route_type"] = ""

    for idx, row in df.iterrows():
        route_names_raw = row.get("Route_names") or ""
        route_name = str(row.get("Route_name") or "").strip()
        if (not route_name) and route_names_raw:
            try:
                loaded_routes = json.loads(route_names_raw) if isinstance(route_names_raw, str) else route_names_raw
            except Exception:
                loaded_routes = []
            if isinstance(loaded_routes, list) and loaded_routes:
                route_name = str(loaded_routes[0] or "").strip()
        fallback_state = str(row.get("State") or row.get("sender_province") or "")
        route_info = parse_route_identity(route_name, fallback_state=fallback_state)
        df.at[idx, "Route_name"] = route_name
        drivers_raw = row.get("Drivers") or ""
        contractors_raw = row.get("Contractors") or ""

        existing_driver = str(row.get("Driver") or "").strip()
        if (not existing_driver) and drivers_raw:
            try:
                loaded_drivers = json.loads(drivers_raw) if isinstance(drivers_raw, str) else drivers_raw
            except Exception:
                loaded_drivers = []
            if isinstance(loaded_drivers, list) and loaded_drivers:
                existing_driver = str(loaded_drivers[0] or "").strip()
        df.at[idx, "Driver"] = existing_driver or route_info["Driver"]

        existing_hub = str(row.get("Hub") or "").strip()
        df.at[idx, "Hub"] = existing_hub or route_info["Hub"]

        existing_contractor = str(row.get("Contractor") or "").strip()
        if (not existing_contractor) and contractors_raw:
            try:
                loaded_contractors = json.loads(contractors_raw) if isinstance(contractors_raw, str) else contractors_raw
            except Exception:
                loaded_contractors = []
            if isinstance(loaded_contractors, list) and loaded_contractors:
                existing_contractor = str(loaded_contractors[0] or "").strip()
        df.at[idx, "Contractor"] = existing_contractor or route_info["Contractor"]
        existing_route_type = str(row.get("Route_type") or "").strip()
        df.at[idx, "Route_type"] = existing_route_type or route_info["Route_type"]
    return df


def split_pickup_routes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    hub_series = df["Hub"].fillna("").astype(str).str.strip().str.upper()
    route_name_series = df.get("Route_name", "").fillna("").astype(str).str.upper()
    if "Route_type" in df.columns:
        route_type_series = df["Route_type"].fillna("").astype(str).str.strip().str.lower()
    else:
        route_type_series = pd.Series("", index=df.index)
    pickup_mask = (
        hub_series.eq("PU")
        | route_type_series.eq("pickup")
        | route_name_series.str.contains(r"\bPU\b", regex=True)
        | route_name_series.str.contains(r"\bPICK\s*UP\b", regex=True)
    )    

    pickup_df = df.loc[pickup_mask].copy()
    non_pickup_df = df.loc[~pickup_mask].copy()
    return non_pickup_df, pickup_df


def build_customer_address_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "sender_company",
        "sender_province",
        "sender_city",
        "sender_address",
        "out_for_delivery_time",
        "tracking_id",
    ]

    output_columns = [
        tr("customer_name"),
        tr("pickup_warehouse"),
        tr("state_group"),
        tr("package_count"),
    ]
    if df.empty or any(col not in df.columns for col in required_columns):
        return pd.DataFrame(columns=output_columns)

    work_df = df[required_columns].copy()
    for col in ["sender_company", "sender_province", "sender_city", "sender_address"]:
        work_df[col] = work_df[col].fillna("").astype(str).str.strip()

    work_df["shipping_address"] = (
        work_df["sender_province"]
        + " "
        + work_df["sender_city"]
        + " "
        + work_df["sender_address"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    work_df["_ofd_day"] = pd.to_datetime(work_df["out_for_delivery_time"], errors="coerce").dt.date
    work_df["_state_group"] = work_df["sender_province"].map(normalize_state)

    work_df = work_df[
        (work_df["sender_company"] != "")
        & (~work_df["sender_company"].str.casefold().eq("wyd china"))
        & (work_df["shipping_address"] != "")
        & work_df["_ofd_day"].notna()
    ]
    if work_df.empty:
        return pd.DataFrame(columns=output_columns)

    summary = (
        work_df.groupby(["_state_group", "sender_company", "shipping_address"], as_index=False)
        .agg(
            package_count=("tracking_id", "count"),
        )
        .sort_values(
            by=["_state_group", "sender_company", "package_count", "shipping_address"],
            ascending=[True, True, False, True],
        )
    )

    return summary.rename(
        columns={
            "sender_company": tr("customer_name"),
            "_state_group": tr("state_group"),
            "shipping_address": tr("pickup_warehouse"),
            "package_count": tr("package_count"),
        }
    )[output_columns]

def build_invalid_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    invalid_mask = (
        df["Route_name"].fillna("").astype(str).str.strip().ne("")
        & df["Contractor"].fillna("").astype(str).str.strip().eq("")
    )
    invalid_df = df.loc[invalid_mask, ["tracking_id", "Route_name"]].copy()
    if invalid_df.empty:
        return invalid_df

    invalid_df["Route_name"] = invalid_df["Route_name"].fillna("").astype(str).str.strip()
    invalid_df.loc[invalid_df["Route_name"] == "", "Route_name"] = "(empty)"
    grouped = (
        invalid_df.groupby("Route_name", dropna=False)
        .agg(
            tracking_count=("tracking_id", "count"),
            tracking_ids=("tracking_id", lambda s: ", ".join(s.astype(str).head(5))),
        )
        .reset_index()
        .sort_values(by=["tracking_count", "Route_name"], ascending=[False, True])
    )
    return grouped


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="export")
    return output.getvalue()



def build_lost_package_analysis(df: pd.DataFrame, fetch_reference_time: datetime | None = None) -> dict[str, Any]:
    df = df.copy()
    ofd_raw_column = "first_out_for_delivery_date" if "first_out_for_delivery_date" in df.columns else "out_for_delivery_time"
    datetime_fallback_columns = {
        "last_scanned_dt": "last_scanned_time",
        "ofd_dt": ofd_raw_column,
        "attempted_dt": "attempted_time",
        "delivered_dt": "delivered_time",
    }
    for dt_col, raw_col in datetime_fallback_columns.items():
        if dt_col in df.columns:
            continue
        if raw_col in df.columns:
            df[dt_col] = to_datetime_series(df, raw_col)
        else:
            df[dt_col] = pd.NaT

    def _load_interval_events(intervals_raw: Any) -> list[dict[str, Any]]:
        if isinstance(intervals_raw, list):
            return [item for item in intervals_raw if isinstance(item, dict)]
        if isinstance(intervals_raw, str):
            text = intervals_raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                return []
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        return []

    def _last_event_is_cancel(intervals_raw: Any) -> bool:
        events = _load_interval_events(intervals_raw)
        if not events:
            return False
        indexed = list(enumerate(events))
        indexed.sort(key=lambda item: ((event_ts(item[1]) is None), event_ts(item[1]) or -1, item[0]))
        last_event = indexed[-1][1]
        return event_type(last_event) == "cancel"

    base_mask = df["last_scanned_dt"].notna()
    scanned_base = df[base_mask].copy()
    if scanned_base.empty:
        return {
            "scanned_base": scanned_base,
            "lost_mask": pd.Series(False, index=df.index),
            "candidate_mask": pd.Series(False, index=df.index),
            "immature_mask": pd.Series(False, index=df.index),
        }

    # Warehouse-lost should only include packages that never reached downstream
    # delivery flow (OFD/attempted/delivered). Using "events after last scan"
    # can over-count because last_scanned_dt is often already the latest event.
    has_downstream_event = (
        scanned_base["ofd_dt"].notna()
        | scanned_base["attempted_dt"].notna()
        | scanned_base["delivered_dt"].notna()
    )
    candidate_mask_base = ~has_downstream_event

    if fetch_reference_time is None:
        fetch_reference_time_utc = datetime.now(timezone.utc)
    elif fetch_reference_time.tzinfo is None:
        fetch_reference_time_utc = fetch_reference_time.replace(tzinfo=timezone.utc)
    else:
        fetch_reference_time_utc = fetch_reference_time.astimezone(timezone.utc)

    last_scanned_utc = pd.to_datetime(scanned_base["last_scanned_dt"], errors="coerce", utc=True)
    last_scan_age_hours = (fetch_reference_time_utc - last_scanned_utc).dt.total_seconds() / 3600
    immature_mask_base = last_scan_age_hours < 120

    customer_service_mask_base = pd.Series(False, index=scanned_base.index)
    if "has_customer_service" in scanned_base.columns:
        customer_service_mask_base = (
            scanned_base["has_customer_service"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )

    canceled_last_mask_base = pd.Series(False, index=scanned_base.index)
    if "Intervals" in scanned_base.columns:
        canceled_last_mask_base = scanned_base["Intervals"].map(_last_event_is_cancel)

    lost_mask_base = candidate_mask_base & (~immature_mask_base) & (~customer_service_mask_base) & (~canceled_last_mask_base)

    candidate_mask = pd.Series(False, index=df.index)
    candidate_mask.loc[scanned_base.index] = candidate_mask_base.to_numpy()

    immature_mask = pd.Series(False, index=df.index)
    immature_mask.loc[scanned_base.index] = immature_mask_base.to_numpy()

    lost_mask = pd.Series(False, index=df.index)
    lost_mask.loc[scanned_base.index] = lost_mask_base.to_numpy()

    return {
        "scanned_base": scanned_base,
        "lost_mask": lost_mask,
        "candidate_mask": candidate_mask,
        "immature_mask": immature_mask,
    }

def is_unknown_dimension_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    return "未知" in text or "unknown" in text



def extract_route_numbers(series: pd.Series, limit: int = 5) -> str:
    if series.empty:
        return ""

    route_numbers: list[str] = []
    for value in series.fillna("").astype(str):
        route_name = value.strip()
        if not route_name:
            continue

        parts = [part.strip() for part in route_name.split("-") if part.strip()]
        if len(parts) >= 2:
            route_no = parts[1]
        else:
            match = re.search(r"([A-Za-z]*\d+[A-Za-z0-9]*)", route_name)
            route_no = match.group(1) if match else ""

        if route_no:
            route_numbers.append(route_no)

    if not route_numbers:
        return ""

    unique_route_numbers = list(dict.fromkeys(route_numbers))
    return ", ".join(unique_route_numbers[:limit])

def format_unknown_dimension_name(base_name: str, source_df: pd.DataFrame) -> str:
    if "未知" not in base_name or "Route_name" not in source_df.columns:
        return base_name

    route_preview = extract_route_numbers(source_df["Route_name"], limit=5)
    if not route_preview:
        return base_name
    return f"{base_name}（路线号：{route_preview}）"
