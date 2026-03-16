from collections import Counter
from datetime import datetime, timezone
from typing import Any
import os
import io 
import re 
import json

from utils.utils import *
from utils.constants import * 

# How many POD images to export per tracking_id (each image can have its own quality.feedback/score)
POD_IMAGE_EXPORT_N = int(os.getenv("POD_IMAGE_EXPORT_N", "5"))
POD_IMAGE_EXPORT_N = int(read_config("POD_IMAGE_EXPORT_N", str(POD_IMAGE_EXPORT_N)))
APP_VERSION = read_config("APP_VERSION", "a0.0.5")

POD_COLUMNS: list[str] = []
for i in range(1, POD_IMAGE_EXPORT_N + 1):
    POD_COLUMNS += [f"pod_feedback_{i}", f"pod_score_{i}"]

EXPORT_EXCLUDED_COLUMNS = set(POD_COLUMNS)


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

def latest_route_assignment(events: list[dict[str, Any]]) -> str:
    candidates: list[tuple[int, int, str, bool]] = []
    for idx, event in enumerate(events):
        route_name = parse_route(event_description(event))
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
        route_name = parse_route(event_description(event))
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

    pod_obj = container.get("pod")
    if isinstance(pod_obj, dict):
        images = pod_obj.get("images")
        if isinstance(images, list):
            all_images.extend([x for x in images if isinstance(x, dict)])
    elif isinstance(pod_obj, list):
        for pod_entry in pod_obj:
            if not isinstance(pod_entry, dict):
                continue
            images = pod_entry.get("images")
            if isinstance(images, list):
                all_images.extend([x for x in images if isinstance(x, dict)])

    pods_obj = container.get("pods")
    if isinstance(pods_obj, dict):
        pod_list = pods_obj.get("pod")
        if isinstance(pod_list, list):
            for pod_entry in pod_list:
                if not isinstance(pod_entry, dict):
                    continue
                images = pod_entry.get("images")
                if isinstance(images, list):
                    all_images.extend([x for x in images if isinstance(x, dict)])
        elif isinstance(pod_list, dict):
            images = pod_list.get("images")
            if isinstance(images, list):
                all_images.extend([x for x in images if isinstance(x, dict)])

    return all_images


def _event_has_pod_marker(event: dict[str, Any] | None, payload: dict[str, Any] | None = None) -> bool:
    if not isinstance(event, dict):
        return False

    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    candidate_nodes: list[Any] = [event, event.get("logItem"), event.get("log")]
    if isinstance(payload, dict):
        candidate_nodes.append(payload)

    for node in candidate_nodes:
        if not isinstance(node, dict):
            continue

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


def is_pod_compliant_for_event(event: dict[str, Any] | None, payload: dict[str, Any] | None = None) -> bool:
    if not event:
        return False

    pod_images = extract_pod_images_from_success_event(event)
    if len(pod_images) < 3 and isinstance(payload, dict):
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
        q = img.get("quality")
        if not isinstance(q, dict):
            continue
        score = str(q.get("score") or "").strip()
        if score:
            try:
                if float(score) != 0:
                    non_zero_scored_count += 1
            except (TypeError, ValueError):
                non_zero_scored_count += 1
    if pod_count >= 3 and non_zero_scored_count >= 1:
        return True

    # Fallback: some carriers do not return quality scores for every POD image,
    # but the event still includes valid POD evidence.
    if pod_count >= 1:
        return True

    return _event_has_pod_marker(event, payload=payload)


def build_intervals(events: list[dict[str, Any]], payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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

    for event in events:
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

        if evt_type in {"fail", "failed", "failure", "out-for-delivery", "ofd", "outfordelivery", "success", "delivered"}:
            route = parse_route(description)
            if route:
                node["route"] = route

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
    warehouse_events: list[dict[str, Any]] = []

    for evt in events:
        if event_type(evt) != "warehouse":
            continue

        evt_ts = event_ts(evt)
        if ofd_ts is not None and evt_ts is not None and evt_ts > ofd_ts:
            continue

        description = event_description(evt)
        if not extract_hub_name_from_warehouse_description(description):
            continue

        warehouse_events.append(evt)

    if not warehouse_events:
        return ""

    warehouse_events.sort(key=lambda e: ((event_ts(e) if event_ts(e) is not None else -1), events.index(e)))
    latest_event = warehouse_events[-1]
    return extract_hub_name_from_warehouse_description(event_description(latest_event))


def infer_hub_from_pre_ofd_scan(events: list[dict[str, Any]], ofd_evt: dict[str, Any] | None) -> str:
    ofd_ts = event_ts(ofd_evt) if ofd_evt else None
    scan_events: list[dict[str, Any]] = []

    for evt in events:
        description = event_description(evt)
        desc_lower = description.strip().lower()
        if not (desc_lower.startswith("scan at") or desc_lower.startswith("scanned at")):
            continue

        evt_ts = event_ts(evt)
        if ofd_ts is not None and evt_ts is not None and evt_ts > ofd_ts:
            continue

        scan_events.append(evt)

    if not scan_events:
        return ""

    scan_events.sort(key=lambda e: ((event_ts(e) if event_ts(e) is not None else -1), events.index(e)))
    target_evt = scan_events[-1]
    return extract_hub_from_scan_description(event_description(target_evt))


def build_row(tracking_id: str, payload: dict[str, Any]) -> dict[str, str]:
    events = normalize_events(payload)
    intervals = build_intervals(events, payload=payload)
    is_delivered = any(str(x.get("type") or "").strip().lower() in {"success", "delivered"} for x in intervals)

    scanned_predicate = lambda e: (
        event_type(e) in {"scan", "warehouse", "picked-up", "pickup"}
        or (
            (desc := str(e.get("description", "")).strip().lower()).startswith("scan at")
            or desc.startswith("scanned at")
            or "scan" in desc
        )
    )
    first_scanned_evt = first_event_by_predicate(events, scanned_predicate)

    ofd_events = events_by_predicate(events, lambda e: event_type(e) in {"out-for-delivery", "ofd", "outfordelivery"})
    fail_events = events_by_predicate(events, lambda e: event_type(e) in {"fail", "failed", "failure"})
    success_events = events_by_predicate(events, lambda e: event_type(e) in {"success", "delivered"})

    ofd_evt = ofd_events[0] if ofd_events else None
    warehouse_hub = infer_hub_from_pre_ofd_warehouse(events, ofd_evt)
    scan_hub = infer_hub_from_pre_ofd_scan(events, ofd_evt)
    fail_evt = fail_events[0] if fail_events else None
    success_evt = success_events[0] if success_events else None
    latest_route = latest_route_assignment(events)
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
    first_scanned_time = to_local_dt(event_ts(first_scanned_evt) if first_scanned_evt else None)
    out_for_delivery_time = to_local_dt(event_ts(ofd_evt) if ofd_evt else None)
    attempted_time = to_local_dt(event_ts(fail_evt) if fail_evt else None)
    delivered_time = to_local_dt(event_ts(success_evt) if success_evt else None)

    failed_route = parse_route(event_description(fail_evt)) if fail_evt else ""
    success_route = parse_route(event_description(success_evt)) if success_evt else ""
    ofd_route = parse_route(event_description(ofd_evt)) if ofd_evt else ""
    primary_route = choose_primary_route(
        ofd_route=ofd_route,
        failed_route=failed_route,
        success_route=success_route,
        fallback_route=latest_route,
    )

    structured_identity = extract_route_identity_from_payload(payload)
    route_name_value = str(structured_identity.get("Route_name") or primary_route or "").strip()

    raw_router_messages = ""
    if isinstance(payload, (dict, list)):
        raw_router_messages = json.dumps(payload, ensure_ascii=False)
    elif payload is not None:
        raw_router_messages = str(payload)

    row: dict[str, str] = {
        "tracking_id": tracking_id,
        "Driver": str(structured_identity.get("Driver") or "").strip(),
        "Hub": str(warehouse_hub or structured_identity.get("Hub") or scan_hub or "").strip(),
        "Contractor": str(structured_identity.get("Contractor") or "").strip(),
        "router_messages": raw_router_messages,
        "created_time": fmt_dt(created_time),
        "first_scanned_time": fmt_dt(first_scanned_time),
        "last_scanned_time": fmt_dt(to_local_dt(event_ts(events[-1]) if events else None)),
        "out_for_delivery_time": fmt_dt(out_for_delivery_time),
        "attempted_time": fmt_dt(attempted_time),
        "delivered_time": fmt_dt(delivered_time),
        "entered_costomer_service": "",
        "beans_pod_link": build_beans_tracking_link(tracking_id),
        "ofd_route": ofd_route,
        "failed_route": failed_route,
        "success_route": success_route,
        "Route_name": route_name_value,
        "Route_type": str(structured_identity.get("Route_type") or "delivery").strip(),
        "Intervals": json.dumps(intervals, ensure_ascii=False),
        "Is_delivered": "true" if is_delivered else "false",
    }

    for i in range(1, POD_IMAGE_EXPORT_N + 1):
        row[f"pod_feedback_{i}"] = ""
        row[f"pod_score_{i}"] = ""

    pod_images = extract_pod_images_from_success_event(success_evt)
    for i, img in enumerate(pod_images[:POD_IMAGE_EXPORT_N], start=1):
        q = img.get("quality")
        if not isinstance(q, dict):
            continue
        row[f"pod_feedback_{i}"] = str(q.get("feedback") or q.get("qualifiedFeedback") or "").strip()
        row[f"pod_score_{i}"] = str(q.get("score") or "").strip()

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


def extract_route_identity_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    route_name = _first_non_empty(
        *(_find_values_by_key(payload, "routeName", limit=2)),
        *(_find_values_by_key(payload, "listRouteName", limit=2)),
        *(_find_values_by_key(payload, "route", limit=2)),
    )
    assignee_name = _first_non_empty(
        *(_find_values_by_key(payload, "assigneeName", limit=2)),
        *(_find_values_by_key(payload, "driverName", limit=2)),
        *(_find_values_by_key(payload, "name", limit=2)),
    )
    warehouse_name = _first_non_empty(
        *(_find_values_by_key(payload, "warehouseName", limit=2)),
        *(_find_values_by_key(payload, "warehouse", limit=2)),
        *(_find_values_by_key(payload, "hub", limit=2)),
    )
    company_name = _first_non_empty(
        *(_find_values_by_key(payload, "companyName", limit=2)),
        *(_find_values_by_key(payload, "thirdPartyCompanyName", limit=2)),
        *(_find_values_by_key(payload, "contractor", limit=2)),
    )

    fallback = parse_route_identity(route_name, fallback_state="") if route_name else {"Hub": "", "Contractor": "", "Driver": "", "Route_type": ""}
    hub = normalize_hub_name(warehouse_name or fallback.get("Hub", ""))
    contractor = company_name or fallback.get("Contractor", "")
    driver = assignee_name or fallback.get("Driver", "")
    route_type = "pickup" if (hub == "PU") else (fallback.get("Route_type", "delivery") or "delivery")

    return {
        "Route_name": route_name,
        "Route_names": route_name,
        "Driver": driver,
        "Hub": hub,
        "Contractor": contractor,
        "Route_type": route_type,
    }


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
        route_name = str(
            row.get("Route_name")
            or row.get("success_route")
            or row.get("failed_route")
            or row.get("ofd_route")
            or ""
        ).strip()
        fallback_state = str(row.get("State") or row.get("sender_province") or "")
        route_info = parse_route_identity(route_name, fallback_state=fallback_state)
        df.at[idx, "Route_name"] = route_name
        existing_driver = str(row.get("Driver") or "").strip()
        df.at[idx, "Driver"] = existing_driver or route_info["Driver"]
        fallback_hub = normalize_hub_name(row.get("Hub") or "", fallback_state=fallback_state)
        parsed_hub = normalize_hub_name(route_info["Hub"], fallback_state=fallback_state)
        df.at[idx, "Hub"] = fallback_hub or parsed_hub
        existing_contractor = str(row.get("Contractor") or "").strip()
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
    base_mask = df["last_scanned_dt"].notna()
    scanned_base = df[base_mask].copy()
    if scanned_base.empty:
        return {
            "scanned_base": scanned_base,
            "lost_mask": pd.Series(False, index=df.index),
            "candidate_mask": pd.Series(False, index=df.index),
            "immature_mask": pd.Series(False, index=df.index),
        }

    time_window_end = scanned_base["last_scanned_dt"] + pd.Timedelta(hours=120)
    has_event_within_120h = (
        (scanned_base["ofd_dt"].notna() & (scanned_base["ofd_dt"] > scanned_base["last_scanned_dt"]) & (scanned_base["ofd_dt"] <= time_window_end))
        | (
            scanned_base["attempted_dt"].notna()
            & (scanned_base["attempted_dt"] > scanned_base["last_scanned_dt"])
            & (scanned_base["attempted_dt"] <= time_window_end)
        )
        | (
            scanned_base["delivered_dt"].notna()
            & (scanned_base["delivered_dt"] > scanned_base["last_scanned_dt"])
            & (scanned_base["delivered_dt"] <= time_window_end)
        )
    )

    candidate_mask_base = ~has_event_within_120h

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

    lost_mask_base = candidate_mask_base & (~immature_mask_base) & (~customer_service_mask_base)

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
