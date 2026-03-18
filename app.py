from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import ast
import json
import re

import utils.db as db
import utils.report as report_utils
import utils.routes as route_utils
from utils.utils import *
from utils.db import * 
from utils.routes import *
from utils.report import * 
from utils.constants import * 

import pandas as pd
import streamlit as st


def _load_intervals(intervals_raw: Any) -> list[dict[str, Any]]:
    if isinstance(intervals_raw, list):
        return [item for item in intervals_raw if isinstance(item, dict)]
    if isinstance(intervals_raw, str):
        text = intervals_raw.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            try:
                loaded = ast.literal_eval(text)
            except Exception:
                return []
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    return []




def _parse_list_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(x).strip() for x in loaded if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in text.split(",") if x.strip()]


def _pod_qualified(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "是"}


def ensure_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    def _first_list_item(value: Any) -> str:
        items = _parse_list_cell(value)
        return items[0] if items else ""

    if "Contractor" not in out.columns and "Contractors" in out.columns:
        out["Contractor"] = out["Contractors"].map(_first_list_item)
    if "Driver" not in out.columns and "Drivers" in out.columns:
        out["Driver"] = out["Drivers"].map(_first_list_item)
    if "Route_name" not in out.columns and "Route_names" in out.columns:
        out["Route_name"] = out["Route_names"].map(_first_list_item)

    def _time_from_intervals(intervals_raw: Any, target_types: set[str]) -> str:
        intervals = _load_intervals(intervals_raw)
        for evt in intervals:
            evt_type = str(evt.get("type") or "").strip().lower()
            if evt_type in target_types:
                return fmt_dt(to_local_dt(evt.get("time")))
        return ""

    if "out_for_delivery_time" not in out.columns:
        out["out_for_delivery_time"] = out.get("Intervals", "").map(
            lambda raw: _time_from_intervals(raw, {"out-for-delivery", "ofd", "outfordelivery"})
        )
    if "attempted_time" not in out.columns:
        out["attempted_time"] = out.get("Intervals", "").map(
            lambda raw: _time_from_intervals(raw, {"fail", "failed", "failure"})
        )

    return out

def build_route_attempts_view(
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    success_types = {"success"}
    fail_types = {"fail"}
    ofd_types = {"out-for-delivery"}

    route_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    canceled_rows: list[dict[str, Any]] = []
    lost_rows: list[dict[str, Any]] = []

    for _, row in source_df.iterrows():
        tracking_id = str(row.get("tracking_id") or "").strip()
        intervals = _load_intervals(row.get("Intervals"))
        if not intervals:
            unresolved_rows.append(
                {
                    "tracking_id": tracking_id,
                    "Hub": row.get("Hub", ""),
                    "created_time": row.get("created_time", ""),
                    "reason": "Intervals missing or invalid",
                }
            )
            continue

        last_event = intervals[-1] if intervals else {}
        last_event_type = route_utils.event_type(last_event)
        if last_event_type == "cancel":
            canceled_rows.append(
                {
                    "tracking_id": tracking_id,
                    "Hub": row.get("Hub", ""),
                    "created_time": row.get("created_time", ""),
                    "cancel_time": fmt_dt(to_local_dt(last_event.get("time"))),
                    "reason": "Last event type is cancel",
                }
            )
            continue

        found_ofd = False
        idx = 0
        while idx < len(intervals):
            event = intervals[idx]
            event_type_value = route_utils.event_type(event)
            if event_type_value not in ofd_types:
                idx += 1
                continue

            found_ofd = True

            current_ofd_event = event
            search_idx = idx + 1
            matched_terminal = None

            while search_idx < len(intervals):
                candidate_event = intervals[search_idx]
                candidate_type = route_utils.event_type(candidate_event)

                if candidate_type in ofd_types:
                    current_ofd_event = candidate_event
                    search_idx += 1
                    continue

                if candidate_type in (success_types | fail_types):
                    matched_terminal = candidate_event
                    break

                search_idx += 1

            if matched_terminal is None:
                lost_rows.append(
                    {
                        "tracking_id": tracking_id,
                        "Hub": row.get("Hub", ""),
                        "created_time": row.get("created_time", ""),
                        "out_for_delivery_time": fmt_dt(to_local_dt(current_ofd_event.get("time"))),
                        "reason": "No success/fail event found after out-for-delivery",
                    }
                )
                break

            matched_type = route_utils.event_type(matched_terminal)
            route_name = str(current_ofd_event.get("route") or matched_terminal.get("route") or "").strip()
            route_rows.append(
                {
                    "route": route_name or "UNKNOWN_ROUTE",
                    "result": "success" if matched_type in success_types else "fail",
                    "tracking_id": tracking_id,
                    "Region": row.get("Region", ""),
                    "Hub": row.get("Hub", ""),
                    "created_time": row.get("created_time", ""),
                    "out_for_delivery_time": fmt_dt(to_local_dt(current_ofd_event.get("time"))),
                    "finish_time": fmt_dt(to_local_dt(matched_terminal.get("time"))),
                    "POD是否合格": "是" if _pod_qualified(matched_terminal.get("POD")) else "否",
                }
            )
            idx = search_idx + 1

        if not found_ofd:
            unresolved_rows.append(
                {
                    "tracking_id": tracking_id,
                    "Hub": row.get("Hub", ""),
                    "created_time": row.get("created_time", ""),
                    "reason": "No out-for-delivery event found",
                }
            )

    route_df = pd.DataFrame(route_rows)
    unresolved_df = pd.DataFrame(unresolved_rows)
    canceled_df = pd.DataFrame(canceled_rows)
    lost_df = pd.DataFrame(lost_rows)

    if not route_df.empty:
        route_df = route_df.sort_values(by=["route", "out_for_delivery_time", "tracking_id"], na_position="last")
    if not unresolved_df.empty:
        unresolved_df = unresolved_df.sort_values(by=["reason", "tracking_id"], na_position="last")
    if not canceled_df.empty:
        canceled_df = canceled_df.sort_values(by=["tracking_id"], na_position="last")
    if not lost_df.empty:
        lost_df = lost_df.sort_values(by=["tracking_id"], na_position="last")
    return route_df, unresolved_df, canceled_df, lost_df


def build_multi_route_tracking_view(route_attempts_df: pd.DataFrame) -> pd.DataFrame:
    if route_attempts_df.empty:
        return pd.DataFrame()

    source_df = route_attempts_df.copy()
    source_df["route"] = source_df["route"].fillna("").astype(str).str.strip().replace("", "UNKNOWN_ROUTE")

    if "out_for_delivery_time" in source_df.columns:
        source_df["_ofd_dt"] = pd.to_datetime(source_df["out_for_delivery_time"], errors="coerce")
        source_df = source_df.sort_values(by=["tracking_id", "_ofd_dt"], na_position="last")

    rows: list[dict[str, Any]] = []
    for tracking_id, group in source_df.groupby("tracking_id", dropna=False):
        route_sequence = group["route"].tolist()
        unique_routes = list(dict.fromkeys(route_sequence))
        if len(unique_routes) <= 1:
            continue

        rows.append(
            {
                "tracking_id": str(tracking_id or "").strip(),
                "Hub": group["Hub"].iloc[0] if "Hub" in group.columns else "",
                "created_time": group["created_time"].iloc[0] if "created_time" in group.columns else "",
                "attempt_count": len(route_sequence),
                "unique_route_count": len(unique_routes),
                "route_sequence": " -> ".join(route_sequence),
                "unique_routes": ", ".join(unique_routes),
                "result_sequence": " -> ".join(group["result"].fillna("").astype(str).tolist()) if "result" in group.columns else "",
            }
        )

    multi_route_df = pd.DataFrame(rows)
    if multi_route_df.empty:
        return multi_route_df
    return multi_route_df.sort_values(by=["unique_route_count", "attempt_count", "tracking_id"], ascending=[False, False, True], na_position="last")


def build_route_attempt_metrics(route_attempts_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metric_names = ["24h妥投率", "48h妥投率", "72h妥投率", "24h尝试率"]
    if route_attempts_df.empty:
        return {name: {"hit": 0, "total": 0, "rate": 0.0} for name in metric_names}

    source_df = route_attempts_df.copy()
    source_df["result"] = source_df["result"].fillna("").astype(str).str.strip().str.lower()
    source_df["_ofd_dt"] = pd.to_datetime(source_df["out_for_delivery_time"], errors="coerce")
    source_df["_finish_dt"] = pd.to_datetime(source_df["finish_time"], errors="coerce")
    source_df["_elapsed_hours"] = (source_df["_finish_dt"] - source_df["_ofd_dt"]).dt.total_seconds() / 3600

    total_count = len(source_df)
    non_negative_duration_mask = source_df["_elapsed_hours"].notna() & (source_df["_elapsed_hours"] >= 0)

    def _metric_payload(mask: pd.Series) -> dict[str, Any]:
        hit_count = int(mask.sum())
        return {
            "hit": hit_count,
            "total": total_count,
            "rate": rate(hit_count, total_count),
        }

    success_mask = source_df["result"].eq("success") & non_negative_duration_mask
    attempt_mask = source_df["result"].isin(["success", "fail"]) & non_negative_duration_mask

    return {
        "24h妥投率": _metric_payload(success_mask & (source_df["_elapsed_hours"] < 24)),
        "48h妥投率": _metric_payload(success_mask & (source_df["_elapsed_hours"] < 48)),
        "72h妥投率": _metric_payload(success_mask & (source_df["_elapsed_hours"] < 72)),
        "24h尝试率": _metric_payload(attempt_mask & (source_df["_elapsed_hours"] < 24)),
    }


def _event_time_to_dt(event: dict[str, Any]) -> datetime | None:
    ts = route_utils.event_ts(event)
    if ts is None:
        raw_time = event.get("time")
        try:
            ts = int(raw_time) if raw_time is not None else None
        except (TypeError, ValueError):
            ts = None
    if ts is None:
        return None
    return to_local_dt(ts)


def build_dsp_hub_metrics(source_df: pd.DataFrame, route_attempts_df: pd.DataFrame) -> dict[str, Any]:
    base = {
        "dsp": {
            "pod_qualified_rate": {"hit": 0, "total": 0, "rate": 0.0},
            "lost_rate": {"hit": 0, "total": 0, "rate": 0.0},
        },
        "hub": {
            "scan_rates": {},
            "avg_first_track_to_sort_scan_hours": 0.0,
            "first_track_to_sort_scan_sample": 0,
            "intercept_success_rate": {"hit": 0, "total": 0, "rate": 0.0},
            "warehouse_lost_rate": {"hit": 0, "total": 0, "rate": 0.0},
        },
    }

    if route_attempts_df.empty and source_df.empty:
        return base

    if not route_attempts_df.empty:
        attempt_df = route_attempts_df.copy()
        pod_hit = int(attempt_df.get("POD是否合格", pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("是").sum())
        pod_total = len(attempt_df)
        base["dsp"]["pod_qualified_rate"] = {"hit": pod_hit, "total": pod_total, "rate": rate(pod_hit, pod_total)}

    total_tracking_rows = len(source_df)
    first_to_sort_scan_hours: list[float] = []
    canceled_count = 0
    intercept_success_count = 0
    warehouse_base_count = 0
    warehouse_lost_count = 0
    dsp_lost_count = 0

    def _is_warehouse_type(type_value: str) -> bool:
        normalized = str(type_value or "").strip().lower()
        return "warehouse" in normalized

    def _is_sorting_type(type_value: str) -> bool:
        normalized = str(type_value or "").strip().lower()
        return "sorting" in normalized or normalized == "sort"

    def _is_sort_scanned_event(event_item: dict[str, Any]) -> bool:
        if not _is_sorting_type(event_item["type"]):
            return False
        description = str(route_utils.event_description(event_item["event"]) or "").strip().lower()
        return "scanned at" in description

    def _find_first_network_event(
        events: list[dict[str, Any]],
        first_item: dict[str, Any],
    ) -> dict[str, Any] | None:
        first_time_ms = first_item["time_ms"]
        candidates = [
            item
            for item in events
            if item["time_ms"] >= first_time_ms
            and (_is_warehouse_type(item["type"]) or _is_sort_scanned_event(item))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item["time_ms"] - first_time_ms)

    def _interval_ts_ms(event: dict[str, Any]) -> int | None:
        ts = route_utils.event_ts(event)
        if ts is None:
            ts = event.get("time")
        if ts is None:
            return None
        try:
            return int(ts)
        except (TypeError, ValueError):
            return None

    for _, row in source_df.iterrows():
        intervals = _load_intervals(row.get("Intervals"))
        if not intervals:
            continue

        normalized_events = [
            {
                "event": evt,
                "type": route_utils.event_type(evt),
                "dt": _event_time_to_dt(evt),
                "time_ms": _interval_ts_ms(evt),
            }
            for evt in intervals
        ]
        normalized_events = [item for item in normalized_events if item["dt"] is not None and item["time_ms"] is not None]
        if not normalized_events:
            continue
        normalized_events.sort(key=lambda item: item["dt"])

        first_event_time = normalized_events[0]["dt"]
        cancel_index = next((idx for idx, item in enumerate(normalized_events) if item["type"] == "cancel"), None)
        is_canceled = cancel_index is not None

        for idx, item in enumerate(normalized_events):
            if item["type"] != "out-for-delivery":
                continue
            if idx == len(normalized_events) - 1:
                dsp_lost_count += 1

        if is_canceled:
            canceled_count += 1
            has_delivery_before_cancel = any(
                item["type"] in {"out-for-delivery", "success", "fail"} for item in normalized_events[: cancel_index + 1]
            )
            if not has_delivery_before_cancel:
                intercept_success_count += 1
        # 上网时效按 intervals 的“第一个节点 -> 最近的 warehouse 或最近的 sort(Scanned at)”计算，取耗时更短者。
        first_interval_item = normalized_events[0]
        first_network_item = _find_first_network_event(normalized_events, first_interval_item)
        if first_network_item is not None:
            elapsed_hours = (first_network_item["time_ms"] - first_interval_item["time_ms"]) / 3_600_000
            if elapsed_hours >= 0:
                first_to_sort_scan_hours.append(elapsed_hours)

        warehouse_or_sorting_indices = [
            idx
            for idx, item in enumerate(normalized_events)
            if _is_warehouse_type(item["type"]) or _is_sorting_type(item["type"])
        ]
        if warehouse_or_sorting_indices:
            warehouse_base_count += 1
            if warehouse_or_sorting_indices[-1] == len(normalized_events) - 1:
                warehouse_lost_count += 1

    attempt_total_for_lost = len(route_attempts_df) if not route_attempts_df.empty else 0
    base["dsp"]["lost_rate"] = {
        "hit": dsp_lost_count,
        "total": attempt_total_for_lost,
        "rate": rate(dsp_lost_count, attempt_total_for_lost),
    }

    for threshold in [12, 24, 48, 72]:
        hit = int(sum(hours < threshold for hours in first_to_sort_scan_hours))
        base["hub"]["scan_rates"][f"{threshold}h"] = {
            "hit": hit,
            "total": total_tracking_rows,
            "rate": rate(hit, total_tracking_rows),
        }

    avg_hours = float(sum(first_to_sort_scan_hours) / len(first_to_sort_scan_hours)) if first_to_sort_scan_hours else 0.0
    base["hub"]["avg_first_track_to_sort_scan_hours"] = avg_hours
    base["hub"]["first_track_to_sort_scan_sample"] = len(first_to_sort_scan_hours)
    base["hub"]["intercept_success_rate"] = {
        "hit": intercept_success_count,
        "total": canceled_count,
        "rate": rate(intercept_success_count, canceled_count),
    }
    base["hub"]["warehouse_lost_rate"] = {
        "hit": warehouse_lost_count,
        "total": warehouse_base_count,
        "rate": rate(warehouse_lost_count, warehouse_base_count),
    }
    return base


def _filter_df_by_datetime_window(
    source_df: pd.DataFrame,
    datetime_column: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> pd.DataFrame:
    if source_df.empty or datetime_column not in source_df.columns:
        return source_df
    if start_dt is None and end_dt is None:
        return source_df

    dt_series = pd.to_datetime(source_df[datetime_column], errors="coerce")
    mask = pd.Series(True, index=source_df.index)
    if start_dt is not None:
        mask &= dt_series >= start_dt
    if end_dt is not None:
        mask &= dt_series <= end_dt
    return source_df.loc[mask].copy()


def build_hub_scan_detail_table(source_df: pd.DataFrame, thresholds: list[int] | None = None) -> pd.DataFrame:
    thresholds = thresholds or [12, 24, 48, 72]
    if source_df.empty:
        return pd.DataFrame(
            columns=[
                "tracking_id",
                "Region",
                "State",
                "Hub",
                "created_time",
                "是否计入分母",
                "是否有有效intervals",
                "首节点时间",
                "首个上网节点时间",
                "首节点到上网节点时长(h)",
            ]
            + [f"<{threshold}h是否命中分子" for threshold in thresholds]
        )

    detail_df = source_df.copy()
    detail_df["tracking_id"] = detail_df.get("tracking_id", "").fillna("").astype(str).str.strip()
    detail_df["Region"] = detail_df.get("Region", "")
    detail_df["State"] = detail_df.get("State", "")
    detail_df["Hub"] = detail_df.get("Hub", "")
    detail_df["created_time"] = detail_df.get("created_time", "")

    first_event_time_values: list[str] = []
    first_sort_scan_time_values: list[str] = []
    elapsed_hours_values: list[float | None] = []
    has_valid_intervals_values: list[bool] = []

    def _is_sorting_type(type_value: str) -> bool:
        normalized = str(type_value or "").strip().lower()
        return "sorting" in normalized or normalized == "sort"

    def _is_warehouse_type(type_value: str) -> bool:
        normalized = str(type_value or "").strip().lower()
        return "warehouse" in normalized

    def _is_sort_scanned_event(event_item: dict[str, Any]) -> bool:
        if not _is_sorting_type(event_item["type"]):
            return False
        description = str(route_utils.event_description(event_item["event"]) or "").strip().lower()
        return "scanned at" in description

    def _find_first_network_event(
        events: list[dict[str, Any]],
        first_item: dict[str, Any],
    ) -> dict[str, Any] | None:
        first_time_ms = first_item["time_ms"]
        candidates = [
            item
            for item in events
            if item["time_ms"] >= first_time_ms
            and (_is_warehouse_type(item["type"]) or _is_sort_scanned_event(item))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item["time_ms"] - first_time_ms)

    def _interval_ts_ms(event: dict[str, Any]) -> int | None:
        ts = route_utils.event_ts(event)
        if ts is None:
            ts = event.get("time")
        if ts is None:
            return None
        try:
            return int(ts)
        except (TypeError, ValueError):
            return None

    for _, row in detail_df.iterrows():
        intervals = _load_intervals(row.get("Intervals"))
        if not intervals:
            has_valid_intervals_values.append(False)
            first_event_time_values.append("")
            first_sort_scan_time_values.append("")
            elapsed_hours_values.append(None)
            continue

        normalized_events = [
            {
                "event": evt,
                "type": route_utils.event_type(evt),
                "time_ms": _interval_ts_ms(evt),
                "dt": _event_time_to_dt(evt),
            }
            for evt in intervals
        ]
        normalized_events = [item for item in normalized_events if item["dt"] is not None and item["time_ms"] is not None]
        if not normalized_events:
            has_valid_intervals_values.append(False)
            first_event_time_values.append("")
            first_sort_scan_time_values.append("")
            elapsed_hours_values.append(None)
            continue

        normalized_events.sort(key=lambda item: item["dt"])
        has_valid_intervals_values.append(True)
        first_item = normalized_events[0]
        first_network_item = _find_first_network_event(normalized_events, first_item)

        first_event_time_values.append(fmt_dt(first_item["dt"]))
        if first_network_item is None:
            first_sort_scan_time_values.append("")
            elapsed_hours_values.append(None)
            continue

        first_sort_scan_time_values.append(fmt_dt(first_network_item["dt"]))
        elapsed_hours = (first_network_item["time_ms"] - first_item["time_ms"]) / 3_600_000
        elapsed_hours_values.append(elapsed_hours if elapsed_hours >= 0 else None)

    detail_df["是否计入分母"] = "是"
    detail_df["是否有有效intervals"] = ["是" if x else "否" for x in has_valid_intervals_values]
    detail_df["首节点时间"] = first_event_time_values
    detail_df["首个上网节点时间"] = first_sort_scan_time_values
    detail_df["首节点到上网节点时长(h)"] = elapsed_hours_values

    for threshold in thresholds:
        detail_df[f"<{threshold}h是否命中分子"] = detail_df["首节点到上网节点时长(h)"].apply(
            lambda value: "是" if pd.notna(value) and value < threshold else "否"
        )

    display_columns = [
        "tracking_id",
        "Region",
        "State",
        "Hub",
        "created_time",
        "是否计入分母",
        "是否有有效intervals",
        "首节点时间",
        "首个上网节点时间",
        "首节点到上网节点时长(h)",
    ] + [f"<{threshold}h是否命中分子" for threshold in thresholds]

    return detail_df.loc[:, display_columns]


def build_tracking_display_df(
    source_df: pd.DataFrame,
    route_attempts_df: pd.DataFrame,
    unresolved_attempts_df: pd.DataFrame,
    canceled_attempts_df: pd.DataFrame,
    lost_attempts_df: pd.DataFrame,
) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()

    display_df = source_df.copy()
    display_df["tracking_id"] = display_df.get("tracking_id", "").fillna("").astype(str).str.strip()

    route_latest = pd.DataFrame(columns=["tracking_id", "attempt_result", "last_route", "last_out_for_delivery_time", "last_finish_time"])
    if not route_attempts_df.empty:
        route_sorted = route_attempts_df.copy()
        route_sorted["_finish_dt"] = pd.to_datetime(route_sorted.get("finish_time"), errors="coerce")
        route_sorted = route_sorted.sort_values(by=["tracking_id", "_finish_dt"], na_position="last")
        route_latest = (
            route_sorted.groupby("tracking_id", as_index=False)
            .tail(1)
            .rename(
                columns={
                    "result": "attempt_result",
                    "route": "last_route",
                    "out_for_delivery_time": "last_out_for_delivery_time",
                    "finish_time": "last_finish_time",
                }
            )[["tracking_id", "attempt_result", "last_route", "last_out_for_delivery_time", "last_finish_time"]]
        )

    unresolved_ids = set(unresolved_attempts_df.get("tracking_id", pd.Series(dtype=str)).dropna().astype(str).str.strip().tolist())
    canceled_ids = set(canceled_attempts_df.get("tracking_id", pd.Series(dtype=str)).dropna().astype(str).str.strip().tolist())
    lost_ids = set(lost_attempts_df.get("tracking_id", pd.Series(dtype=str)).dropna().astype(str).str.strip().tolist())

    display_df = display_df.merge(route_latest, on="tracking_id", how="left")
    display_df["display_status"] = "processing"
    display_df.loc[display_df["tracking_id"].isin(canceled_ids), "display_status"] = "canceled"
    display_df.loc[display_df["tracking_id"].isin(lost_ids), "display_status"] = "lost_after_ofd"
    display_df.loc[display_df["tracking_id"].isin(unresolved_ids), "display_status"] = "unresolved"
    display_df.loc[display_df["attempt_result"].eq("success"), "display_status"] = "delivered"
    display_df.loc[display_df["attempt_result"].eq("fail"), "display_status"] = "failed_attempt"

    preferred_columns = [
        "tracking_id",
        "display_status",
        "attempt_result",
        "last_route",
        "last_out_for_delivery_time",
        "last_finish_time",
        "delivered_time",
        "Region",
        "State",
        "Hub",
        "Contractor",
        "Driver",
        "created_time",
    ]
    existing_columns = [col for col in preferred_columns if col in display_df.columns]
    return display_df.loc[:, existing_columns].sort_values(by=["display_status", "tracking_id"], na_position="last")


def build_timeliness_quality_breakdown_table(route_attempts_df: pd.DataFrame, thresholds: list[int] | None = None) -> pd.DataFrame:
    thresholds = thresholds or [24, 48, 72]
    if route_attempts_df.empty:
        return pd.DataFrame(columns=["Dimension", "Sample Count"])

    source_df = route_attempts_df.copy()
    source_df["result"] = source_df["result"].fillna("").astype(str).str.strip().str.lower()
    source_df["_ofd_dt"] = pd.to_datetime(source_df["out_for_delivery_time"], errors="coerce")
    source_df["_finish_dt"] = pd.to_datetime(source_df["finish_time"], errors="coerce")
    source_df["_elapsed_hours"] = (source_df["_finish_dt"] - source_df["_ofd_dt"]).dt.total_seconds() / 3600
    source_df["_is_success"] = source_df["result"].eq("success")
    source_df["_valid_duration"] = source_df["_elapsed_hours"].notna() & (source_df["_elapsed_hours"] >= 0)
    source_df["_region_norm"] = source_df["Region"].apply(normalize_region)
    source_df["_hub_norm"] = source_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")

    def _row_payload(scope_name: str, scope_df: pd.DataFrame) -> dict[str, Any]:
        total_count = len(scope_df)
        row = {"Dimension": scope_name, "Sample Count": total_count}
        for threshold in thresholds:
            hit_count = int((scope_df["_is_success"] & scope_df["_valid_duration"] & (scope_df["_elapsed_hours"] < threshold)).sum())
            row[f"<{threshold}h Hit"] = hit_count
            row[f"<{threshold}h Delivery Rate"] = rate(hit_count, total_count)
        return row

    rows: list[dict[str, Any]] = [_row_payload("Overall", source_df)]
    for region_code in ["EA", "WE"]:
        region_df = source_df[source_df["_region_norm"] == region_code]
        rows.append(_row_payload(region_code, region_df))
        if region_df.empty:
            continue

        for hub_name in sorted(region_df["_hub_norm"].unique()):
            hub_df = region_df[region_df["_hub_norm"] == hub_name]
            rows.append(_row_payload(hub_name, hub_df))

    table_df = pd.DataFrame(rows)
    for threshold in thresholds:
        percent_col = f"<{threshold}h Delivery Rate"
        if percent_col in table_df.columns:
            table_df[percent_col] = table_df[percent_col].map(lambda x: f"{x:.2%}")
    return table_df


def render_percentage_pie(
    title: str,
    hit_count: int,
    total_count: int,
    hit_label: str = "Hit",
    miss_label: str = "Miss",
    chart_key: str | None = None,
    container: Any | None = None,
) -> None:
    target = container or st
    if total_count <= 0:
        target.info(f"{title}: no data available")
        return

    miss_count = max(total_count - hit_count, 0)
    chart_df = pd.DataFrame({"Category": [hit_label, miss_label], "Count": [hit_count, miss_count]})
    chart_df = chart_df[chart_df["Count"] > 0]
    chart_df["Rate"] = (chart_df["Count"] / total_count).map(lambda x: f"{x:.2%}")

    target.caption(title)
    target.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "arc", "outerRadius": 100},
            "encoding": {
                "theta": {"field": "Count", "type": "quantitative"},
                "color": {"field": "Category", "type": "nominal"},
                "tooltip": [
                    {"field": "Category", "type": "nominal"},
                    {"field": "Count", "type": "quantitative"},
                    {"field": "Rate", "type": "nominal"},
                ],
            },
        },
        use_container_width=True,
        key=chart_key or f"pie_{title}_{hit_count}_{total_count}_{hit_label}_{miss_label}",
    )

def _append_delivery_breakdown_rows(
    rows: list[dict[str, Any]],
    scope_name: str,
    source_df: pd.DataFrame,
    thresholds: list[int],
) -> None:
    total_count = len(source_df)
    row: dict[str, Any] = {"Dimension": scope_name, "Sample Count": total_count}
    for threshold in thresholds:
        hit_col = f"within_{threshold}h"
        hit_count = int(source_df[hit_col].sum()) if hit_col in source_df.columns else 0
        row[f"<{threshold}h Hit"] = hit_count
        row[f"<{threshold}h Delivery Rate"] = rate(hit_count, total_count)

    sample_tracking_ids = (
        source_df["tracking_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().head(8).tolist()
        if "tracking_id" in source_df.columns
        else []
    )
    row["Sample Tracking IDs"] = ", ".join(sample_tracking_ids)
    rows.append(row)

def style_breakdown_rows(table_df: pd.DataFrame) -> Any:
    if table_df.empty or "Dimension" not in table_df.columns:
        return table_df

    level_colors = {
        0: "#f3f4f6",
        1: "#dbeafe",
        2: "#e0f2fe",
        3: "#ecfccb",
    }

    def _style_row(row: pd.Series) -> list[str]:
        dimension_name = str(row.get("Dimension", ""))
        leading_spaces = len(dimension_name) - len(dimension_name.lstrip(" "))
        indent_level = min(3, leading_spaces // 2)
        color = level_colors.get(indent_level, "#ffffff")
        return [f"background-color: {color}"] * len(row)

    return table_df.style.apply(_style_row, axis=1)


def build_delivery_breakdown_table(delivered_detail_df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    if delivered_detail_df.empty:
        return pd.DataFrame(columns=["Dimension", "Sample Count"])

    source_df = delivered_detail_df.copy()
    source_df["region_norm"] = source_df["Region"].apply(normalize_region)

    rows: list[dict[str, Any]] = []
    _append_delivery_breakdown_rows(rows, "Overall", source_df, thresholds)

    for region_code, region_name in [("WE", "West"), ("EA", "East")]:
        region_df = source_df[source_df["region_norm"] == region_code]
        _append_delivery_breakdown_rows(rows, region_name, region_df, thresholds)
        if region_df.empty:
            continue

        for hub_name in sorted(region_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub").unique()):
            hub_df = region_df[region_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub") == hub_name]
            hub_display_name = format_unknown_dimension_name(hub_name, hub_df)
            _append_delivery_breakdown_rows(rows, f"  {hub_display_name}", hub_df, thresholds)

            contractor_series = hub_df["Contractor"].fillna("Unknown Contractor").astype(str).str.strip().replace("", "Unknown Contractor")
            for contractor_name in sorted(contractor_series.unique()):
                contractor_df = hub_df[contractor_series == contractor_name]
                contractor_display_name = format_unknown_dimension_name(contractor_name, contractor_df)
                _append_delivery_breakdown_rows(rows, f"    {contractor_display_name}", contractor_df, thresholds)

    table_df = pd.DataFrame(rows)
    percent_cols = [f"<{threshold}h Delivery Rate" for threshold in thresholds]
    for col in percent_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.2%}")
    return table_df


def apply_manual_dimension_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    contractor_overrides = st.session_state.get("unknown_contractor_overrides", {})
    if not contractor_overrides:
        return df

    updated_df = df.copy()
    if "Hub" not in updated_df.columns or "Contractor" not in updated_df.columns:
        return updated_df

    normalized_hub_series = updated_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
    contractor_unknown_mask = updated_df["Contractor"].map(is_unknown_dimension_value)

    for hub_name, contractor_name in contractor_overrides.items():
        hub_value = str(hub_name or "").strip()
        contractor_value = str(contractor_name or "").strip()
        if not hub_value or not contractor_value:
            continue

        target_hub_mask = normalized_hub_series == hub_value
        updated_df.loc[target_hub_mask & contractor_unknown_mask, "Contractor"] = contractor_value

    return updated_df

def render_compact_kpi_row(kpi_payload: dict[str, Any]) -> None:
    metric_map = _metric_lookup(kpi_payload)
    st.markdown(f"#### {tr('compact_title')}")

    metric_specs = [
        ("<24h delivery rate", "24h妥投率"),
        ("<48h delivery rate", "48h妥投率"),
        ("<72h delivery rate", "72h妥投率"),
        ("24h attempt rate", "24h Attempt率"),
    ]
    cols = st.columns(len(metric_specs))
    for idx, (metric_key, label) in enumerate(metric_specs):
        metric = metric_map.get(metric_key)
        if not metric:
            cols[idx].metric(label, "0.00%", "0/0")
        else:
            cols[idx].metric(label, f"{metric['rate']:.2%}", f"{metric['hit']}/{metric['total']}")


def _metric_lookup(kpi_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = kpi_payload.get("metrics", []) if isinstance(kpi_payload, dict) else []
    return {
        str(item.get("metric")): item
        for item in metrics
        if isinstance(item, dict) and str(item.get("metric", "")).strip()
    }


def _upsert_kpi_metric_and_chart(
    kpi_payload: dict[str, Any],
    *,
    category: str,
    metric_name: str,
    hit_count: int,
    total_count: int,
    hit_label: str,
    miss_label: str,
) -> None:
    metrics = kpi_payload.setdefault("metrics", [])
    charts = kpi_payload.setdefault("charts", [])

    metrics = [m for m in metrics if not (isinstance(m, dict) and m.get("metric") == metric_name)]
    charts = [c for c in charts if not (isinstance(c, dict) and c.get("chart") == metric_name)]

    metric_rate = rate(hit_count, total_count)
    metrics.append(
        {
            "category": category,
            "metric": metric_name,
            "hit": int(hit_count),
            "total": int(total_count),
            "rate": metric_rate,
        }
    )

    miss_count = max(int(total_count) - int(hit_count), 0)
    charts.extend(
        [
            {"chart": metric_name, "category": hit_label, "count": int(hit_count), "rate": metric_rate},
            {
                "chart": metric_name,
                "category": miss_label,
                "count": miss_count,
                "rate": rate(miss_count, total_count),
            },
        ]
    )

    kpi_payload["metrics"] = metrics
    kpi_payload["charts"] = charts

def render_daily_kpi_charts(result_df: pd.DataFrame) -> None:
    chart_df = result_df.copy()
    chart_df["_created_date"] = to_datetime_series(chart_df, "created_time").dt.date
    chart_df["_weight"] = pd.to_numeric(chart_df.get("Weight", ""), errors="coerce")
    chart_df["_evaluation_weight"] = chart_df["_weight"]
    if chart_df["_evaluation_weight"].notna().sum() == 0:
        chart_df["_evaluation_weight"] = calculate_package_evaluation_weight(chart_df)

    created_count_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")
        .size()
        .rename("Package Count")
        .reset_index()
        .sort_values("_created_date")
    )
    evaluation_weight_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")["_evaluation_weight"]
        .mean()
        .rename("Evaluation Weight")
        .reset_index()
        .sort_values("_created_date")
    )

    normalized_weight = report_utils._normalize_weight_to_unit(chart_df["_weight"])
    valid_weight_df = chart_df[normalized_weight.notna() & (normalized_weight > 0)].copy()
    if not valid_weight_df.empty:
        weight_labels = report_utils._weight_bucket_labels(valid_weight_df)
        valid_weight_df["_weight_bucket"] = normalized_weight.loc[valid_weight_df.index].astype(int).astype(str)
        weight_bucket_df = (
            valid_weight_df.groupby("_weight_bucket", observed=False)
            .size()
            .reindex(weight_labels, fill_value=0)
            .rename("Package Count")
            .reset_index()
            .rename(columns={"_weight_bucket": "Weight Bucket"})
        )
        weight_bucket_df = weight_bucket_df[weight_bucket_df["Package Count"] > 0]
    else:
        weight_bucket_df = pd.DataFrame(columns=["Weight Bucket", "Package Count"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {tr('daily_created_chart')}")
        if created_count_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.line_chart(created_count_df.set_index("_created_date")["Package Count"])

    with c2:
        st.markdown(f"#### {tr('weight_bucket_chart')}")
        if weight_bucket_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.bar_chart(weight_bucket_df.set_index("Weight Bucket")["Package Count"])

    st.markdown(f"#### {tr('daily_eval_weight_chart')}")
    if evaluation_weight_df.empty:
        st.info(tr("eval_weight_empty"))
    else:
        st.line_chart(evaluation_weight_df.set_index("_created_date")["Evaluation Weight"])

def render_kpi_charts(
    result_df: pd.DataFrame,
    layout_mode: str,
    fetch_reference_time: datetime | None = None,
    route_attempts_df: pd.DataFrame | None = None,
    show_details: bool = False,
    report_start_dt: datetime | None = None,
    report_end_dt: datetime | None = None,
    exclude_atl_wdr: bool = True,
) -> dict[str, Any]:
    st.subheader(tr("kpi_title"))
    if result_df.empty:
        st.info(tr("kpi_empty"))
        return {"metrics": [], "charts": [], "has_monthly_lost_data": False, "monthly_lost": pd.DataFrame()}

    kpi_payload = build_kpi_report_payload(
        result_df,
        fetch_reference_time=fetch_reference_time,
    )
    refresh_key = str(int(fetch_reference_time.timestamp())) if fetch_reference_time else "no_fetch_ts"
    metric_map = _metric_lookup(kpi_payload)
    metric_source_df = route_attempts_df if route_attempts_df is not None else pd.DataFrame()

    excluded_hubs = {"ATL", "WDR"}
    metric_scope_df = metric_source_df.copy()
    hub_scope_df = result_df.copy()

    if exclude_atl_wdr:
        if not metric_scope_df.empty and "Hub" in metric_scope_df.columns:
            metric_scope_df = metric_scope_df[
                ~metric_scope_df["Hub"].fillna("").astype(str).str.strip().str.upper().isin(excluded_hubs)
            ].copy()
        if not hub_scope_df.empty and "Hub" in hub_scope_df.columns:
            hub_scope_df = hub_scope_df[
                ~hub_scope_df["Hub"].fillna("").astype(str).str.strip().str.upper().isin(excluded_hubs)
            ].copy()

    dsp_metric_scope_df = _filter_df_by_datetime_window(
        metric_scope_df,
        "out_for_delivery_time",
        start_dt=report_start_dt,
        end_dt=report_end_dt,
    )
    hub_metric_scope_df = _filter_df_by_datetime_window(
        hub_scope_df,
        "created_time",
        start_dt=report_start_dt,
        end_dt=report_end_dt,
    )

    route_attempt_metrics = build_route_attempt_metrics(dsp_metric_scope_df)

    st.markdown("#### OFD派送时效看板（按每次派送计算）")
    st.caption("派送尝试时效指标已迁移至下方“DSP相关指标”。")

    dsp_hub_metrics = build_dsp_hub_metrics(hub_metric_scope_df, dsp_metric_scope_df)

    _upsert_kpi_metric_and_chart(
        kpi_payload,
        category="dsp_assessment",
        metric_name="POD qualified rate",
        hit_count=int(dsp_hub_metrics["dsp"]["pod_qualified_rate"]["hit"]),
        total_count=int(dsp_hub_metrics["dsp"]["pod_qualified_rate"]["total"]),
        hit_label="Qualified",
        miss_label="Not qualified",
    )
    dsp_lost_hit = int(dsp_hub_metrics["dsp"]["lost_rate"]["hit"])
    dsp_lost_total = int(dsp_hub_metrics["dsp"]["lost_rate"]["total"])
    _upsert_kpi_metric_and_chart(
        kpi_payload,
        category="dsp_assessment",
        metric_name="DSP lost rate",
        hit_count=max(dsp_lost_total - dsp_lost_hit, 0),
        total_count=dsp_lost_total,
        hit_label="Not lost",
        miss_label="Lost after OFD",
    )
    warehouse_lost_hit = int(dsp_hub_metrics["hub"]["warehouse_lost_rate"]["hit"])
    warehouse_lost_total = int(dsp_hub_metrics["hub"]["warehouse_lost_rate"]["total"])
    _upsert_kpi_metric_and_chart(
        kpi_payload,
        category="hub_assessment",
        metric_name="Warehouse lost rate",
        hit_count=max(warehouse_lost_total - warehouse_lost_hit, 0),
        total_count=warehouse_lost_total,
        hit_label="Not lost",
        miss_label="Warehouse/sorting lost",
    )
    _upsert_kpi_metric_and_chart(
        kpi_payload,
        category="hub_assessment",
        metric_name="Intercept success rate",
        hit_count=int(dsp_hub_metrics["hub"]["intercept_success_rate"]["hit"]),
        total_count=int(dsp_hub_metrics["hub"]["intercept_success_rate"]["total"]),
        hit_label="Intercepted successfully",
        miss_label="Intercept failed",
    )

    st.markdown("#### DSP相关指标")
    dsp_cols = st.columns(2)
    pod_metric = dsp_hub_metrics["dsp"]["pod_qualified_rate"]
    dsp_lost_metric = dsp_hub_metrics["dsp"]["lost_rate"]
    dsp_not_lost_hit = max(int(dsp_lost_metric["total"]) - int(dsp_lost_metric["hit"]), 0)
    dsp_not_lost_rate = rate(dsp_not_lost_hit, int(dsp_lost_metric["total"]))
    dsp_cols[0].metric("POD合格率", f"{pod_metric['rate']:.2%}", f"{pod_metric['hit']}/{pod_metric['total']}")
    dsp_cols[1].metric("DSP非丢件率（1-丢件率）", f"{dsp_not_lost_rate:.2%}", f"{dsp_not_lost_hit}/{dsp_lost_metric['total']}")

    metric_specs = ["24h妥投率", "48h妥投率", "72h妥投率", "24h尝试率"]
    metric_cols = st.columns(len(metric_specs))
    for i, label in enumerate(metric_specs):
        metric = route_attempt_metrics.get(label, {"rate": 0.0, "hit": 0, "total": 0})
        metric_cols[i].metric(label, f"{metric['rate']:.2%}", f"{metric['hit']}/{metric['total']}")

    pie_cols = st.columns(len(metric_specs))
    for i, label in enumerate(metric_specs):
        metric = route_attempt_metrics.get(label, {"rate": 0.0, "hit": 0, "total": 0})
        if label == "24h尝试率":
            hit_label = "24h内完成尝试"
            miss_label = "超24h或缺失终态"
        else:
            hit_label = f"{label}达标"
            miss_label = f"{label}未达标"
        render_percentage_pie(
            title=f"{label} 构成",
            hit_count=int(metric["hit"]),
            total_count=int(metric["total"]),
            hit_label=hit_label,
            miss_label=miss_label,
            chart_key=f"attempt_{i}_{refresh_key}",
            container=pie_cols[i],
        )

    st.caption("派送尝试时效口径：基于“按派送尝试整理的Route明细”表计算，分母=该表全部条目。")

    st.markdown("#### Hub相关指标")
    hub_scan_specs = ["12h", "24h", "48h", "72h"]
    hub_scan_cols = st.columns(len(hub_scan_specs))
    for idx, key in enumerate(hub_scan_specs):
        metric = dsp_hub_metrics["hub"]["scan_rates"].get(key, {"rate": 0.0, "hit": 0, "total": 0})
        hub_scan_cols[idx].metric(f"<{key}上网率", f"{metric['rate']:.2%}", f"{metric['hit']}/{metric['total']}")

    hub_cols = st.columns(3)
    avg_hours = dsp_hub_metrics["hub"]["avg_first_track_to_sort_scan_hours"]
    avg_sample = dsp_hub_metrics["hub"]["first_track_to_sort_scan_sample"]
    intercept_metric = dsp_hub_metrics["hub"]["intercept_success_rate"]
    warehouse_lost_metric = dsp_hub_metrics["hub"]["warehouse_lost_rate"]
    warehouse_not_lost_hit = max(int(warehouse_lost_metric["total"]) - int(warehouse_lost_metric["hit"]), 0)
    warehouse_not_lost_rate = rate(warehouse_not_lost_hit, int(warehouse_lost_metric["total"]))
    hub_cols[0].metric("首轨迹到首个上网节点平均时长（warehouse或Sort Scanned at）", f"{avg_hours:.2f}h", f"样本 {avg_sample}")
    hub_cols[1].metric("拦截成功率", f"{intercept_metric['rate']:.2%}", f"{intercept_metric['hit']}/{intercept_metric['total']}")
    hub_cols[2].metric("仓库非丢件率（1-丢件率）", f"{warehouse_not_lost_rate:.2%}", f"{warehouse_not_lost_hit}/{warehouse_lost_metric['total']}")

    st.caption("Hub口径：上网率按单号统计（分母=全部单号，分子=intervals首节点到最近的warehouse或最近的type=sort且description含Scanned at节点，取耗时更短者并在阈值内）；拦截成功率分母=取消件；仓库丢件率分母=出现过warehouse/sorting的包裹。")

    st.markdown("#### Hub上网率明细（分母/分子）")
    hub_scan_detail_df = build_hub_scan_detail_table(hub_metric_scope_df, thresholds=[12, 24, 48, 72])
    hub_scan_header_cols = st.columns([4, 1])
    hub_scan_header_cols[1].download_button(
        "下载Hub上网率明细",
        data=hub_scan_detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"hub_scan_rate_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=hub_scan_detail_df.empty,
    )
    _render_optional_dataframe(
        hub_scan_detail_df,
        show_details,
        "暂无 Hub 上网率明细。",
        "已隐藏 Hub 上网率明细表格；如需查看请勾选“显示详细数据表”。",
        hide_index=True,
    )

    st.markdown(f"#### {tr('timeliness_quality_breakdown_title')}")
    timeliness_quality_df = build_timeliness_quality_breakdown_table(dsp_metric_scope_df, thresholds=[24, 48, 72])
    _render_optional_dataframe(
        style_breakdown_rows(timeliness_quality_df),
        show_details,
        tr("kpi_empty"),
        "已隐藏时效与质量明细表格；如需查看请勾选“显示详细数据表”。",
        hide_index=True,
    )

    attempt_detail_export_df = dsp_metric_scope_df.copy()

    attempt_header_cols = st.columns([4, 1])
    attempt_header_cols[1].download_button(
        "下载Route明细",
        data=attempt_detail_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"route_attempt_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=attempt_detail_export_df.empty,
    )
    _render_optional_dataframe(
        attempt_detail_export_df,
        show_details,
        "暂无 Route 明细。",
        "已隐藏 Route 明细表格；如需查看请勾选“显示详细数据表”。",
    )

    if layout_mode == "compact":
        selected_eval_weight = calculate_package_evaluation_weight(result_df).sum()
        st.metric(tr("compact_eval_weight"), f"{selected_eval_weight:.2f}")
        return kpi_payload

    render_daily_kpi_charts(result_df)

    st.markdown("#### 12/24/48/72h Scan Rate (Pickup -> Scan)")
    scan_detail_columns = [
        "tracking_id",
        "Region",
        "State",
        "shipperName",
        "created_time",
        "first_scanned_time",
    ]
    scan_detail_df = result_df.reindex(columns=scan_detail_columns, fill_value="").copy()
    scan_detail_df["created_dt"] = to_datetime_series(scan_detail_df, "created_time")
    scan_detail_df["first_scanned_dt"] = to_datetime_series(scan_detail_df, "first_scanned_time")
    scan_detail_df["created_to_scan_hours"] = (
        scan_detail_df["first_scanned_dt"] - scan_detail_df["created_dt"]
    ).dt.total_seconds() / 3600
    for threshold in [12, 24, 48, 72]:
        scan_detail_df[f"within_{threshold}h"] = (
            scan_detail_df["first_scanned_dt"].notna()
            & (scan_detail_df["created_to_scan_hours"] >= 0)
            & (scan_detail_df["created_to_scan_hours"] < threshold)
        )

    scan_header_cols = st.columns([4, 1])
    scan_header_cols[1].download_button(
        tr("download_scan"),
        data=scan_detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"scan_rate_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=scan_detail_df.empty,
    )

    scan_cols = st.columns(4)
    for i, threshold in enumerate([12, 24, 48, 72]):
        metric = metric_map.get(f"<{threshold}h scan rate")
        if not metric:
            scan_cols[i].metric(f"<{threshold}h scan rate", "0.00%", "0/0")
            scan_cols[i].info("no data available")
            continue
        scan_cols[i].metric(
            metric["metric"],
            f"{metric['rate']:.2%}",
            f"{metric['hit']}/{metric['total']}",
        )

    st.markdown("#### 数据总览")
    overview_columns = list(result_df.columns)
    preferred_front_columns = [
        "tracking_id",
        "Region",
        "State",
        "shipperName",
        "created_time",
        "first_scanned_time",
        "last_scanned_time",
        "out_for_delivery_time",
        "attempted_time",
        "delivered_time",
    ]
    ordered_columns = [col for col in preferred_front_columns if col in overview_columns] + [
        col for col in overview_columns if col not in preferred_front_columns
    ]
    overview_df = result_df.reindex(columns=ordered_columns).copy()
    overview_df["created_dt"] = to_datetime_series(overview_df, "created_time")
    overview_df["first_scanned_dt"] = to_datetime_series(overview_df, "first_scanned_time")
    overview_df["created_to_scan_hours"] = (
        overview_df["first_scanned_dt"] - overview_df["created_dt"]
    ).dt.total_seconds() / 3600
    for threshold in [12, 24, 48, 72]:
        overview_df[f"within_{threshold}h"] = (
            overview_df["first_scanned_dt"].notna()
            & (overview_df["created_to_scan_hours"] >= 0)
            & (overview_df["created_to_scan_hours"] < threshold)
        )
    st.dataframe(overview_df, use_container_width=True)

    scan_detail_df = scan_detail_df.drop(columns=["created_dt", "first_scanned_dt"])

    st.markdown("#### Monthly Lost Rate (no events within 120h after Last Scan; exclude waybills not yet 120h old)")
    monthly_lost_metric = next((m for m in kpi_payload["metrics"] if m.get("metric") == "overall monthly lost rate"), None)

    first_scanned_dt = to_datetime_series(result_df, "first_scanned_time")
    last_scanned_dt = to_datetime_series(result_df, "last_scanned_time")
    ofd_dt = to_datetime_series(result_df, "out_for_delivery_time")
    attempted_dt = to_datetime_series(result_df, "attempted_time")
    delivered_dt = to_datetime_series(result_df, "delivered_time")

    analysis_df = result_df.copy()
    analysis_df["first_scanned_dt"] = first_scanned_dt
    analysis_df["last_scanned_dt"] = last_scanned_dt
    analysis_df["ofd_dt"] = ofd_dt
    analysis_df["attempted_dt"] = attempted_dt
    analysis_df["delivered_dt"] = delivered_dt

    lost_analysis = build_lost_package_analysis(analysis_df, fetch_reference_time=fetch_reference_time)
    lost_condition = lost_analysis["lost_mask"]

    lost_detail_columns = [
        "tracking_id",
        "Region",
        "State",
        "shipperName",
        "created_time",
        "first_scanned_time",
        "last_scanned_time",
        "out_for_delivery_time",
        "attempted_time",
        "delivered_time",
    ]
    lost_detail_df = result_df.loc[lost_condition].copy()
    for column in lost_detail_columns:
        if column not in lost_detail_df.columns:
            lost_detail_df[column] = pd.NA
    lost_detail_df = lost_detail_df[lost_detail_columns]

    if kpi_payload.get("has_monthly_lost_data") and monthly_lost_metric:
        metric_cols = st.columns([2, 1])
        metric_cols[0].metric("Overall monthly lost rate", f"{monthly_lost_metric['rate']:.2%}")
        metric_cols[1].download_button(
            tr("download_lost"),
            data=lost_detail_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"lost_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=lost_detail_df.empty,
        )
        render_percentage_pie(
            "Lost share",
            int(monthly_lost_metric["hit"]),
            int(monthly_lost_metric["total"]),
            hit_label="Lost",
            miss_label="Not lost",
            chart_key=f"lost_{refresh_key}",
        )
        st.markdown(f"##### {tr('lost_detail')}")
        _render_optional_dataframe(
            lost_detail_df,
            show_details,
            tr("lost_empty"),
            "已隐藏丢包明细表格；如需查看请勾选“显示详细数据表”。",
        )
    else:
        st.info(tr("lost_no_scan"))

    st.markdown("#### Monthly Damage Rate (Reserved)")
    st.info("Reserved area: monthly damage rate metric to be implemented.")

    st.markdown("#### Interception Success Rate (Reserved)")
    st.info("Reserved area: interception success-rate metric to be implemented.")

    return kpi_payload


def build_layout_specific_report_payload(kpi_payload: dict[str, Any], layout_mode: str) -> dict[str, Any]:
    if layout_mode != "compact":
        return kpi_payload

    compact_metric_names = {"<24h delivery rate", "<48h delivery rate", "<72h delivery rate", "POD qualified rate", "24h attempt rate", "DSP lost rate", "Warehouse lost rate", "Intercept success rate"}
    compact_metrics = [m for m in kpi_payload.get("metrics", []) if m.get("metric") in compact_metric_names]
    compact_charts = [c for c in kpi_payload.get("charts", []) if c.get("chart") in compact_metric_names]

    return {
        **kpi_payload,
        "metrics": compact_metrics,
        "charts": compact_charts,
    }


def _render_optional_dataframe(df: Any, show_details: bool, empty_message: str, hidden_message: str, *, use_container_width: bool = True, hide_index: bool = False) -> None:
    source_df = df.data if hasattr(df, "data") and isinstance(df.data, pd.DataFrame) else df
    if isinstance(source_df, pd.DataFrame) and source_df.empty:
        st.info(empty_message)
        return
    if show_details:
        st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)
        return
    st.caption(hidden_message)


def build_layout_specific_export_df(filtered_df: pd.DataFrame, layout_mode: str) -> pd.DataFrame:
    if layout_mode != "compact":
        return build_export_df(filtered_df)

    non_pickup_df, _ = route_utils.split_pickup_routes(filtered_df)
    delivered_detail_columns = [
        "tracking_id",
        "Region",
        "State",
        "shipperName",
        "Hub",
        "Contractor",
        "Route_name",
        "out_for_delivery_time",
        "delivered_time",
    ]
    delivered_detail_df = build_non_pickup_detail_df(non_pickup_df, delivered_detail_columns)
    delivered_detail_df["ofd_dt"] = to_datetime_series(delivered_detail_df, "out_for_delivery_time")
    delivered_detail_df["delivered_dt"] = to_datetime_series(delivered_detail_df, "delivered_time")
    delivered_detail_df["ofd_to_delivered_hours"] = (
        delivered_detail_df["delivered_dt"] - delivered_detail_df["ofd_dt"]
    ).dt.total_seconds() / 3600
    delivered_detail_df["within_24h"] = (
        delivered_detail_df["delivered_dt"].notna()
        & (delivered_detail_df["ofd_to_delivered_hours"] >= 0)
        & (delivered_detail_df["ofd_to_delivered_hours"] < 24)
    )

    compact_breakdown_df = build_delivery_breakdown_table(delivered_detail_df, thresholds=[24])
    return compact_breakdown_df



def build_detailed_report_detail_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    non_pickup_df, _ = route_utils.split_pickup_routes(filtered_df)
    detail_columns = [
        "tracking_id",
        "Region",
        "State",
        "shipperName",
        "Hub",
        "Contractor",
        "Route_name",
        "created_time",
        "first_scanned_time",
        "last_scanned_time",
        "out_for_delivery_time",
        "attempted_time",
        "first_pod_complience",
        "second_pod_complience",
        "third_pod_complience",
        "entered_costomer_service",
        "beans_pod_link",
        "delivered_time",
    ]
    detail_df = build_non_pickup_detail_df(non_pickup_df, detail_columns)
    return detail_df


def build_non_pickup_detail_df(non_pickup_df: pd.DataFrame, detail_columns: list[str]) -> pd.DataFrame:
    out_for_delivery_time_series = non_pickup_df.get("out_for_delivery_time", pd.Series("", index=non_pickup_df.index))
    out_for_delivery_mask = out_for_delivery_time_series.notna() & out_for_delivery_time_series.astype(str).str.strip().ne("")

    detail_df = non_pickup_df.loc[out_for_delivery_mask].copy()
    for col in detail_columns:
        if col not in detail_df.columns:
            detail_df[col] = ""

    return detail_df.loc[:, detail_columns]


def process_tracking_ids(
    dedup_ids: list[str],
    receive_province_map: dict[str, str],
    sender_info_map: dict[str, dict[str, str]],
    router_messages_map: dict[str, Any],
    route_metadata_map: dict[str, dict[str, str]],
    progress_bar,
    status_text,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    failures: list[dict[str, str]] = []

    result_columns = OUTPUT_COLUMNS + POD_COLUMNS

    if not dedup_ids:
        return pd.DataFrame(columns=result_columns), failures

    total = len(dedup_ids)
    completed = 0

    def worker(tracking_id: str) -> tuple[str, dict[str, str], dict[str, str] | None]:
        try:
            payload = router_messages_map.get(tracking_id)
            if payload is None:
                return tracking_id, empty_row(tracking_id), {"tracking_id": tracking_id, "reason": "router_messages not found in DB"}

            normalized_payload = _normalize_router_payload(payload)

            if isinstance(normalized_payload, (dict, list)):
                row = build_row(tracking_id, normalized_payload, route_metadata_map=route_metadata_map)
                return tracking_id, row, None

            row = empty_row(tracking_id)
            return tracking_id, row, {"tracking_id": tracking_id, "reason": "router_messages is not valid JSON object/array"}
        except Exception as e:  # noqa: BLE001
            row = empty_row(tracking_id)
            return tracking_id, row, {"tracking_id": tracking_id, "reason": str(e)}

    max_workers = min(API_MAX_WORKERS, total)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tid = {executor.submit(worker, tracking_id): tracking_id for tracking_id in dedup_ids}

        for future in as_completed(future_to_tid):
            tracking_id, row, failure = future.result()
            state = str(receive_province_map.get(tracking_id) or "").strip()
            row["State"] = normalize_state(state)
            row["Region"] = infer_region_from_state(state)
            sender_info = sender_info_map.get(tracking_id, {})
            row["sender_company"] = str(sender_info.get("sender_company") or "").strip()
            row["sender_province"] = str(sender_info.get("sender_province") or "").strip()
            row["sender_city"] = str(sender_info.get("sender_city") or "").strip()
            row["sender_address"] = str(sender_info.get("sender_address") or "").strip()
            rows_by_id[tracking_id] = row

            if failure:
                failures.append(failure)

            completed += 1
            progress_value = progress_start + (progress_end - progress_start) * (completed / total)
            progress_bar.progress(progress_value)
            status_text.text(tr("processing", completed=completed, total=total, tracking_id=tracking_id))

    ordered_rows = [rows_by_id[tid] for tid in dedup_ids]
    return pd.DataFrame(ordered_rows, columns=result_columns), failures




def _normalize_router_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    text_payload = payload.strip()
    if not text_payload:
        return payload
    try:
        return json.loads(text_payload)
    except json.JSONDecodeError:
        return payload


def normalize_router_messages_map(router_messages_map: dict[str, Any]) -> dict[str, Any]:
    return {tracking_id: _normalize_router_payload(payload) for tracking_id, payload in router_messages_map.items()}

def _parse_address_components(full_address: Any) -> dict[str, str]:
    """Parse a full US-style address into address/state/city components."""
    address = str(full_address or "").strip()
    if not address:
        return {"address": "", "state": "", "city": ""}

    parts = [part.strip() for part in address.split(",") if str(part).strip()]
    state = ""
    city = ""

    if len(parts) >= 4:
        city = parts[-3]
        state = parts[-2]
    elif len(parts) == 3:
        city = parts[-2]
        state_match = re.search(r"\b([A-Za-z]{2})\b", parts[-1])
        state = state_match.group(1).upper() if state_match else parts[-1]
    elif len(parts) == 2:
        city = parts[-1]

    state = re.sub(r"[^A-Za-z]", "", str(state)).upper()[:2]
    city = str(city or "").strip()

    return {
        "address": address,
        "state": state,
        "city": city,
    }


def _extract_address_maps_from_router_payload(
    dedup_ids: list[str],
    router_messages_map: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """
    Build receiver-state map and sender-info map from router_messages label events.

    Receiver: event type=label and description contains "Label created" (but not pickup label).
    Sender: event type=label and description contains "Pickup label created".
    """
    receive_province_map: dict[str, str] = {}
    sender_info_map: dict[str, dict[str, str]] = {}

    for tracking_id in dedup_ids:
        payload = _normalize_router_payload(router_messages_map.get(tracking_id))

        if not isinstance(payload, (dict, list)):
            continue

        events = normalize_events(payload)
        receiver_address = ""
        sender_address = ""

        for event in events:
            if str(event.get("type") or "").strip().lower() != "label":
                continue

            description = str(event.get("description") or "").strip().lower()
            item = event.get("item")
            address = ""
            if isinstance(item, dict):
                address = str(item.get("address") or "").strip()
            if not address:
                continue

            if "pickup label created" in description:
                if not sender_address:
                    sender_address = address
                continue

            if "label created" in description and not receiver_address:
                receiver_address = address

            if receiver_address and sender_address:
                break

        receiver_parts = _parse_address_components(receiver_address)
        sender_parts = _parse_address_components(sender_address)

        if receiver_parts["state"]:
            receive_province_map[tracking_id] = receiver_parts["state"]

        sender_info_map[tracking_id] = {
            "sender_company": "",
            "sender_province": sender_parts["state"],
            "sender_city": sender_parts["city"],
            "sender_address": sender_parts["address"],
        }

    return receive_province_map, sender_info_map

def main() -> None:
    st.set_page_config(page_title="Fimile US Shipment Operations Dashboard", layout="wide")
    st.title(tr("app_title"))
    st.caption(tr("version", version=APP_VERSION))

    st.info(tr("route_info"))

    if "dedup_ids" not in st.session_state:
        st.session_state["dedup_ids"] = []
    if "db_raw_ids" not in st.session_state:
        st.session_state["db_raw_ids"] = []
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "failures" not in st.session_state:
        st.session_state["failures"] = []
    today = date.today()
    default_query_end = today
    default_query_start = today - timedelta(days=1)
    date_input_min = today - timedelta(days=365 * 5)
    date_input_max = today + timedelta(days=365 * 2)

    if "query_start_date" not in st.session_state:
        st.session_state["query_start_date"] = default_query_start
    if "query_end_date" not in st.session_state:
        st.session_state["query_end_date"] = default_query_end
    if "fetch_clicked_at" not in st.session_state:
        st.session_state["fetch_clicked_at"] = None
    if "report_filter_start_date" not in st.session_state:
        st.session_state["report_filter_start_date"] = None
    if "report_filter_end_date" not in st.session_state:
        st.session_state["report_filter_end_date"] = None
    if "applied_report_filter_start_date" not in st.session_state:
        st.session_state["applied_report_filter_start_date"] = None
    if "applied_report_filter_end_date" not in st.session_state:
        st.session_state["applied_report_filter_end_date"] = None
    if "metrics_ready" not in st.session_state:
        st.session_state["metrics_ready"] = False
    if "exclude_atl_wdr" not in st.session_state:
        st.session_state["exclude_atl_wdr"] = True
    if "language" not in st.session_state:
        st.session_state["language"] = "zh"
    if "unknown_contractor_overrides" not in st.session_state:
        st.session_state["unknown_contractor_overrides"] = {}
    if "contractor_override_hub" not in st.session_state:
        st.session_state["contractor_override_hub"] = ""
    if "contractor_override_name" not in st.session_state:
        st.session_state["contractor_override_name"] = ""
    st.selectbox(
        tr("language_label"),
        options=["zh", "en"],
        format_func=lambda x: tr("language_zh") if x == "zh" else tr("language_en"),
        key="language",
    )

    st.subheader(tr("input_section"))
    st.caption(f"{tr('input_mode')}: {tr('mode_db')}")

    st.session_state["query_start_date"] = min(max(st.session_state["query_start_date"], date_input_min), date_input_max)
    st.session_state["query_end_date"] = min(max(st.session_state["query_end_date"], date_input_min), date_input_max)

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input(
            tr("start_date"),
            key="query_start_date",
            min_value=date_input_min,
            max_value=date_input_max,
        )
    with c2:
        end_d = st.date_input(
            tr("end_date"),
            key="query_end_date",
            min_value=date_input_min,
            max_value=date_input_max,
        )

    raw_ids: list[str] = st.session_state.get("db_raw_ids", [])
    run_btn = st.button(tr("load_merge_btn"), type="primary", key="load_merge_btn")
    if run_btn:
        st.session_state["metrics_ready"] = False
        st.session_state["applied_report_filter_start_date"] = None
        st.session_state["applied_report_filter_end_date"] = None
        clear_query_caches()
        with st.spinner(tr("loading_db")):
            try:
                raw_ids = fetch_tracking_numbers_by_date(start_d, end_d)
                st.session_state["db_raw_ids"] = raw_ids
                if not raw_ids:
                    st.warning(tr("no_tracking_found"))
            except Exception as e:
                st.error(str(e))
                raw_ids = []
                st.session_state["db_raw_ids"] = []

    if raw_ids:
        with st.expander(tr("db_preview", count=len(raw_ids)), expanded=False):
            st.write(raw_ids[:50])

    st.markdown(f"#### {tr('report_filter_title')}")
    filter_cols = st.columns([1, 1, 0.7])
    with filter_cols[0]:
        st.date_input(tr("report_filter_start_label"), key="report_filter_start_date", value=None)
    with filter_cols[1]:
        st.date_input(tr("report_filter_end_label"), key="report_filter_end_date", value=None)
    with filter_cols[2]:
        st.write("")
        st.write("")
        calc_btn = st.button(
            tr("compute_metrics_btn"),
            type="secondary",
            key="compute_metrics_btn",
            use_container_width=True,
            disabled=st.session_state.get('result_df') is None,
        )

    st.toggle(tr("exclude_atl_wdr_toggle"), key="exclude_atl_wdr", value=True)

    cleaned, dedup_ids, counter = normalize_tracking_ids(raw_ids, uppercase=False)
    duplicate_ids = [k for k, v in counter.items() if v > 1]
    st.session_state["dedup_ids"] = dedup_ids

    m1, m2, m3 = st.columns(3)
    m1.metric("input_count", len(cleaned))
    m2.metric("unique_count", len(dedup_ids))
    m3.metric("duplicate_count", len(cleaned) - len(dedup_ids))

    if duplicate_ids:
        with st.expander(tr("duplicate_ids")):
            st.write(duplicate_ids)

    st.subheader(tr("fetch_section"))

    if run_btn and dedup_ids:
        st.session_state["fetch_clicked_at"] = datetime.now()
        failures: list[dict[str, str]] = []
        receive_province_map: dict[str, str] = {}
        sender_info_map: dict[str, dict[str, str]] = {}
        router_messages_map: dict[str, Any] = {}
        route_metadata_map: dict[str, dict[str, str]] = {}
        progress = st.progress(0)
        status = st.empty()

        setup_steps = 4
        completed_setup_steps = 0

        def update_setup_progress(message: str) -> None:
            nonlocal completed_setup_steps
            completed_setup_steps += 1
            progress.progress((completed_setup_steps / setup_steps) * 0.3)
            status.text(message)

        try:
            fetch_router_messages = getattr(db, "fetch_router_messages_map", None)
            if callable(fetch_router_messages):
                router_messages_map = normalize_router_messages_map(fetch_router_messages(tuple(dedup_ids)))
            else:
                fallback_fetch_router_messages = globals().get("fetch_router_messages_map")
                if callable(fallback_fetch_router_messages):
                    router_messages_map = normalize_router_messages_map(fallback_fetch_router_messages(tuple(dedup_ids)))
                else:
                    raise AttributeError(
                        f"module 'utils.db' has no attribute 'fetch_router_messages_map' (loaded from {getattr(db, '__file__', 'unknown')})"
                    )
        except Exception as e:
            st.warning(f"Failed to load router_messages from DB: {e}")
            router_messages_map = {}
        finally:
            update_setup_progress("Loading route event payloads...")

        try:
            route_metadata_map = build_route_metadata_map(router_messages_map)
        except Exception as e:
            st.warning(f"Failed to build route metadata cache: {e}")
            route_metadata_map = {}
        finally:
            update_setup_progress("Building route/assignee cache...")

        try:
            receive_province_map, sender_info_map = _extract_address_maps_from_router_payload(
                dedup_ids,
                router_messages_map,
            )
        except Exception as e:
            st.warning(tr("state_region_fail", error=e))
            receive_province_map = {}
            sender_info_map = {}
        finally:
            update_setup_progress("Loading recipient location data...")
            update_setup_progress("Loading sender profile data...")

        result_df, failures = process_tracking_ids(
            dedup_ids=dedup_ids,
            receive_province_map=receive_province_map,
            sender_info_map=sender_info_map,
            router_messages_map=router_messages_map,
            route_metadata_map=route_metadata_map,
            progress_bar=progress,
            status_text=status,
            progress_start=0.3,
            progress_end=1.0,
        )

        result_df = fill_route_identity_columns(result_df)
        result_df = ensure_compatibility_columns(result_df)

        st.session_state["result_df"] = result_df
        st.session_state["failures"] = failures

        status.text(tr("done"))
        st.session_state["metrics_ready"] = False
        st.session_state["applied_report_filter_start_date"] = None
        st.session_state["applied_report_filter_end_date"] = None

    result_df: pd.DataFrame | None = st.session_state.get("result_df")
    failures: list[dict[str, str]] = st.session_state.get("failures", [])

    if result_df is not None and calc_btn:
        start_date = st.session_state.get("report_filter_start_date")
        end_date = st.session_state.get("report_filter_end_date")
        if start_date is not None and end_date is not None and start_date > end_date:
            st.error(tr("report_filter_invalid"))
            st.session_state["metrics_ready"] = False
        else:
            st.session_state["applied_report_filter_start_date"] = start_date
            st.session_state["applied_report_filter_end_date"] = end_date
            st.session_state["metrics_ready"] = True

    if result_df is not None:
        known_hub_states = set(HUB_BY_STATE.keys())
        state_series = result_df["State"].fillna("").astype(str).str.strip().str.upper()
        unknown_states = sorted({state for state in state_series if state and state not in known_hub_states})
        if unknown_states:
            st.warning(f"发现未配置 HUB 映射的 State：{', '.join(unknown_states)}")

        available_hubs = sorted(
            result_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub").unique().tolist()
        )
        if available_hubs and st.session_state["contractor_override_hub"] not in available_hubs:
            st.session_state["contractor_override_hub"] = available_hubs[0]

        override_c1, override_c2, override_c3 = st.columns([2, 2, 1])
        with override_c1:
            st.selectbox(
                tr("override_hub_dropdown_label"),
                options=available_hubs,
                key="contractor_override_hub",
            )
        with override_c2:
            st.text_input(
                tr("override_contractor_input_label"),
                key="contractor_override_name",
                placeholder=tr("override_contractor_input_placeholder"),
            )
        with override_c3:
            st.write("")
            if st.button(tr("override_apply_btn"), use_container_width=True, key="override_apply_btn"):
                selected_hub = str(st.session_state.get("contractor_override_hub", "")).strip()
                contractor_name = str(st.session_state.get("contractor_override_name", "")).strip()
                if selected_hub and contractor_name:
                    overrides = dict(st.session_state.get("unknown_contractor_overrides", {}))
                    overrides[selected_hub] = contractor_name
                    st.session_state["unknown_contractor_overrides"] = overrides
                    st.success(tr("override_apply_success", hub=selected_hub, contractor=contractor_name))
                else:
                    st.warning(tr("override_apply_validation"))

        if st.session_state.get("unknown_contractor_overrides"):
            st.caption(tr("override_applied_title"))
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Hub": hub, "Contractor": contractor}
                        for hub, contractor in st.session_state["unknown_contractor_overrides"].items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        result_df = apply_manual_dimension_overrides(result_df)

        metrics_ready = bool(st.session_state.get("metrics_ready", False))
        applied_filter_start = st.session_state.get("applied_report_filter_start_date")
        applied_filter_end = st.session_state.get("applied_report_filter_end_date")

        report_start_dt = None
        report_end_dt = None
        if applied_filter_start is not None:
            report_start_dt = datetime.combine(applied_filter_start, datetime.min.time())
        if applied_filter_end is not None:
            report_end_dt = datetime.combine(applied_filter_end, datetime.max.time())

        success_count = len(result_df) - len(failures)
        fail_count = len(failures)

        s1, s2 = st.columns(2)
        s1.metric(tr("success_count"), success_count)
        s2.metric(tr("fail_count"), fail_count)

        if failures:
            st.error(tr("request_fail"))
            fail_df = pd.DataFrame(failures)
            st.dataframe(fail_df, use_container_width=True)
            st.download_button(
                tr("download_fail_csv"),
                data=fail_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        filtered_df = result_df
        if not metrics_ready:
            st.info(tr("compute_metrics_prompt"))

        excluded_hub_df = pd.DataFrame()
        excluded_hub_route_attempts_df = pd.DataFrame()
        if st.session_state.get("exclude_atl_wdr", True):
            excluded_hub_df = filtered_df[
                filtered_df["Hub"].fillna("").astype(str).str.strip().str.upper().isin(["ATL", "WDR"])
            ].copy()

        if metrics_ready and (applied_filter_start is not None or applied_filter_end is not None):
            applied_parts = []
            if applied_filter_start is not None:
                applied_parts.append(f"{tr('report_filter_start_label')}: {applied_filter_start.strftime('%Y/%m/%d')}")
            if applied_filter_end is not None:
                applied_parts.append(f"{tr('report_filter_end_label')}: {applied_filter_end.strftime('%Y/%m/%d')}")
            st.caption(tr("compute_metrics_applied_caption", filters='；'.join(applied_parts)))
        elif metrics_ready:
            st.caption(tr("compute_metrics_applied_caption", filters=tr("report_filter_empty")))

        if not metrics_ready:
            return

        layout_mode = st.radio(
            tr("layout_mode_label"),
            options=["detailed", "compact"],
            index=0,
            horizontal=True,
            format_func=lambda x: tr("layout_mode_detailed") if x == "detailed" else tr("layout_mode_compact"),
            key="kpi_layout_mode",
        )
        show_detailed_tables = st.checkbox(
            "显示详细数据表（数据量大时建议关闭）",
            value=False,
            help="关闭后仅保留数据总览表格和下载按钮，以减少发送到浏览器的数据量。",
            key="show_detailed_tables",
        )

        route_attempts_df, unresolved_attempts_df, canceled_attempts_df, lost_attempts_df = build_route_attempts_view(filtered_df)
        if not excluded_hub_df.empty:
            excluded_hub_route_attempts_df, _, _, _ = build_route_attempts_view(excluded_hub_df)

        kpi_payload = render_kpi_charts(
            filtered_df,
            layout_mode=layout_mode,
            fetch_reference_time=st.session_state.get("fetch_clicked_at"),
            route_attempts_df=route_attempts_df,
            show_details=show_detailed_tables,
            report_start_dt=report_start_dt,
            report_end_dt=report_end_dt,
            exclude_atl_wdr=bool(st.session_state.get("exclude_atl_wdr", True)),
        )

        if st.session_state.get("exclude_atl_wdr", True):
            with st.expander("ATL/WDR 数据隔离区（不计入指标计算）", expanded=False):
                st.caption("当开启“去除WDR和ATL”时，这两个Hub的数据会在指标计算中排除，但会保留在此处单独展示。")
                st.write(f"ATL/WDR 总记录数：{len(excluded_hub_df)}")
                if excluded_hub_df.empty:
                    st.info("当前没有 ATL/WDR 数据。")
                else:
                    _render_optional_dataframe(
                        excluded_hub_df,
                        show_detailed_tables,
                        "当前没有 ATL/WDR 数据。",
                        "已隐藏 ATL/WDR 原始数据表格；如需查看请勾选“显示详细数据表”。",
                    )
                    st.markdown("**ATL/WDR Route尝试明细**")
                    _render_optional_dataframe(
                        excluded_hub_route_attempts_df,
                        show_detailed_tables,
                        "当前没有 ATL/WDR Route 尝试明细。",
                        "已隐藏 ATL/WDR Route 尝试明细表格；如需查看请勾选“显示详细数据表”。",
                    )

        st.subheader(tr("result_preview"))
        display_df = build_tracking_display_df(
            filtered_df,
            route_attempts_df=route_attempts_df,
            unresolved_attempts_df=unresolved_attempts_df,
            canceled_attempts_df=canceled_attempts_df,
            lost_attempts_df=lost_attempts_df,
        )
        if display_df.empty:
            st.info("No records under current filters.")
        else:
            _render_optional_dataframe(
                display_df,
                show_detailed_tables,
                "No records under current filters.",
                "已隐藏结果预览表格；如需查看请勾选“显示详细数据表”。",
            )

        st.subheader(tr("customer_summary_section"))
        customer_summary_df = build_customer_address_summary(filtered_df)
        if customer_summary_df.empty:
            st.info(tr("customer_summary_empty"))
        else:
            _render_optional_dataframe(
                customer_summary_df,
                show_detailed_tables,
                tr("customer_summary_empty"),
                "已隐藏取货仓库表格；如需查看请勾选“显示详细数据表”。",
            )
            st.download_button(
                tr("download_customer_summary"),
                data=customer_summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"customer_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        st.subheader(tr("route_attempts_section"))
        _render_optional_dataframe(
            route_attempts_df,
            show_detailed_tables,
            tr("route_attempts_empty"),
            "已隐藏 Route 尝试明细表格；如需查看请勾选“显示详细数据表”。",
        )

        multi_route_tracking_df = build_multi_route_tracking_view(route_attempts_df)
        st.subheader(tr("multi_route_tracking_section"))
        _render_optional_dataframe(
            multi_route_tracking_df,
            show_detailed_tables,
            tr("multi_route_tracking_empty"),
            "已隐藏多次 Route 单号表格；如需查看请勾选“显示详细数据表”。",
        )

        st.subheader(tr("route_attempts_unresolved_section"))
        _render_optional_dataframe(
            unresolved_attempts_df,
            show_detailed_tables,
            tr("route_attempts_unresolved_empty"),
            "已隐藏未解析 Route 明细表格；如需查看请勾选“显示详细数据表”。",
        )

        st.subheader(tr("route_attempts_canceled_section"))
        _render_optional_dataframe(
            canceled_attempts_df,
            show_detailed_tables,
            tr("route_attempts_canceled_empty"),
            "已隐藏取消件明细表格；如需查看请勾选“显示详细数据表”。",
        )

        st.subheader(tr("route_attempts_lost_section"))
        _render_optional_dataframe(
            lost_attempts_df,
            show_detailed_tables,
            tr("route_attempts_lost_empty"),
            "已隐藏丢件明细表格；如需查看请勾选“显示详细数据表”。",
        )

        invalid_route_df = build_invalid_route_summary(filtered_df)
        st.subheader(tr("invalid_route_section"))
        _render_optional_dataframe(
            invalid_route_df,
            show_detailed_tables,
            tr("invalid_route_empty"),
            "已隐藏异常 Route 汇总表格；如需查看请勾选“显示详细数据表”。",
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_base_df = filtered_df.copy()
        if st.session_state.get("exclude_atl_wdr", True) and "Hub" in export_base_df.columns:
            export_base_df = export_base_df[
                ~export_base_df["Hub"].fillna("").astype(str).str.strip().str.upper().isin(["ATL", "WDR"])
            ].copy()

        export_df = build_layout_specific_export_df(export_base_df, layout_mode)
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        report_payload = build_layout_specific_report_payload(kpi_payload, layout_mode)
        kpi_report_data = None
        c_csv, c_report = st.columns(2)
        c_csv.download_button(
            tr("download_csv"),
            data=csv_data,
            file_name=f"export_{layout_mode}_{stamp}.csv",
            mime="text/csv",
        )
        report_detail_df = build_detailed_report_detail_df(export_base_df) if layout_mode == "detailed" else export_df
        report_filename = f"kpi_report_{layout_mode}_{stamp}.xlsx"
        try:
            kpi_report_data = kpi_report_to_excel_bytes(
                report_payload,
                report_detail_df,
                layout_mode=layout_mode,
                source_df=export_base_df,
            )
            c_report.download_button(
                tr("download_report"),
                data=kpi_report_data,
                file_name=report_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as err:
            c_report.error(tr("report_export_failed").format(error=str(err)))


if __name__ == "__main__":
    main()
