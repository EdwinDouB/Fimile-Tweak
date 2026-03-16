from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import ast
import json

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
                    "POD是否合格": "是" if bool(matched_terminal.get("POD")) else "否",
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

def render_daily_kpi_charts(result_df: pd.DataFrame) -> None:
    chart_df = result_df.copy()
    chart_df["_created_date"] = to_datetime_series(chart_df, "created_time").dt.date
    chart_df["_delivered_date"] = to_datetime_series(chart_df, "delivered_time").dt.date
    chart_df["_evaluation_weight"] = calculate_package_evaluation_weight(chart_df)

    created_count_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")
        .size()
        .rename("Package Count")
        .reset_index()
        .sort_values("_created_date")
    )
    delivered_count_df = (
        chart_df[chart_df["_delivered_date"].notna()]
        .groupby("_delivered_date")
        .size()
        .rename("Package Count")
        .reset_index()
        .sort_values("_delivered_date")
    )
    evaluation_weight_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")["_evaluation_weight"]
        .sum()
        .rename("Evaluation Weight")
        .reset_index()
        .sort_values("_created_date")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {tr('daily_created_chart')}")
        if created_count_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.line_chart(created_count_df.set_index("_created_date")["Package Count"])

    with c2:
        st.markdown(f"#### {tr('daily_delivered_chart')}")
        if delivered_count_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.line_chart(delivered_count_df.set_index("_delivered_date")["Package Count"])

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
    route_attempt_metrics = build_route_attempt_metrics(metric_source_df)

    st.markdown("#### OFD派送时效看板（按每次派送计算）")
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

    st.caption("口径：基于“按派送尝试整理的Route明细”表计算，分母=该表全部条目。")

    st.markdown(f"#### {tr('timeliness_quality_breakdown_title')}")
    timeliness_quality_df = build_timeliness_quality_breakdown_table(metric_source_df, thresholds=[24, 48, 72])
    st.dataframe(style_breakdown_rows(timeliness_quality_df), use_container_width=True, hide_index=True)

    attempt_detail_export_df = metric_source_df.copy()

    attempt_header_cols = st.columns([4, 1])
    attempt_header_cols[1].download_button(
        "下载Route明细",
        data=attempt_detail_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"route_attempt_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=attempt_detail_export_df.empty,
    )
    st.dataframe(attempt_detail_export_df, use_container_width=True)

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
    scan_detail_df = scan_detail_df.drop(columns=["created_dt", "first_scanned_dt"])

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
        render_percentage_pie(
            title=f"<{threshold}h scan share",
            hit_count=int(metric["hit"]),
            total_count=int(metric["total"]),
            hit_label=f"<{threshold}h scanned",
            miss_label=f">={threshold}h or unscanned",
            chart_key=f"scan_{threshold}_{refresh_key}",
            container=scan_cols[i],
        )

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
        if lost_detail_df.empty:
            st.info(tr("lost_empty"))
        else:
            st.dataframe(lost_detail_df, use_container_width=True)
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

    compact_metric_names = {"<24h delivery rate", "<48h delivery rate", "<72h delivery rate", "Manual POD qualified rate", "24h attempt rate"}
    compact_metrics = [m for m in kpi_payload.get("metrics", []) if m.get("metric") in compact_metric_names]
    compact_charts = [c for c in kpi_payload.get("charts", []) if c.get("chart") in compact_metric_names]

    return {
        **kpi_payload,
        "metrics": compact_metrics,
        "charts": compact_charts,
    }


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
        "first_out_for_delivery_date",
        "first_failed_date",
        "first_pod_complience",
        "second_out_for_delivery_date",
        "second_failed_date",
        "second_pod_complience",
        "third_out_for_delivery_date",
        "third_failed_date",
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

    def _serialize_router_messages(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return str(payload)

    def worker(tracking_id: str) -> tuple[str, dict[str, str], dict[str, str] | None]:
        try:
            payload = router_messages_map.get(tracking_id)
            if payload is None:
                return tracking_id, empty_row(tracking_id), {"tracking_id": tracking_id, "reason": "router_messages not found in DB"}

            normalized_payload = payload
            if isinstance(payload, str):
                text_payload = payload.strip()
                if text_payload:
                    try:
                        normalized_payload = json.loads(text_payload)
                    except json.JSONDecodeError:
                        normalized_payload = payload

            if isinstance(normalized_payload, (dict, list)):
                row = build_row(tracking_id, normalized_payload)
                return tracking_id, row, None

            row = empty_row(tracking_id)
            row["router_messages"] = _serialize_router_messages(payload)
            return tracking_id, row, {"tracking_id": tracking_id, "reason": "router_messages is not valid JSON object/array"}
        except Exception as e:  # noqa: BLE001
            row = empty_row(tracking_id)
            row["router_messages"] = _serialize_router_messages(router_messages_map.get(tracking_id))
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

def _filter_options_from_df(source_df: pd.DataFrame, column: str) -> list[str]:
    values = source_df[column].fillna("").astype(str).str.strip()
    return sorted([item for item in values.unique().tolist() if item])


def _apply_dimension_filters(
    source_df: pd.DataFrame,
    selected_region: str,
    selected_state: str,
    selected_driver: str,
    selected_hub: str,
    selected_contractor: str,
) -> pd.DataFrame:
    filtered_df = source_df.copy()
    all_value = tr("all")

    if selected_region != all_value:
        filtered_df = filtered_df[filtered_df["Region"].fillna("").astype(str).str.strip() == selected_region]
    if selected_state != all_value:
        filtered_df = filtered_df[filtered_df["State"].fillna("").astype(str).str.strip() == selected_state]
    if selected_driver != all_value:
        filtered_df = filtered_df[filtered_df["Driver"].fillna("").astype(str).str.strip() == selected_driver]
    if selected_hub != all_value:
        filtered_df = filtered_df[filtered_df["Hub"].fillna("").astype(str).str.strip() == selected_hub]
    if selected_contractor != all_value:
        filtered_df = filtered_df[filtered_df["Contractor"].fillna("").astype(str).str.strip() == selected_contractor]

    return filtered_df


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
    if "delivery_filter_start" not in st.session_state:
        st.session_state["delivery_filter_start"] = st.session_state["query_start_date"]
    if "delivery_filter_end" not in st.session_state:
        st.session_state["delivery_filter_end"] = st.session_state["query_end_date"]
    if "applied_delivery_filter_start" not in st.session_state:
        st.session_state["applied_delivery_filter_start"] = st.session_state["delivery_filter_start"]
    if "applied_delivery_filter_end" not in st.session_state:
        st.session_state["applied_delivery_filter_end"] = st.session_state["delivery_filter_end"]
    if "fetch_clicked_at" not in st.session_state:
        st.session_state["fetch_clicked_at"] = None
    if "language" not in st.session_state:
        st.session_state["language"] = "zh"
    if "hide_unknown_dimensions" not in st.session_state:
        st.session_state["hide_unknown_dimensions"] = False
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

    all_value = tr("all")

    def _is_all_value(value: object) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower()
        return normalized in {all_value.strip().lower(), "all", "全部"}

    for filter_key in ["region_filter", "state_filter", "driver_filter", "hub_filter", "contractor_filter"]:
        if filter_key not in st.session_state or _is_all_value(st.session_state[filter_key]):
            st.session_state[filter_key] = all_value

    for applied_filter_key, base_filter_key in [
        ("applied_region_filter", "region_filter"),
        ("applied_state_filter", "state_filter"),
        ("applied_driver_filter", "driver_filter"),
        ("applied_hub_filter", "hub_filter"),
        ("applied_contractor_filter", "contractor_filter"),
    ]:
        if applied_filter_key not in st.session_state or _is_all_value(st.session_state[applied_filter_key]):
            st.session_state[applied_filter_key] = st.session_state[base_filter_key]

    st.subheader(tr("input_section"))
    st.caption(f"{tr('input_mode')}: {tr('mode_db')}")

    st.session_state["query_start_date"] = min(max(st.session_state["query_start_date"], date_input_min), date_input_max)
    st.session_state["query_end_date"] = min(max(st.session_state["query_end_date"], date_input_min), date_input_max)
    st.session_state["delivery_filter_start"] = min(max(st.session_state["delivery_filter_start"], date_input_min), date_input_max)
    st.session_state["delivery_filter_end"] = min(max(st.session_state["delivery_filter_end"], date_input_min), date_input_max)

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
        clear_query_caches()
        with st.spinner(tr("loading_db")):
            try:
                raw_ids = fetch_tracking_numbers_by_date(start_d, end_d)
                st.session_state["db_raw_ids"] = raw_ids
                if not raw_ids:
                    st.warning(tr("no_tracking_found"))
                st.session_state["delivery_filter_start"] = start_d
                st.session_state["delivery_filter_end"] = end_d
                st.session_state["applied_delivery_filter_start"] = start_d
                st.session_state["applied_delivery_filter_end"] = end_d
            except Exception as e:
                st.error(str(e))
                raw_ids = []
                st.session_state["db_raw_ids"] = []

    if raw_ids:
        with st.expander(tr("db_preview", count=len(raw_ids)), expanded=False):
            st.write(raw_ids[:50])

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
        progress = st.progress(0)
        status = st.empty()

        setup_steps = 3
        completed_setup_steps = 0

        def update_setup_progress(message: str) -> None:
            nonlocal completed_setup_steps
            completed_setup_steps += 1
            progress.progress((completed_setup_steps / setup_steps) * 0.3)
            status.text(message)

        try:
            receive_province_map = fetch_receive_province_map(tuple(dedup_ids))
        except Exception as e:
            st.warning(tr("state_region_fail", error=e))
        finally:
            update_setup_progress("Loading recipient location data...")

        try:
            sender_info_map = fetch_sender_info_map(tuple(dedup_ids))
        except Exception:
            sender_info_map = {}
        finally:
            update_setup_progress("Loading sender profile data...")

        try:
            fetch_router_messages = getattr(db, "fetch_router_messages_map", None)
            if callable(fetch_router_messages):
                router_messages_map = fetch_router_messages(tuple(dedup_ids))
            else:
                fallback_fetch_router_messages = globals().get("fetch_router_messages_map")
                if callable(fallback_fetch_router_messages):
                    router_messages_map = fallback_fetch_router_messages(tuple(dedup_ids))
                else:
                    raise AttributeError(
                        f"module 'utils.db' has no attribute 'fetch_router_messages_map' (loaded from {getattr(db, '__file__', 'unknown')})"
                    )
        except Exception as e:
            st.warning(f"Failed to load router_messages from DB: {e}")
            router_messages_map = {}
        finally:
            update_setup_progress("Loading route event payloads...")

        result_df, failures = process_tracking_ids(
            dedup_ids=dedup_ids,
            receive_province_map=receive_province_map,
            sender_info_map=sender_info_map,
            router_messages_map=router_messages_map,
            progress_bar=progress,
            status_text=status,
            progress_start=0.3,
            progress_end=1.0,
        )

        result_df = fill_route_identity_columns(result_df)

        st.session_state["result_df"] = result_df
        st.session_state["failures"] = failures

        status.text(tr("done"))

    result_df: pd.DataFrame | None = st.session_state.get("result_df")
    failures: list[dict[str, str]] = st.session_state.get("failures", [])

    if result_df is not None:
        known_hub_states = set(HUB_BY_STATE.keys())
        state_series = result_df["State"].fillna("").astype(str).str.strip().str.upper()
        unknown_states = sorted({state for state in state_series if state and state not in known_hub_states})
        if unknown_states:
            st.warning(f"发现未配置 HUB 映射的 State：{', '.join(unknown_states)}")

        st.subheader(tr("filter_view"))
        toggle_label = tr("show_unknown_btn") if st.session_state.get("hide_unknown_dimensions", False) else tr("hide_unknown_btn")
        if st.button(toggle_label, key="toggle_unknown_dimensions"):
            st.session_state["hide_unknown_dimensions"] = not st.session_state.get("hide_unknown_dimensions", False)

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

        all_value = tr("all")
        region_options = [all_value] + _filter_options_from_df(result_df, "Region")
        if st.session_state["region_filter"] not in region_options:
            st.session_state["region_filter"] = all_value

        state_candidate_df = _apply_dimension_filters(
            result_df,
            selected_region=st.session_state["region_filter"],
            selected_state=all_value,
            selected_driver=all_value,
            selected_hub=all_value,
            selected_contractor=all_value,
        )
        state_options = [all_value] + _filter_options_from_df(state_candidate_df, "State")
        if st.session_state["state_filter"] not in state_options:
            st.session_state["state_filter"] = all_value

        driver_candidate_df = _apply_dimension_filters(
            result_df,
            selected_region=st.session_state["region_filter"],
            selected_state=st.session_state["state_filter"],
            selected_driver=all_value,
            selected_hub=st.session_state["hub_filter"],
            selected_contractor=st.session_state["contractor_filter"],
        )
        driver_options = [all_value] + _filter_options_from_df(driver_candidate_df, "Driver")
        if st.session_state["driver_filter"] not in driver_options:
            st.session_state["driver_filter"] = all_value

        hub_candidate_df = _apply_dimension_filters(
            result_df,
            selected_region=st.session_state["region_filter"],
            selected_state=st.session_state["state_filter"],
            selected_driver=st.session_state["driver_filter"],
            selected_hub=all_value,
            selected_contractor=st.session_state["contractor_filter"],
        )
        hub_options = [all_value] + _filter_options_from_df(hub_candidate_df, "Hub")
        if st.session_state["hub_filter"] not in hub_options:
            st.session_state["hub_filter"] = all_value

        contractor_candidate_df = _apply_dimension_filters(
            result_df,
            selected_region=st.session_state["region_filter"],
            selected_state=st.session_state["state_filter"],
            selected_driver=st.session_state["driver_filter"],
            selected_hub=st.session_state["hub_filter"],
            selected_contractor=all_value,
        )
        contractor_options = [all_value] + _filter_options_from_df(contractor_candidate_df, "Contractor")
        if st.session_state["contractor_filter"] not in contractor_options:
            st.session_state["contractor_filter"] = all_value

        filter_c1, filter_c2, filter_c3, filter_c4, filter_c5, filter_c6 = st.columns([1, 1, 1, 1, 1, 0.8])
        with filter_c1:
            selected_region = st.selectbox("Region", options=region_options, key="region_filter")
        with filter_c2:
            selected_state = st.selectbox("State", options=state_options, key="state_filter")
        with filter_c3:
            selected_driver = st.selectbox("Driver", options=driver_options, key="driver_filter")
        with filter_c4:
            selected_hub = st.selectbox("Hub", options=hub_options, key="hub_filter")
        with filter_c5:
            selected_contractor = st.selectbox("Contractor", options=contractor_options, key="contractor_filter")

        def _reset_filters() -> None:
            st.session_state["region_filter"] = all_value
            st.session_state["state_filter"] = all_value
            st.session_state["driver_filter"] = all_value
            st.session_state["hub_filter"] = all_value
            st.session_state["contractor_filter"] = all_value
            st.session_state["delivery_filter_start"] = st.session_state["query_start_date"]
            st.session_state["delivery_filter_end"] = st.session_state["query_end_date"]
            st.session_state["applied_region_filter"] = all_value
            st.session_state["applied_state_filter"] = all_value
            st.session_state["applied_driver_filter"] = all_value
            st.session_state["applied_hub_filter"] = all_value
            st.session_state["applied_contractor_filter"] = all_value
            st.session_state["applied_delivery_filter_start"] = st.session_state["query_start_date"]
            st.session_state["applied_delivery_filter_end"] = st.session_state["query_end_date"]

        with filter_c6:
            st.write("")
            st.button(
                tr("reset_filters"),
                use_container_width=True,
                key="reset_filters_btn",
                on_click=_reset_filters,
            )

        delivery_filter_c1, delivery_filter_c2 = st.columns(2)
        with delivery_filter_c1:
            delivery_filter_start = st.date_input(
                tr("delivery_filter_start"),
                key="delivery_filter_start",
                min_value=date_input_min,
                max_value=date_input_max,
            )
        with delivery_filter_c2:
            delivery_filter_end = st.date_input(
                tr("delivery_filter_end"),
                key="delivery_filter_end",
                min_value=date_input_min,
                max_value=date_input_max,
            )

        apply_filter_clicked = st.button(tr("apply_filters"), type="primary", key="apply_filters_btn")
        if apply_filter_clicked:
            st.session_state["applied_region_filter"] = selected_region
            st.session_state["applied_state_filter"] = selected_state
            st.session_state["applied_driver_filter"] = selected_driver
            st.session_state["applied_hub_filter"] = selected_hub
            st.session_state["applied_contractor_filter"] = selected_contractor
            st.session_state["applied_delivery_filter_start"] = delivery_filter_start
            st.session_state["applied_delivery_filter_end"] = delivery_filter_end
            st.rerun()

        filtered_df = _apply_dimension_filters(
            result_df,
            selected_region=st.session_state.get("applied_region_filter", all_value),
            selected_state=st.session_state.get("applied_state_filter", all_value),
            selected_driver=st.session_state.get("applied_driver_filter", all_value),
            selected_hub=st.session_state.get("applied_hub_filter", all_value),
            selected_contractor=st.session_state.get("applied_contractor_filter", all_value),
        )

        applied_delivery_filter_start = st.session_state.get("applied_delivery_filter_start")
        applied_delivery_filter_end = st.session_state.get("applied_delivery_filter_end")

        if applied_delivery_filter_start and applied_delivery_filter_end and applied_delivery_filter_start <= applied_delivery_filter_end:
            delivery_filter_start_ts = pd.Timestamp(applied_delivery_filter_start)
            delivery_filter_end_ts = pd.Timestamp(applied_delivery_filter_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            ofd_date_series = to_datetime_series(filtered_df, "out_for_delivery_time")
            ofd_range_mask = (
                ofd_date_series.notna()
                & (ofd_date_series >= delivery_filter_start_ts)
                & (ofd_date_series <= delivery_filter_end_ts)
            )
            filtered_df = filtered_df[ofd_range_mask]
        elif applied_delivery_filter_start and applied_delivery_filter_end and applied_delivery_filter_start > applied_delivery_filter_end:
            st.warning(tr("delivery_filter_invalid"))

        if st.session_state.get("hide_unknown_dimensions", False):
            known_mask = (~filtered_df["Hub"].map(is_unknown_dimension_value)) & (~filtered_df["Contractor"].map(is_unknown_dimension_value))
            filtered_df = filtered_df[known_mask]

        layout_mode = st.radio(
            tr("layout_mode_label"),
            options=["detailed", "compact"],
            index=0,
            horizontal=True,
            format_func=lambda x: tr("layout_mode_detailed") if x == "detailed" else tr("layout_mode_compact"),
            key="kpi_layout_mode",
        )

        route_attempts_df, unresolved_attempts_df, canceled_attempts_df, lost_attempts_df = build_route_attempts_view(filtered_df)

        kpi_payload = render_kpi_charts(
            filtered_df,
            layout_mode=layout_mode,
            fetch_reference_time=st.session_state.get("fetch_clicked_at"),
            route_attempts_df=route_attempts_df,
        )

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
            st.dataframe(display_df, use_container_width=True)

        st.subheader(tr("customer_summary_section"))
        customer_summary_df = build_customer_address_summary(filtered_df)
        if customer_summary_df.empty:
            st.info(tr("customer_summary_empty"))
        else:
            st.dataframe(customer_summary_df, use_container_width=True)
            st.download_button(
                tr("download_customer_summary"),
                data=customer_summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"customer_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        st.subheader(tr("route_attempts_section"))
        if route_attempts_df.empty:
            st.info(tr("route_attempts_empty"))
        else:
            st.dataframe(route_attempts_df, use_container_width=True)

        multi_route_tracking_df = build_multi_route_tracking_view(route_attempts_df)
        st.subheader(tr("multi_route_tracking_section"))
        if multi_route_tracking_df.empty:
            st.info(tr("multi_route_tracking_empty"))
        else:
            st.dataframe(multi_route_tracking_df, use_container_width=True)

        st.subheader(tr("route_attempts_unresolved_section"))
        if unresolved_attempts_df.empty:
            st.info(tr("route_attempts_unresolved_empty"))
        else:
            st.dataframe(unresolved_attempts_df, use_container_width=True)

        st.subheader(tr("route_attempts_canceled_section"))
        if canceled_attempts_df.empty:
            st.info(tr("route_attempts_canceled_empty"))
        else:
            st.dataframe(canceled_attempts_df, use_container_width=True)

        st.subheader(tr("route_attempts_lost_section"))
        if lost_attempts_df.empty:
            st.info(tr("route_attempts_lost_empty"))
        else:
            st.dataframe(lost_attempts_df, use_container_width=True)

        invalid_route_df = build_invalid_route_summary(filtered_df)
        st.subheader(tr("invalid_route_section"))
        if invalid_route_df.empty:
            st.info(tr("invalid_route_empty"))
        else:
            st.dataframe(invalid_route_df, use_container_width=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_df = build_layout_specific_export_df(filtered_df, layout_mode)
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
        report_detail_df = build_detailed_report_detail_df(filtered_df) if layout_mode == "detailed" else export_df
        try:
            kpi_report_data = kpi_report_to_excel_bytes(
                report_payload,
                report_detail_df,
                layout_mode=layout_mode,
                source_df=filtered_df,
            )
            c_report.download_button(
                tr("download_report"),
                data=kpi_report_data,
                file_name=f"kpi_report_{layout_mode}_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            c_report.warning(tr("report_dep_missing"))


if __name__ == "__main__":
    main()
