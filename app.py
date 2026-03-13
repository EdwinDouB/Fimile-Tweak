from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from utils.utils import *
from utils.db import * 
from utils.routes import *
from utils.report import * 
from utils.constants import * 

import pandas as pd
import streamlit as st


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
    # scan time - created time and check how many hours 
    delivered_24h = next((m for m in kpi_payload["metrics"] if m.get("metric") == "<24h delivery rate"), None)
    pod_compliance_metric = next((m for m in kpi_payload["metrics"] if m.get("metric") == "Manual POD qualified rate"), None)
    attempt_24h_metric = next((m for m in kpi_payload["metrics"] if m.get("metric") == "24h attempt rate"), None)

    st.markdown(f"#### {tr('compact_title')}")
    c1, c2, c3 = st.columns(3)
    if delivered_24h:
        c1.metric("24h Delivery Rate", f"{delivered_24h['rate']:.2%}", f"{delivered_24h['hit']}/{delivered_24h['total']}")
        render_percentage_pie(
            title="24h Delivery Share",
            hit_count=int(delivered_24h["hit"]),
            total_count=int(delivered_24h["total"]),
            hit_label="<24h delivered",
            miss_label=">=24h or undelivered",
            chart_key="compact_delivered_24h",
            container=c1,
        )
    else:
        c1.metric("24h Delivery Rate", "0.00%", "0/0")
        c1.info("24h delivery share: no data available")

    if pod_compliance_metric:
        c2.metric("Manual POD Qualified Rate", f"{pod_compliance_metric['rate']:.2%}", f"{pod_compliance_metric['hit']}/{pod_compliance_metric['total']}")
        render_percentage_pie(
            title="Manual POD Review Share",
            hit_count=int(pod_compliance_metric["hit"]),
            total_count=int(pod_compliance_metric["total"]),
            hit_label="Qualified",
            miss_label="Not Qualified",
            chart_key="compact_pod_compliance",
            container=c2,
        )
    else:
        c2.metric("Manual POD Qualified Rate", "0.00%", "0/0")
        c2.info("Manual POD review share: no data available")

    if attempt_24h_metric:
        c3.metric("24h Attempt Rate", f"{attempt_24h_metric['rate']:.2%}", f"{attempt_24h_metric['hit']}/{attempt_24h_metric['total']}")
        render_percentage_pie(
            title="24h Attempt Share",
            hit_count=int(attempt_24h_metric["hit"]),
            total_count=int(attempt_24h_metric["total"]),
            hit_label="Attempted or delivered within 24h",
            miss_label="No attempt/delivery within 24h",
            chart_key="compact_attempt_24h",
            container=c3,
        )
    else:
        c3.metric("24h Attempt Rate", "0.00%", "0/0")
        c3.info("24h attempt share: no data available")

def render_daily_kpi_charts(result_df: pd.DataFrame) -> None:
    chart_df = result_df.copy()
    chart_df["_created_date"] = pd.to_datetime(chart_df["created_time"], errors="coerce").dt.date
    chart_df["_delivered_date"] = pd.to_datetime(chart_df["delivered_time"], errors="coerce").dt.date
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

def render_kpi_charts(result_df: pd.DataFrame, layout_mode: str, fetch_reference_time: datetime | None = None) -> dict[str, Any]:
    st.subheader(tr("kpi_title"))
    if result_df.empty:
        st.info(tr("kpi_empty"))
        return {"metrics": [], "charts": [], "has_monthly_lost_data": False, "monthly_lost": pd.DataFrame()}

    kpi_payload = build_kpi_report_payload(
        result_df,
        fetch_reference_time=fetch_reference_time,
    )
    refresh_key = str(int(fetch_reference_time.timestamp())) if fetch_reference_time else "no_fetch_ts"

    non_pickup_df, _ = split_pickup_routes(result_df)
    delivered_detail_df = non_pickup_df.loc[
        non_pickup_df["out_for_delivery_time"].notna() & non_pickup_df["out_for_delivery_time"].astype(str).str.strip().ne(""),
        [
            "tracking_id",
            "Region",
            "State",
            "shipperName",
            "Hub",
            "Contractor",
            "Route_name",
            "out_for_delivery_time",
            "delivered_time",
        ],
    ].copy()
    delivered_detail_df["ofd_dt"] = to_datetime_series(delivered_detail_df, "out_for_delivery_time")
    delivered_detail_df["delivered_dt"] = to_datetime_series(delivered_detail_df, "delivered_time")
    delivered_detail_df["ofd_to_delivered_hours"] = (
        delivered_detail_df["delivered_dt"] - delivered_detail_df["ofd_dt"]
    ).dt.total_seconds() / 3600
    for threshold in [24, 48, 72]:
        delivered_detail_df[f"within_{threshold}h"] = (
            delivered_detail_df["delivered_dt"].notna()
            & (delivered_detail_df["ofd_to_delivered_hours"] >= 0)
            & (delivered_detail_df["ofd_to_delivered_hours"] < threshold)
        )

    if layout_mode == "compact":
        render_compact_kpi_row(kpi_payload)
        selected_eval_weight = calculate_package_evaluation_weight(result_df).sum()
        st.metric(tr("compact_eval_weight"), f"{selected_eval_weight:.2f}")
        st.markdown("##### 24h Delivery Rate Details")
        compact_breakdown_df = build_delivery_breakdown_table(delivered_detail_df, thresholds=[24])
        st.dataframe(style_breakdown_rows(compact_breakdown_df), use_container_width=True)
        return kpi_payload

    render_daily_kpi_charts(result_df)

    st.markdown("#### 24/48/72h Delivery Rate (Scan -> Delivered)")
    detailed_breakdown_df = build_delivery_breakdown_table(delivered_detail_df, thresholds=[24, 48, 72])
    st.dataframe(style_breakdown_rows(detailed_breakdown_df), use_container_width=True)
    delivered_detail_df = delivered_detail_df.drop(columns=["ofd_dt", "delivered_dt"])

    delivered_header_cols = st.columns([4, 1])
    delivered_header_cols[1].download_button(
        tr("download_delivered"),
        data=delivered_detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"delivered_rate_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=delivered_detail_df.empty,
    )

    delivered_cols = st.columns(3)
    delivered_metrics = [m for m in kpi_payload["metrics"] if m.get("category") == "delivery_rate_24_48_72"]
    for i, metric in enumerate(delivered_metrics):
        threshold = metric["metric"].replace("<", "").replace("h delivery rate", "")
        delivered_cols[i].metric(
            metric["metric"],
            f"{metric['rate']:.2%}",
            f"{metric['hit']}/{metric['total']}",
        )
        render_percentage_pie(
            title=f"<{threshold}h delivery share",
            hit_count=int(metric["hit"]),
            total_count=int(metric["total"]),
            hit_label=f"<{threshold}h delivered",
            miss_label=f">={threshold}h or undelivered",
            chart_key=f"delivered_{threshold}_{refresh_key}",
            container=delivered_cols[i],
        )

    st.markdown("#### 12/24/48/72h Scan Rate (Pickup -> Scan)")
    scan_detail_df = result_df[
        [
            "tracking_id",
            "Region",
            "State",
            "shipperName",
            "created_time",
            "first_scanned_time",
        ]
    ].copy()
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
    scan_metrics = [m for m in kpi_payload["metrics"] if m.get("category") == "scan_rate_12_24_48_72"]
    for i, metric in enumerate(scan_metrics):
        threshold = metric["metric"].replace("<", "").replace("h scan rate", "")
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

    lost_detail_df = result_df.loc[
        lost_condition,
        [
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
        ],
    ].copy()

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

    compact_metric_names = {"<24h delivery rate", "Manual POD qualified rate", "24h attempt rate"}
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

    non_pickup_df, _ = split_pickup_routes(filtered_df)
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
    non_pickup_df, _ = split_pickup_routes(filtered_df)
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
            if not isinstance(payload, dict):
                return tracking_id, empty_row(tracking_id), {"tracking_id": tracking_id, "reason": "router_messages is not valid JSON object"}
            row = build_row(tracking_id, payload)
            return tracking_id, row, None
        except Exception as e:  # noqa: BLE001
            return tracking_id, empty_row(tracking_id), {"tracking_id": tracking_id, "reason": str(e)}

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
            progress_bar.progress(completed / total)
            status_text.text(tr("processing", completed=completed, total=total, tracking_id=tracking_id))

    ordered_rows = [rows_by_id[tid] for tid in dedup_ids]
    return pd.DataFrame(ordered_rows, columns=result_columns), failures

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
    if "region_filter" not in st.session_state:
        st.session_state["region_filter"] = "ALL"
    if "state_filter" not in st.session_state:
        st.session_state["state_filter"] = "ALL"
    if "driver_filter" not in st.session_state:
        st.session_state["driver_filter"] = "ALL"
    if "hub_filter" not in st.session_state:
        st.session_state["hub_filter"] = "ALL"
    if "contractor_filter" not in st.session_state:
        st.session_state["contractor_filter"] = "ALL"
    if "fetch_clicked_at" not in st.session_state:
        st.session_state["fetch_clicked_at"] = None
    if "language" not in st.session_state:
        st.session_state["language"] = "zh"
    if "hide_unknown_dimensions" not in st.session_state:
        st.session_state["hide_unknown_dimensions"] = False
    if "date_filter_type" not in st.session_state:
        st.session_state["date_filter_type"] = "created"
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

    date_filter_options = {
        tr("date_filter_created"): "created",
        tr("date_filter_delivery"): "delivery",
    }
    selected_date_filter = st.selectbox(
        tr("date_filter_type"),
        options=list(date_filter_options.keys()),
    )
    st.session_state["date_filter_type"] = date_filter_options[selected_date_filter]

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input(tr("start_date"), value=date.today() - timedelta(days=1))
    with c2:
        end_d = st.date_input(tr("end_date"), value=date.today())

    raw_ids: list[str] = []
    btn = st.button(tr("load_btn"), type="primary")
    if btn:
        with st.spinner(tr("loading_db")):
            try:
                if st.session_state.get("date_filter_type") == "delivery":
                    raw_ids = fetch_tracking_numbers_by_delivery_window(start_d, end_d)
                else:
                    raw_ids = fetch_tracking_numbers_by_date(start_d, end_d)
                st.session_state["db_raw_ids"] = raw_ids
                if not raw_ids:
                    st.warning(tr("no_tracking_found"))
            except Exception as e:
                st.error(str(e))
                raw_ids = []
                st.session_state["db_raw_ids"] = []

    if not btn:
        raw_ids = st.session_state.get("db_raw_ids", [])

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

    if st.button(tr("fetch_btn"), type="primary", disabled=not dedup_ids):
        st.session_state["fetch_clicked_at"] = datetime.now()
        failures: list[dict[str, str]] = []
        receive_province_map: dict[str, str] = {}
        sender_info_map: dict[str, dict[str, str]] = {}
        router_messages_map: dict[str, Any] = {}

        try:
            receive_province_map = fetch_receive_province_map(tuple(dedup_ids))
        except Exception as e:
            st.warning(tr("state_region_fail", error=e))

        try:
            sender_info_map = fetch_sender_info_map(tuple(dedup_ids))
        except Exception:
            sender_info_map = {}

        try:
            router_messages_map = fetch_router_messages_map(tuple(dedup_ids))
        except Exception as e:
            st.warning(f"Failed to load router_messages from DB: {e}")
            router_messages_map = {}

        progress = st.progress(0)
        status = st.empty()

        result_df, failures = process_tracking_ids(
            dedup_ids=dedup_ids,
            receive_province_map=receive_province_map,
            sender_info_map=sender_info_map,
            router_messages_map=router_messages_map,
            progress_bar=progress,
            status_text=status,
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
        if st.button(toggle_label):
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
            if st.button(tr("override_apply_btn"), use_container_width=True):
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

        region_series = result_df["Region"].fillna("").astype(str).str.strip()
        region_options = [tr("all")] + sorted([item for item in region_series.unique().tolist() if item])
        if st.session_state["region_filter"] not in region_options:
            st.session_state["region_filter"] = tr("all")

        driver_series = result_df["Driver"].fillna("").astype(str).str.strip()
        driver_options = [tr("all")] + sorted([item for item in driver_series.unique().tolist() if item])
        if st.session_state["driver_filter"] not in driver_options:
            st.session_state["driver_filter"] = tr("all")

        hub_series = result_df["Hub"].fillna("").astype(str).str.strip()
        hub_options = [tr("all")] + sorted([item for item in hub_series.unique().tolist() if item])
        if st.session_state["hub_filter"] not in hub_options:
            st.session_state["hub_filter"] = tr("all")

        contractor_series = result_df["Contractor"].fillna("").astype(str).str.strip()
        contractor_options = [tr("all")] + sorted([item for item in contractor_series.unique().tolist() if item])
        if st.session_state["contractor_filter"] not in contractor_options:
            st.session_state["contractor_filter"] = tr("all")

        filter_c1, filter_c2, filter_c3, filter_c4, filter_c5 = st.columns(5)
        with filter_c1:
            selected_region = st.selectbox("Region", options=region_options, key="region_filter")

        if selected_region == tr("all"):
            available_states_df = result_df
        else:
            available_states_df = result_df[result_df["Region"].fillna("").astype(str).str.strip() == selected_region]

        state_series = available_states_df["State"].fillna("").astype(str).str.strip()
        state_options = [tr("all")] + sorted([item for item in state_series.unique().tolist() if item])
        if st.session_state["state_filter"] not in state_options:
            st.session_state["state_filter"] = tr("all")

        with filter_c2:
            selected_state = st.selectbox("State", options=state_options, key="state_filter")
        with filter_c3:
            selected_driver = st.selectbox("Driver", options=driver_options, key="driver_filter")
        with filter_c4:
            selected_hub = st.selectbox("Hub", options=hub_options, key="hub_filter")
        with filter_c5:
            selected_contractor = st.selectbox("Contractor", options=contractor_options, key="contractor_filter")

        filtered_df = result_df.copy()
        if selected_region != tr("all"):
            filtered_df = filtered_df[
                filtered_df["Region"].fillna("").astype(str).str.strip() == selected_region
            ]
        if selected_state != tr("all"):
            filtered_df = filtered_df[
                filtered_df["State"].fillna("").astype(str).str.strip() == selected_state
            ]
        if selected_driver != tr("all"):
            filtered_df = filtered_df[
                filtered_df["Driver"].fillna("").astype(str).str.strip() == selected_driver
            ]
        if selected_hub != tr("all"):
            filtered_df = filtered_df[
                filtered_df["Hub"].fillna("").astype(str).str.strip() == selected_hub
            ]
        if selected_contractor != tr("all"):
            filtered_df = filtered_df[
                filtered_df["Contractor"].fillna("").astype(str).str.strip() == selected_contractor
            ]

        if st.session_state.get("hide_unknown_dimensions", False):
            known_mask = (~filtered_df["Hub"].map(is_unknown_dimension_value)) & (~filtered_df["Contractor"].map(is_unknown_dimension_value))
            filtered_df = filtered_df[known_mask]


        if st.session_state.get("date_filter_type") == "delivery":
            ofd_dt = pd.to_datetime(filtered_df["out_for_delivery_time"], errors="coerce")
            start_ts = pd.Timestamp(start_d)
            end_ts = pd.Timestamp(end_d) + pd.Timedelta(days=1)
            filtered_df = filtered_df[
                ofd_dt.notna()
                & (ofd_dt >= start_ts)
                & (ofd_dt < end_ts)
            ]

        non_pickup_filtered_df, _ = split_pickup_routes(filtered_df)
        filtered_df = non_pickup_filtered_df.copy()

        layout_mode = st.radio(
            tr("layout_mode_label"),
            options=["detailed", "compact"],
            index=0,
            horizontal=True,
            format_func=lambda x: tr("layout_mode_detailed") if x == "detailed" else tr("layout_mode_compact"),
            key="kpi_layout_mode",
        )

        kpi_payload = render_kpi_charts(filtered_df, layout_mode=layout_mode, fetch_reference_time=st.session_state.get("fetch_clicked_at"))

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
        preview_df = build_export_df(filtered_df)
        st.dataframe(preview_df.head(50), use_container_width=True)

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

