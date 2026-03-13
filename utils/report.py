from utils.utils import to_datetime_series, rate
from datetime import datetime, timezone
from utils.routes import split_pickup_routes, build_lost_package_analysis
from typing import Any
import pandas as pd
import io 
from xlsxwriter.utility import xl_col_to_name


def _resolve_ofd_column(df: pd.DataFrame) -> str:
    if "first_out_for_delivery_date" in df.columns:
        return "first_out_for_delivery_date"
    return "out_for_delivery_time"


def _yes_no_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(["" for _ in range(len(df))], index=df.index, dtype="object")
    return df[col].fillna("").astype(str).str.strip().str.lower()

def _build_detailed_overview_table(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=["Dimension", "Sample Count", "<24h Hit", "<24h Delivery Rate", "<48h Hit", "<48h Delivery Rate", "<72h Hit", "<72h Delivery Rate"])

    source_df = detail_df.copy()
    if "out_for_delivery_time" in source_df.columns:
        source_df["ofd_dt"] = to_datetime_series(source_df, _resolve_ofd_column(source_df))
    else:
        source_df["ofd_dt"] = pd.NaT
    if "delivered_time" in source_df.columns:
        source_df["delivered_dt"] = to_datetime_series(source_df, "delivered_time")
    else:
        source_df["delivered_dt"] = pd.NaT

    source_df["ofd_to_delivered_hours"] = (source_df["delivered_dt"] - source_df["ofd_dt"]).dt.total_seconds() / 3600
    for threshold in [24, 48, 72]:
        source_df[f"within_{threshold}h"] = (
            source_df["delivered_dt"].notna()
            & (source_df["ofd_to_delivered_hours"] >= 0)
            & (source_df["ofd_to_delivered_hours"] < threshold)
        )

    def _append_row(rows: list[dict[str, Any]], dimension: str, sub_df: pd.DataFrame) -> None:
        total_count = len(sub_df)
        row = {"Dimension": dimension, "Sample Count": total_count}
        for threshold in [24, 48, 72]:
            hit = int(sub_df[f"within_{threshold}h"].sum()) if total_count > 0 else 0
            row[f"<{threshold}h Hit"] = hit
            row[f"<{threshold}h Delivery Rate"] = rate(hit, total_count)
        rows.append(row)

    rows: list[dict[str, Any]] = []
    _append_row(rows, "Overall", source_df)

    if "Region" in source_df.columns:
        for region in sorted(source_df["Region"].fillna("Unknown Region").astype(str).str.strip().replace("", "Unknown Region").unique()):
            region_df = source_df[source_df["Region"].fillna("Unknown Region").astype(str).str.strip().replace("", "Unknown Region") == region]
            _append_row(rows, region, region_df)

            if "Hub" in source_df.columns:
                hub_series = region_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
                for hub in sorted(hub_series.unique()):
                    hub_df = region_df[hub_series == hub]
                    _append_row(rows, f"  {hub}", hub_df)

    return pd.DataFrame(rows)


def _build_hub_table(detail_df: pd.DataFrame, hub_name: str) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    hub_df = detail_df[detail_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub") == hub_name].copy()
    if hub_df.empty:
        return pd.DataFrame()

    hub_df["ofd_dt"] = to_datetime_series(hub_df, _resolve_ofd_column(hub_df))
    hub_df["delivered_dt"] = to_datetime_series(hub_df, "delivered_time")
    hub_df["ofd_to_delivered_hours"] = (hub_df["delivered_dt"] - hub_df["ofd_dt"]).dt.total_seconds() / 3600
    for threshold in [24, 48, 72]:
        hub_df[f"within_{threshold}h"] = (
            hub_df["delivered_dt"].notna()
            & (hub_df["ofd_to_delivered_hours"] >= 0)
            & (hub_df["ofd_to_delivered_hours"] < threshold)
        )

    rows: list[dict[str, Any]] = []

    def _append_row(dimension: str, sub_df: pd.DataFrame) -> None:
        total_count = len(sub_df)
        row = {"Dimension": dimension, "Sample Count": total_count}
        for threshold in [24, 48, 72]:
            hit = int(sub_df[f"within_{threshold}h"].sum()) if total_count > 0 else 0
            row[f"<{threshold}h Hit"] = hit
            row[f"<{threshold}h Delivery Rate"] = rate(hit, total_count)
        rows.append(row)

    _append_row(hub_name, hub_df)
    contractor_series = hub_df["Contractor"].fillna("Unknown Contractor").astype(str).str.strip().replace("", "Unknown Contractor")
    for contractor in sorted(contractor_series.unique()):
        _append_row(f"  {contractor}", hub_df[contractor_series == contractor])

    return pd.DataFrame(rows)


def _style_overview_worksheet(worksheet, table_df: pd.DataFrame, start_row: int, workbook) -> None:
    if table_df.empty:
        return

    header_fmt = workbook.add_format({"bold": True, "bg_color": "#1f4e78", "font_color": "#ffffff", "border": 1})
    level_colors = {0: "#f3f4f6", 1: "#dbeafe", 2: "#e0f2fe", 3: "#ecfccb"}
    for col_idx, col_name in enumerate(table_df.columns):
        worksheet.write(start_row, col_idx, col_name, header_fmt)

    for ridx, (_, row) in enumerate(table_df.iterrows(), start=1):
        dimension = str(row.get("Dimension", ""))
        leading_spaces = len(dimension) - len(dimension.lstrip(" "))
        level = min(3, leading_spaces // 2)
        bg = level_colors.get(level, "#ffffff")
        row_fmt = workbook.add_format({"bg_color": bg, "border": 1})
        row_rate_fmt = workbook.add_format({"bg_color": bg, "border": 1, "num_format": "0.00%"})
        for cidx, val in enumerate(row.tolist()):
            col_name = table_df.columns[cidx]
            if "Rate" in col_name:
                worksheet.write(start_row + ridx, cidx, val, row_rate_fmt)
            elif isinstance(val, (int, float)):
                worksheet.write(start_row + ridx, cidx, val, row_fmt)
            else:
                worksheet.write(start_row + ridx, cidx, str(val), row_fmt)

    worksheet.set_column(0, 0, 28)
    worksheet.set_column(1, 7, 18)

def _sanitize_sheet_name(name: str) -> str:
    invalid_chars = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in str(name))
    return cleaned[:31]


def _insert_dashboard_charts(
    worksheet,
    workbook,
    chart_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    chart_row: int,
    chart_col: int,
    data_col: int,
) -> None:
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#e5e7eb", "border": 1})
    data_fmt = workbook.add_format({"border": 1})
    percent_fmt = workbook.add_format({"border": 1, "num_format": "0.00%"})

    sheet_name = worksheet.get_name()


    write_row = 0
    def _write_data_row(chart_name: str, category: str, count: int, value_rate: float) -> int:
        nonlocal write_row
        target_row = write_row
        worksheet.write(target_row, data_col, chart_name, data_fmt)
        worksheet.write(target_row, data_col + 1, category, data_fmt)
        worksheet.write(target_row, data_col + 2, count, data_fmt)
        worksheet.write(target_row, data_col + 3, value_rate, percent_fmt)
        write_row += 1
        return target_row

    for offset, title in enumerate(["chart", "category", "count", "rate"]):
        worksheet.write(0, data_col + offset, title, header_fmt)
    write_row = 1

    chart_positions = {
        "<24h delivery rate": (chart_row, chart_col),
        "<48h delivery rate": (chart_row, chart_col + 8),
        "<72h delivery rate": (chart_row, chart_col + 16),
        "12/24/48/72 scan rate": (chart_row + 16, chart_col),
        "Manual POD qualified rate": (chart_row + 32, chart_col),
        "24h attempt rate": (chart_row + 32, chart_col + 8),
        "lost rate": (chart_row + 32, chart_col + 16),
    }

    for chart_name in ["<24h delivery rate", "<48h delivery rate", "<72h delivery rate", "Manual POD qualified rate", "24h attempt rate", "lost rate"]:
        group = chart_df[chart_df["chart"] == chart_name]
        if group.empty:
            continue
        row_ids = []
        for _, rec in group.iterrows():
            row_ids.append(_write_data_row(chart_name, str(rec.get("category", "")), int(rec.get("count", 0)), float(rec.get("rate", 0))))

        pie = workbook.add_chart({"type": "pie"})
        pie.add_series(
            {
                "name": chart_name,
                "categories": [sheet_name, row_ids[0], data_col + 1, row_ids[-1], data_col + 1],
                "values": [sheet_name, row_ids[0], data_col + 2, row_ids[-1], data_col + 2],
                "data_labels": {"percentage": True, "category": True},
            }
        )
        pie.set_title({"name": chart_name})
        pie.set_style(10)
        worksheet.insert_chart(*chart_positions[chart_name], pie, {"x_scale": 1.0, "y_scale": 1.0})

    scan_metrics = metrics_df[metrics_df["category"] == "scan_rate_12_24_48_72"].copy()
    if not scan_metrics.empty:
        scan_row_start = write_row
        ordered_scan = ["<12h scan rate", "<24h scan rate", "<48h scan rate", "<72h scan rate"]
        for metric_name in ordered_scan:
            rec = scan_metrics[scan_metrics["metric"] == metric_name]
            if rec.empty:
                continue
            value = float(rec.iloc[0].get("rate", 0))
            _write_data_row("12/24/48/72 scan rate", metric_name, 0, value)
        scan_row_end = write_row - 1
        if scan_row_end >= scan_row_start:
            col = workbook.add_chart({"type": "column"})
            col.add_series(
                {
                    "name": "12/24/48/72 scan rate",
                    "categories": [sheet_name, scan_row_start, data_col + 1, scan_row_end, data_col + 1],
                    "values": [sheet_name, scan_row_start, data_col + 3, scan_row_end, data_col + 3],
                    "data_labels": {"value": True, "num_format": "0.00%"},
                }
            )
            col.set_title({"name": "12/24/48/72 scan rate"})
            col.set_y_axis({"num_format": "0%", "min": 0, "max": 1})
            worksheet.insert_chart(*chart_positions["12/24/48/72 scan rate"], col, {"x_scale": 1.0, "y_scale": 1.0})
    # Keep chart source data visible. Some spreadsheet viewers (e.g. WPS)
    # don't plot charts when the source columns are hidden, which makes the
    # exported charts appear blank.
    worksheet.set_column(data_col, data_col + 3, 14)



def build_kpi_report_payload(
    result_df: pd.DataFrame,
    fetch_reference_time: datetime | None = None,
) -> dict[str, Any]:
    df = result_df.copy()
    ofd_col = _resolve_ofd_column(df)
    df["created_dt"] = to_datetime_series(df, "created_time")
    df["first_scanned_dt"] = to_datetime_series(df, "first_scanned_time")
    df["last_scanned_dt"] = to_datetime_series(df, "last_scanned_time")
    df["ofd_dt"] = to_datetime_series(df, ofd_col)
    df["attempted_dt"] = to_datetime_series(df, "attempted_time")
    df["delivered_dt"] = to_datetime_series(df, "delivered_time")
    df["month"] = df["created_dt"].dt.to_period("M").astype(str)
    df.loc[df["month"] == "NaT", "month"] = "Unknown"
    non_pickup_df, _ = split_pickup_routes(df)

    metrics: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []

    non_pickup_df["ofd_to_delivered_hours"] = (non_pickup_df["delivered_dt"] - non_pickup_df["ofd_dt"]).dt.total_seconds() / 3600
    ofd_series = non_pickup_df.get(ofd_col, pd.Series("", index=non_pickup_df.index))
    ofd_present_mask = ofd_series.notna() & ofd_series.astype(str).str.strip().ne("")
    ofd_base = non_pickup_df[ofd_present_mask].copy()

    delivered_within_24h = ofd_base[
        ofd_base["delivered_dt"].notna() & (ofd_base["ofd_to_delivered_hours"] >= 0) & (ofd_base["ofd_to_delivered_hours"] < 24)
    ]

    
    for threshold in [24, 48, 72]:
        within = ofd_base[
            ofd_base["delivered_dt"].notna() & (ofd_base["ofd_to_delivered_hours"] >= 0) & (ofd_base["ofd_to_delivered_hours"] < threshold)
        ]
        metric_name = f"<{threshold}h delivery rate"
        hit_count = len(within)
        total_count = len(ofd_base)
        miss_count = max(total_count - hit_count, 0)
        metrics.append(
            {
                "category": "delivery_rate_24_48_72",
                "metric": metric_name,
                "hit": hit_count,
                "total": total_count,
                "rate": rate(hit_count, total_count),
            }
        )
        chart_rows.extend(
            [
                {"chart": metric_name, "category": f"<{threshold}h delivered", "count": hit_count, "rate": rate(hit_count, total_count)},
                {"chart": metric_name, "category": f">={threshold}h or undelivered", "count": miss_count, "rate": rate(miss_count, total_count)},
            ]
        )

    df["created_to_scan_hours"] = (df["first_scanned_dt"] - df["created_dt"]).dt.total_seconds() / 3600
    total_count = len(df)
    for threshold in [12, 24, 48, 72]:
        within = df[
            df["first_scanned_dt"].notna() & (df["created_to_scan_hours"] >= 0) & (df["created_to_scan_hours"] < threshold)
        ]
        metric_name = f"<{threshold}h scan rate"
        hit_count = len(within)
        miss_count = max(total_count - hit_count, 0)
        metrics.append(
            {
                "category": "scan_rate_12_24_48_72",
                "metric": metric_name,
                "hit": hit_count,
                "total": total_count,
                "rate": rate(hit_count, total_count),
            }
        )
        chart_rows.extend(
            [
                {"chart": metric_name, "category": f"<{threshold}h scanned", "count": hit_count, "rate": rate(hit_count, total_count)},
                {"chart": metric_name, "category": f">={threshold}h or unscanned", "count": miss_count, "rate": rate(miss_count, total_count)},
            ]
        )

    attempt_base = non_pickup_df[ofd_present_mask].copy()
    attempt_base["ofd_to_attempted_hours"] = (attempt_base["attempted_dt"] - attempt_base["ofd_dt"]).dt.total_seconds() / 3600
    attempt_base["ofd_to_delivered_hours"] = (attempt_base["delivered_dt"] - attempt_base["ofd_dt"]).dt.total_seconds() / 3600
    failed_without_delivery_mask = attempt_base["attempted_dt"].notna() & attempt_base["delivered_dt"].isna()
    failed_attempt_within_24h_mask = failed_without_delivery_mask & (attempt_base["ofd_to_attempted_hours"] >= 0) & (attempt_base["ofd_to_attempted_hours"] < 24)
    delivered_within_24h_mask = (
        (attempt_base["delivered_dt"].notna())
        & (attempt_base["ofd_to_delivered_hours"] >= 0)
        & (attempt_base["ofd_to_delivered_hours"] < 24)
    )

    review_base = attempt_base.copy()
    review_base["review_date"] = review_base["ofd_dt"].dt.date.astype(str).replace("NaT", "")
    review_base["driver_name"] = review_base.get("Driver", "")
    review_base["dsp"] = review_base.get("Contractor", "")
    review_base["route_code"] = review_base.get("Route_name", "")
    review_base["delivery_status"] = review_base["delivered_dt"].notna().map(lambda x: "Delivered" if x else "Not Delivered")
    review_base["stop_status"] = "No Attempt"
    review_base.loc[failed_without_delivery_mask, "stop_status"] = "Failed"
    review_base.loc[review_base["delivered_dt"].notna(), "stop_status"] = "Delivered"
    review_base["event_code"] = "NO_ATTEMPT"
    review_base.loc[failed_without_delivery_mask, "event_code"] = "FAILED_STOP"
    review_base.loc[review_base["delivered_dt"].notna(), "event_code"] = "DELIVERED_STOP"
    review_base["event_readable"] = review_base["event_code"].map(
        {
            "DELIVERED_STOP": "Delivered stop",
            "FAILED_STOP": "Failed stop",
            "NO_ATTEMPT": "No attempt/delivery event",
        }
    )
    review_base["beans_link"] = review_base["tracking_id"].astype(str).map(
        lambda tid: f'=HYPERLINK("https://www.beansroute.ai/3pl-manager/tabs.html#searchTrackingId/{tid}", "Open Beans POD")'
    )
    review_base["manual_pod_review_status"] = "Pending"
    review_base["manual_review_note"] = ""
    review_base["attempt_validated"] = False
    review_base.loc[delivered_within_24h_mask, "attempt_validated"] = True
    review_base.loc[failed_attempt_within_24h_mask, "attempt_validated"] = True

    pod_review_export_columns = [
        "tracking_id",
        "route_code",
        "Route_name",
        "review_date",
        "dsp",
        "driver_name",
        "Region",
        "State",
        "Hub",
        "Contractor",
        "stop_status",
        "delivery_status",
        "event_code",
        "event_readable",
        ofd_col,
        "attempted_time",
        "delivered_time",
        "beans_link",
        "manual_pod_review_status",
        "manual_review_note",
        "attempt_validated",
    ]
    existing_pod_columns = [col for col in pod_review_export_columns if col in review_base.columns]
    pod_review_export_df = (
        review_base[existing_pod_columns].copy() if existing_pod_columns else pd.DataFrame(columns=pod_review_export_columns)
    )
    if ofd_col in pod_review_export_df.columns and ofd_col != "out_for_delivery_time":
        pod_review_export_df = pod_review_export_df.rename(columns={ofd_col: "out_for_delivery_time"})

    delivered_for_pod_review = review_base[review_base["stop_status"] == "Delivered"]
    pending_pod_count = int(
        (
            _yes_no_series(delivered_for_pod_review, "first_pod_complience").eq("")
            & _yes_no_series(delivered_for_pod_review, "second_pod_complience").eq("")
            & _yes_no_series(delivered_for_pod_review, "third_pod_complience").eq("")
        ).sum()
    )

    pod_yes_count = int(
        (_yes_no_series(review_base, "first_pod_complience") == "yes").sum()
        + (_yes_no_series(review_base, "second_pod_complience") == "yes").sum()
        + (_yes_no_series(review_base, "third_pod_complience") == "yes").sum()
    )
    pod_no_count = int(
        (_yes_no_series(review_base, "first_pod_complience") == "no").sum()
        + (_yes_no_series(review_base, "second_pod_complience") == "no").sum()
        + (_yes_no_series(review_base, "third_pod_complience") == "no").sum()
    )
    reviewed_delivered_count = pod_yes_count + pod_no_count
    pod_qualified_count = pod_yes_count
    pod_not_qualified_count = pod_no_count
    metrics.append(
        {
            "category": "pod_manual_review",
            "metric": "Manual POD qualified rate",
            "hit": pod_qualified_count,
            "total": reviewed_delivered_count,
            "rate": rate(pod_qualified_count, reviewed_delivered_count),
        }
    )
    metrics.append(
        {
            "category": "pod_manual_review",
            "metric": "pending_manual_pod_review_count",
            "hit": pending_pod_count,
            "total": len(delivered_for_pod_review),
            "rate": rate(pending_pod_count, len(delivered_for_pod_review)),
        }
    )
    chart_rows.extend(
        [
            {
                "chart": "Manual POD qualified rate",
                "category": "Qualified",
                "count": pod_qualified_count,
                "rate": rate(pod_qualified_count, reviewed_delivered_count),
            },
            {
                "chart": "Manual POD qualified rate",
                "category": "Not Qualified",
                "count": pod_not_qualified_count,
                "rate": rate(pod_not_qualified_count, reviewed_delivered_count),
            },
        ]
    )

    failed_without_delivery_count = int(failed_without_delivery_mask.sum())
    attempt_excluded_by_manual_review = int((failed_attempt_within_24h_mask & ~review_base["attempt_validated"]).sum())
    first_yes = _yes_no_series(review_base, "first_pod_complience") == "yes"
    second_yes = _yes_no_series(review_base, "second_pod_complience") == "yes"
    third_yes = _yes_no_series(review_base, "third_pod_complience") == "yes"
    attempt_hit_mask = first_yes | second_yes | third_yes
    attempt_total_count = len(attempt_base)
    attempt_hit_count = int(attempt_hit_mask.sum())
    attempt_miss_count = max(attempt_total_count - attempt_hit_count, 0)
    metrics.append(
        {
            "category": "dsp_assessment",
            "metric": "24h attempt rate",
            "hit": attempt_hit_count,
            "total": attempt_total_count,
            "rate": rate(attempt_hit_count, attempt_total_count),
        }
    )
    chart_rows.extend(
        [
            {
                "chart": "24h attempt rate",
                "category": "Attempted or delivered within 24h",
                "count": attempt_hit_count,
                "rate": rate(attempt_hit_count, attempt_total_count),
            },
            {
                "chart": "24h attempt rate",
                "category": "No attempt/delivery within 24h",
                "count": attempt_miss_count,
                "rate": rate(attempt_miss_count, attempt_total_count),
            },
        ]
    )
    metrics.extend(
        [
            {
                "category": "pod_review",
                "metric": "failed_without_delivery_count",
                "hit": failed_without_delivery_count,
                "total": attempt_total_count,
                "rate": rate(failed_without_delivery_count, attempt_total_count),
            },
            {
                "category": "pod_review",
                "metric": "attempt_excluded_by_manual_review",
                "hit": attempt_excluded_by_manual_review,
                "total": attempt_total_count,
                "rate": rate(attempt_excluded_by_manual_review, attempt_total_count),
            },
        ]
    )

    
    lost_analysis = build_lost_package_analysis(df, fetch_reference_time=fetch_reference_time)
    scanned_base = lost_analysis["scanned_base"]
    scanned_base["lost"] = lost_analysis["lost_mask"].loc[scanned_base.index].astype(int)
    monthly_lost = scanned_base.groupby("month", as_index=False).agg(total=("tracking_id", "count"), lost=("lost", "sum"))
    lost_total = int(monthly_lost["lost"].sum()) if not monthly_lost.empty else 0
    scanned_total = int(monthly_lost["total"].sum()) if not monthly_lost.empty else 0
    metrics.append(
        {
            "category": "monthly_lost_rate_last_scan_120h",
            "metric": "lost rate",
            "hit": lost_total,
            "total": scanned_total,
            "rate": rate(lost_total, scanned_total),
        }
    )
    chart_rows.extend(
        [
            {"chart": "lost rate", "category": "Lost", "count": lost_total, "rate": rate(lost_total, scanned_total)},
            {
                "chart": "lost rate",
                "category": "Not lost",
                "count": max(scanned_total - lost_total, 0),
                "rate": rate(max(scanned_total - lost_total, 0), scanned_total),
            },
        ]
    )

    return {
        "metrics": metrics,
        "charts": chart_rows,
        "has_monthly_lost_data": not monthly_lost.empty,
        "monthly_lost": monthly_lost,
        "pod_review_df": pod_review_export_df,
    }

def kpi_report_to_excel_bytes(
    kpi_payload: dict[str, Any],
    detail_df: pd.DataFrame | None = None,
    layout_mode: str = "detailed",
    source_df: pd.DataFrame | None = None,
) -> bytes:
    output = io.BytesIO()
    metrics_df = pd.DataFrame(kpi_payload["metrics"])
    chart_df = pd.DataFrame(kpi_payload["charts"])

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        if layout_mode == "detailed" and detail_df is not None and not detail_df.empty:
            overview_ws = workbook.add_worksheet("overview")
            overview_table = _build_detailed_overview_table(detail_df)
            _style_overview_worksheet(overview_ws, overview_table, 0, workbook)
            _insert_dashboard_charts(
                overview_ws,
                workbook,
                chart_df,
                metrics_df,
                chart_row=0,
                chart_col=max(len(overview_table.columns) + 2, 10),
                data_col=max(len(overview_table.columns) + 28, 36),
            )

            hub_series = detail_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
            hub_source = source_df if source_df is not None and not source_df.empty else detail_df
            hub_source_series = hub_source["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
            for hub_name in sorted(hub_series.unique()):
                hub_table = _build_hub_table(detail_df, hub_name)
                if hub_table.empty:
                    continue
                hub_df = hub_source[hub_source_series == hub_name].copy()
                if hub_df.empty and "tracking_id" in hub_source.columns and "tracking_id" in detail_df.columns:
                    hub_tracking_ids = set(
                        detail_df[
                            detail_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
                            == hub_name
                        ]["tracking_id"].astype(str)
                    )
                    if hub_tracking_ids:
                        hub_df = hub_source[hub_source["tracking_id"].astype(str).isin(hub_tracking_ids)].copy()
                hub_payload = build_kpi_report_payload(hub_df)
                hub_chart_df = pd.DataFrame(hub_payload["charts"])
                hub_metrics_df = pd.DataFrame(hub_payload["metrics"])
                sheet_name = _sanitize_sheet_name(f"HUB_{hub_name}")
                hub_ws = workbook.add_worksheet(sheet_name)
                _style_overview_worksheet(hub_ws, hub_table, 0, workbook)
                _insert_dashboard_charts(
                    hub_ws,
                    workbook,
                    hub_chart_df,
                    hub_metrics_df,
                    chart_row=0,
                    chart_col=max(len(hub_table.columns) + 2, 10),
                    data_col=max(len(hub_table.columns) + 28, 36),
                )

        metrics_df.to_excel(writer, index=False, sheet_name="kpi_summary")
        chart_df.to_excel(writer, index=False, sheet_name="kpi_chart_data")
        if detail_df is not None and not detail_df.empty:
            detail_df.to_excel(writer, index=False, sheet_name="detail_data")

            if not chart_df.empty:
                detail_columns = {name: idx for idx, name in enumerate(detail_df.columns)}
                yes_cols = [
                    detail_columns.get("first_pod_complience"),
                    detail_columns.get("second_pod_complience"),
                    detail_columns.get("third_pod_complience"),
                ]
                yes_cols = [idx for idx in yes_cols if idx is not None]
                if yes_cols:
                    total_rows = len(detail_df) + 1

                    def _countif_sum(target: str) -> str:
                        exprs = []
                        for col_idx in yes_cols:
                            col_name = xl_col_to_name(col_idx)
                            exprs.append(f'COUNTIF(detail_data!${col_name}$2:${col_name}${total_rows},"{target}")')
                        return "+".join(exprs) if exprs else "0"

                    for ridx, rec in chart_df.iterrows():
                        chart_name = str(rec.get("chart", ""))
                        category = str(rec.get("category", "")).strip().lower()
                        excel_row = ridx + 1
                        if chart_name == "Manual POD qualified rate":
                            if "not" in category:
                                writer.sheets["kpi_chart_data"].write_formula(excel_row, 2, f'={_countif_sum("No")}')
                            else:
                                writer.sheets["kpi_chart_data"].write_formula(excel_row, 2, f'={_countif_sum("Yes")}')
                        if chart_name == "24h attempt rate":
                            if "no" in category:
                                writer.sheets["kpi_chart_data"].write_formula(
                                    excel_row,
                                    2,
                                    f'=MAX(COUNTA(detail_data!$A$2:$A${total_rows})-({_countif_sum("Yes")}),0)',
                                )
                            else:
                                writer.sheets["kpi_chart_data"].write_formula(excel_row, 2, f'={_countif_sum("Yes")}')

        pod_review_df = kpi_payload.get("pod_review_df")
        if isinstance(pod_review_df, pd.DataFrame):
            pod_review_df.to_excel(writer, index=False, sheet_name="manual_review_data")

        data_ws = writer.sheets["kpi_chart_data"]
        chart_ws = workbook.add_worksheet("kpi_charts")

        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        summary_ws = writer.sheets["kpi_summary"]
        summary_ws.set_column("A:B", 40)
        summary_ws.set_column("C:D", 12)
        summary_ws.set_column("E:E", 14, percent_fmt)
        data_ws.set_column("A:B", 40)
        data_ws.set_column("C:C", 12)
        data_ws.set_column("D:D", 14, percent_fmt)
        if detail_df is not None and not detail_df.empty:
            detail_ws = writer.sheets["detail_data"]
            detail_ws.set_column(0, max(len(detail_df.columns) - 1, 0), 20)
        if "manual_review_data" in writer.sheets:
            pod_ws = writer.sheets["manual_review_data"]
            pod_ws.set_column(0, 16, 20)
            pod_ws.set_column(17, 17, 60)
            pod_ws.set_column(18, 20, 24)

        row_cursor = 0
        col_cursor = 0
        max_cols = 3
        chart_order = list(chart_df["chart"].dropna().astype(str).unique())
        def _chart_group_key(name: str) -> int:
            lower = name.lower()
            if "delivery" in lower:
                return 0
            if "scan" in lower:
                return 1
            if "pod" in lower or "attempt" in lower:
                return 2
            if "lost" in lower:
                return 3
            return 4

        chart_order = sorted(chart_order, key=lambda n: (_chart_group_key(n), chart_order.index(n)))

        for chart_name in chart_order:
            group = chart_df[chart_df["chart"] == chart_name]

            rows = group.index.to_list()
            if not rows:
                continue
            excel_rows = [r + 1 for r in rows]

            pie = workbook.add_chart({"type": "pie"})
            pie.add_series(
                {
                    "name": chart_name,
                    "categories": ["kpi_chart_data", excel_rows[0], 1, excel_rows[-1], 1],
                    "values": ["kpi_chart_data", excel_rows[0], 2, excel_rows[-1], 2],
                    "data_labels": {"percentage": True, "category": True},
                }
            )
            pie.set_title({"name": chart_name})
            pie.set_style(10)

            chart_ws.insert_chart(row_cursor, col_cursor, pie, {"x_scale": 1.0, "y_scale": 1.0})
            col_cursor += 8
            if col_cursor >= max_cols * 8:
                col_cursor = 0
                row_cursor += 16

    return output.getvalue()
