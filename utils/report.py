from utils.utils import to_datetime_series, rate
from datetime import datetime, timezone
from utils.routes import split_pickup_routes, build_lost_package_analysis
from typing import Any
import ast
import json
import math
import pandas as pd
import io 
from xlsxwriter.utility import xl_col_to_name

WEIGHT_DISTRIBUTION_START_ROW = 100


def _resolve_ofd_column(df: pd.DataFrame) -> str:
    if "first_out_for_delivery_date" in df.columns:
        return "first_out_for_delivery_date"
    return "out_for_delivery_time"


def _yes_no_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(["" for _ in range(len(df))], index=df.index, dtype="object")
    return df[col].fillna("").astype(str).str.strip().str.lower()


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


def _build_delivery_attempts_df(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()

    success_types = {"success", "delivered"}
    fail_types = {"fail", "failed", "failure"}
    ofd_types = {"out-for-delivery", "ofd", "outfordelivery"}

    attempts: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        intervals = _load_intervals(row.get("Intervals"))
        if not intervals:
            continue

        idx = 0
        while idx < len(intervals):
            event = intervals[idx]
            event_type_value = str(event.get("type") or "").strip().lower()
            if event_type_value not in ofd_types:
                idx += 1
                continue

            current_ofd_event = event
            search_idx = idx + 1
            matched_terminal = None
            while search_idx < len(intervals):
                candidate_event = intervals[search_idx]
                candidate_type = str(candidate_event.get("type") or "").strip().lower()
                if candidate_type in ofd_types:
                    current_ofd_event = candidate_event
                    search_idx += 1
                    continue
                if candidate_type in (success_types | fail_types):
                    matched_terminal = candidate_event
                    break
                search_idx += 1

            attempt_row = row.to_dict()
            attempt_row["out_for_delivery_time"] = current_ofd_event.get("time")
            attempt_row["terminal_time"] = matched_terminal.get("time") if matched_terminal else None
            if matched_terminal is None:
                attempt_row["attempt_result"] = "lost"
                idx = len(intervals)
            else:
                matched_type = str(matched_terminal.get("type") or "").strip().lower()
                if matched_type in success_types:
                    attempt_row["attempt_result"] = "success"
                elif matched_type in fail_types:
                    attempt_row["attempt_result"] = "fail"
                else:
                    attempt_row["attempt_result"] = "lost"
                idx = search_idx + 1

            attempts.append(attempt_row)

    return pd.DataFrame(attempts)


def build_attempt_kpi_detail_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build attempt-level KPI detail for OFD based delivery/attempt metrics."""
    attempt_level_df = _build_delivery_attempts_df(source_df)
    if attempt_level_df.empty:
        return attempt_level_df

    attempt_level_df = attempt_level_df.copy()
    attempt_level_df["ofd_dt"] = _parse_attempt_event_time(attempt_level_df["out_for_delivery_time"])
    attempt_level_df["terminal_dt"] = _parse_attempt_event_time(attempt_level_df["terminal_time"])
    attempt_level_df["ofd_to_terminal_hours"] = (
        attempt_level_df["terminal_dt"] - attempt_level_df["ofd_dt"]
    ).dt.total_seconds() / 3600

    for threshold in [24, 48, 72]:
        attempt_level_df[f"delivered_within_{threshold}h"] = (
            attempt_level_df["attempt_result"].eq("success")
            & attempt_level_df["terminal_dt"].notna()
            & (attempt_level_df["ofd_to_terminal_hours"] >= 0)
            & (attempt_level_df["ofd_to_terminal_hours"] < threshold)
        )

    attempt_level_df["attempt_within_24h"] = (
        attempt_level_df["attempt_result"].isin(["success", "fail"])
        & attempt_level_df["terminal_dt"].notna()
        & (attempt_level_df["ofd_to_terminal_hours"] >= 0)
        & (attempt_level_df["ofd_to_terminal_hours"] < 24)
    )
    return attempt_level_df


def _parse_attempt_event_time(series: pd.Series) -> pd.Series:
    """Parse event timestamps supporting unix-ms/unix-s/ISO text."""
    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    def _normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            # Mongo-like payloads: {"$date": {"$numberLong": "..."}} / {"$date": "..."}
            if "$date" in value:
                return _normalize_value(value.get("$date"))
            if "$numberLong" in value:
                return value.get("$numberLong")
            if "time" in value:
                return _normalize_value(value.get("time"))
        return value

    text_series = series.map(_normalize_value).astype("object")
    numeric = pd.to_numeric(text_series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        numeric_values = numeric[numeric_mask]
        millisecond_mask = numeric_values.abs() >= 1e11
        if millisecond_mask.any():
            parsed.loc[numeric_values[millisecond_mask].index] = pd.to_datetime(
                numeric_values[millisecond_mask], unit="ms", errors="coerce", utc=True
            ).dt.tz_convert(None)

        second_mask = ~millisecond_mask
        if second_mask.any():
            parsed.loc[numeric_values[second_mask].index] = pd.to_datetime(
                numeric_values[second_mask], unit="s", errors="coerce", utc=True
            ).dt.tz_convert(None)

    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(text_series[fallback_mask], errors="coerce", utc=True).dt.tz_convert(None)

    return parsed

def _build_detailed_overview_table(detail_df: pd.DataFrame, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=[
            "Dimension", "Sample Count", "<24h Hit", "<24h Delivery Rate", "<48h Hit", "<48h Delivery Rate", "<72h Hit", "<72h Delivery Rate",
            "<12h Scan Rate", "<24h Scan Rate", "<48h Scan Rate", "<72h Scan Rate", "POD Qualified Rate", "24h Attempt Rate", "DSP Lost Rate", "Warehouse Lost Rate", "Lost Rate", "Intercept Success Rate",
        ])

    source_attempt_df = detail_df.copy()
    source_tracking_df = source_df.copy() if source_df is not None and not source_df.empty else source_attempt_df.copy()

    ofd_col = _resolve_ofd_column(source_attempt_df)
    source_attempt_df["ofd_dt"] = to_datetime_series(source_attempt_df, ofd_col)
    if "delivered_time" in source_attempt_df.columns:
        source_attempt_df["delivered_dt"] = to_datetime_series(source_attempt_df, "delivered_time")
    else:
        source_attempt_df["delivered_dt"] = pd.NaT

    source_attempt_df["ofd_to_delivered_hours"] = (source_attempt_df["delivered_dt"] - source_attempt_df["ofd_dt"]).dt.total_seconds() / 3600
    for threshold in [24, 48, 72]:
        source_attempt_df[f"within_{threshold}h"] = (
            source_attempt_df["delivered_dt"].notna()
            & (source_attempt_df["ofd_to_delivered_hours"] >= 0)
            & (source_attempt_df["ofd_to_delivered_hours"] < threshold)
        )

    def _append_row(rows: list[dict[str, Any]], dimension: str, sub_df: pd.DataFrame, sub_source_df: pd.DataFrame) -> None:
        total_count = len(sub_df)
        row = {"Dimension": dimension, "Sample Count": total_count}
        for threshold in [24, 48, 72]:
            hit = int(sub_df[f"within_{threshold}h"].sum()) if total_count > 0 else 0
            row[f"<{threshold}h Hit"] = hit
            row[f"<{threshold}h Delivery Rate"] = rate(hit, total_count)

        sub_payload = build_kpi_report_payload(sub_source_df)
        metric_map = {
            str(item.get("metric")): item
            for item in sub_payload.get("metrics", [])
            if isinstance(item, dict)
        }
        pod_rate_from_data = _pod_rate_from_detail_rows(sub_df)
        row["<12h Scan Rate"] = _metric_rate(metric_map, "<12h scan rate")
        row["<24h Scan Rate"] = _metric_rate(metric_map, "<24h scan rate")
        row["<48h Scan Rate"] = _metric_rate(metric_map, "<48h scan rate")
        row["<72h Scan Rate"] = _metric_rate(metric_map, "<72h scan rate")
        row["POD Qualified Rate"] = _safe_rate_value(_first_available_metric_rate(metric_map, ["POD qualified rate", "Manual POD qualified rate"]) or pod_rate_from_data)
        row["24h Attempt Rate"] = _safe_rate_value(_metric_rate(metric_map, "24h attempt rate"))
        row["DSP Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "DSP lost rate"))
        row["Warehouse Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "Warehouse lost rate"))
        row["Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "lost rate"))
        rows.append(row)

    rows: list[dict[str, Any]] = []
    _append_row(rows, "Overall", source_attempt_df, source_tracking_df)

    if "Region" in source_attempt_df.columns:
        attempt_region_series = source_attempt_df["Region"].fillna("Unknown Region").astype(str).str.strip().replace("", "Unknown Region")
        tracking_region_series = source_tracking_df["Region"].fillna("Unknown Region").astype(str).str.strip().replace("", "Unknown Region") if "Region" in source_tracking_df.columns else pd.Series(["Unknown Region"] * len(source_tracking_df), index=source_tracking_df.index)
        for region in sorted(attempt_region_series.unique()):
            region_df = source_attempt_df[attempt_region_series == region]
            region_source_df = source_tracking_df[tracking_region_series == region]
            _append_row(rows, region, region_df, region_source_df)

            if "Hub" in source_attempt_df.columns:
                hub_series = region_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
                for hub in sorted(hub_series.unique()):
                    hub_df = region_df[hub_series == hub]
                    if "Hub" in region_source_df.columns:
                        source_hub_series = region_source_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
                        hub_source_df = region_source_df[source_hub_series == hub]
                    else:
                        hub_source_df = region_source_df
                    _append_row(rows, f"  {hub}", hub_df, hub_source_df)

    return pd.DataFrame(rows)


def _build_hub_table(detail_df: pd.DataFrame, hub_name: str, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    hub_df = detail_df[detail_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub") == hub_name].copy()
    hub_source_df = source_df.copy() if source_df is not None and not source_df.empty else detail_df.copy()
    if "Hub" in hub_source_df.columns:
        hub_source_df = hub_source_df[hub_source_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub") == hub_name].copy()
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

        contractor_source_df = hub_source_df
        if dimension.startswith("  ") and "Contractor" in hub_source_df.columns:
            contractor = dimension.strip()
            contractor_source_df = hub_source_df[
                hub_source_df["Contractor"].fillna("Unknown Contractor").astype(str).str.strip().replace("", "Unknown Contractor") == contractor
            ]

        sub_payload = build_kpi_report_payload(contractor_source_df)
        metric_map = {
            str(item.get("metric")): item
            for item in sub_payload.get("metrics", [])
            if isinstance(item, dict)
        }
        pod_rate_from_data = _pod_rate_from_detail_rows(sub_df)
        row["<12h Scan Rate"] = _metric_rate(metric_map, "<12h scan rate")
        row["<24h Scan Rate"] = _metric_rate(metric_map, "<24h scan rate")
        row["<48h Scan Rate"] = _metric_rate(metric_map, "<48h scan rate")
        row["<72h Scan Rate"] = _metric_rate(metric_map, "<72h scan rate")
        row["POD Qualified Rate"] = _safe_rate_value(_first_available_metric_rate(metric_map, ["POD qualified rate", "Manual POD qualified rate"]) or pod_rate_from_data)
        row["24h Attempt Rate"] = _safe_rate_value(_metric_rate(metric_map, "24h attempt rate"))
        row["DSP Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "DSP lost rate"))
        row["Warehouse Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "Warehouse lost rate"))
        row["Lost Rate"] = _safe_rate_value(_metric_rate(metric_map, "lost rate"))
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
    worksheet.set_column(1, max(len(table_df.columns) - 1, 1), 18)

def _sanitize_sheet_name(name: str) -> str:
    invalid_chars = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in str(name))
    return cleaned[:31]


def _metric_rate(metric_map: dict[str, dict[str, Any]], metric_name: str) -> float:
    metric = metric_map.get(metric_name, {})
    if not isinstance(metric, dict):
        return 0.0
    hit = metric.get("hit", 0)
    total = metric.get("total", 0)
    direct_rate = metric.get("rate", 0.0)
    if total:
        return rate(hit, total)
    value = float(direct_rate or 0.0)
    return 0.0 if pd.isna(value) else value


def _first_available_metric_rate(metric_map: dict[str, dict[str, Any]], metric_names: list[str]) -> float:
    for metric_name in metric_names:
        if metric_name in metric_map:
            return _metric_rate(metric_map, metric_name)
    return 0.0


def _safe_rate_value(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return 0.0 if pd.isna(result) else result


def _pod_rate_from_detail_rows(sub_df: pd.DataFrame) -> float:
    if sub_df is None or sub_df.empty:
        return 0.0

    if "POD是否合格" in sub_df.columns:
        pod_hit = int(sub_df["POD是否合格"].fillna("").astype(str).str.strip().eq("是").sum())
        return rate(pod_hit, len(sub_df))

    compliance_cols = [
        col for col in ["first_pod_complience", "second_pod_complience", "third_pod_complience"] if col in sub_df.columns
    ]
    if not compliance_cols:
        return 0.0

    yes_count = 0
    no_count = 0
    for col in compliance_cols:
        normalized = _yes_no_series(sub_df, col)
        yes_count += int(normalized.eq("yes").sum())
        no_count += int(normalized.eq("no").sum())

    reviewed_count = yes_count + no_count
    return rate(yes_count, reviewed_count)


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
        "24h attempt rate": (chart_row + 16, chart_col + 8),
        "POD qualified rate": (chart_row + 16, chart_col + 16),
        "lost rate": (chart_row + 32, chart_col),
        "DSP lost rate": (chart_row + 32, chart_col + 8),
        "Warehouse lost rate": (chart_row + 32, chart_col + 16),
    }

    for chart_name in ["<24h delivery rate", "<48h delivery rate", "<72h delivery rate", "24h attempt rate", "POD qualified rate", "lost rate", "DSP lost rate", "Warehouse lost rate"]:
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


def _normalize_weight_to_unit(weight_series: pd.Series) -> pd.Series:
    numeric_weight = pd.to_numeric(weight_series, errors="coerce")
    normalized_weight = numeric_weight.copy()
    positive_mask = normalized_weight > 0
    normalized_weight.loc[positive_mask] = normalized_weight.loc[positive_mask].apply(math.ceil)
    return normalized_weight


def _weight_bucket_labels(df: pd.DataFrame | None = None) -> list[str]:
    weight_series = _resolve_weight_series(df) if isinstance(df, pd.DataFrame) else pd.Series(dtype="float64")
    normalized_weight = _normalize_weight_to_unit(weight_series)
    positive_weight = normalized_weight[normalized_weight.notna() & (normalized_weight > 0)]
    if positive_weight.empty:
        return []
    max_weight = int(positive_weight.max())
    return [str(weight) for weight in range(1, max_weight + 1)]


def _resolve_weight_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    weight_candidates = ["Weight", "weight", "package_weight"]
    for col in weight_candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def _resolve_weight_distribution(df: pd.DataFrame, dimension_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    labels = _weight_bucket_labels(df)
    if not labels:
        return pd.DataFrame()

    weight_series = _resolve_weight_series(df)
    normalized_weight = _normalize_weight_to_unit(weight_series)
    valid_df = df[normalized_weight.notna() & (normalized_weight > 0)].copy()
    if valid_df.empty:
        return pd.DataFrame()

    valid_df["_resolved_weight_unit"] = normalized_weight.loc[valid_df.index].astype(int).astype(str)

    if dimension_col is None:
        counts = (
            valid_df.groupby("_resolved_weight_unit", observed=False)
            .size()
            .reindex(labels, fill_value=0)
            .astype(int)
        )
        return pd.DataFrame([{"维度": "总体", **counts.to_dict()}])

    if dimension_col not in valid_df.columns:
        return pd.DataFrame()

    dimension_series = valid_df[dimension_col].fillna(f"Unknown {dimension_col}").astype(str).str.strip().replace("", f"Unknown {dimension_col}")
    grouped = (
        valid_df.assign(_dimension=dimension_series)
        .groupby(["_dimension", "_resolved_weight_unit"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
        .astype(int)
        .reset_index()
        .rename(columns={"_dimension": "维度"})
        .sort_values("维度")
    )
    return grouped


def _write_weight_distribution_section(
    worksheet,
    workbook,
    start_row: int,
    title: str,
    table_df: pd.DataFrame,
    chart_anchor_col: int | None = None,
    chart_anchor_row: int | None = None,
    deferred_charts: list[dict[str, Any]] | None = None,
) -> int:
    if table_df.empty:
        return start_row

    title_fmt = workbook.add_format({"bold": True, "font_size": 12})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#e5e7eb", "border": 1})
    cell_fmt = workbook.add_format({"border": 1})

    worksheet.write(start_row, 0, title, title_fmt)
    header_row = start_row + 1
    for cidx, col in enumerate(table_df.columns):
        worksheet.write(header_row, cidx, col, header_fmt)

    for ridx, (_, row) in enumerate(table_df.iterrows(), start=header_row + 1):
        for cidx, col in enumerate(table_df.columns):
            worksheet.write(ridx, cidx, row[col], cell_fmt)

    worksheet.set_column(0, 0, 22)
    worksheet.set_column(1, max(len(table_df.columns) - 1, 1), 10)

    first_data_row = header_row + 1
    last_data_row = header_row + len(table_df)
    last_col = len(table_df.columns) - 1

    row_totals = table_df.iloc[:, 1:].sum(axis=1)

    def _build_pct_labels(values: list[Any], total: float) -> list[dict[str, str]]:
        if total <= 0:
            return [{"value": "0.0%"} for _ in values]
        return [
            {"value": f"{(float(value) / float(total)) * 100:.1f}%"}
            for value in values
        ]

    chart_opts = {"type": "column"}
    if len(table_df) > 1:
        chart_opts["subtype"] = "stacked"
    chart = workbook.add_chart(chart_opts)
    if len(table_df) == 1:
        single_values = table_df.iloc[0, 1:].tolist()
        single_total = float(row_totals.iloc[0]) if not row_totals.empty else 0.0
        chart.add_series(
            {
                "name": title,
                "categories": [worksheet.get_name(), header_row, 1, header_row, last_col],
                "values": [worksheet.get_name(), first_data_row, 1, first_data_row, last_col],
                "data_labels": {"custom": _build_pct_labels(single_values, single_total)},
            }
        )
    else:
        for cidx in range(1, len(table_df.columns)):
            series_values = table_df.iloc[:, cidx].tolist()
            pct_labels = [
                {
                    "value": f"{((float(series_values[idx]) / float(row_totals.iloc[idx])) * 100):.1f}%"
                    if float(row_totals.iloc[idx]) > 0
                    else "0.0%"
                }
                for idx in range(len(series_values))
            ]
            chart.add_series(
                {
                    "name": [worksheet.get_name(), header_row, cidx],
                    "categories": [worksheet.get_name(), first_data_row, 0, last_data_row, 0],
                    "values": [worksheet.get_name(), first_data_row, cidx, last_data_row, cidx],
                    "data_labels": {"custom": pct_labels},
                }
            )
    chart.set_title({"name": title})
    chart.set_legend({"position": "bottom"})
    chart_options = {"x_scale": 1.2, "y_scale": 1.1}
    if deferred_charts is not None:
        deferred_charts.append({"chart": chart, "options": chart_options})
    elif chart_anchor_col is not None:
        worksheet.insert_chart(chart_anchor_row if chart_anchor_row is not None else start_row, chart_anchor_col, chart, chart_options)

    return last_data_row + 3


def _insert_deferred_charts_grid(
    worksheet,
    deferred_charts: list[dict[str, Any]],
    start_row: int,
    start_col: int,
    charts_per_row: int = 3,
    row_step: int = 16,
    col_step: int = 8,
) -> None:
    if not deferred_charts:
        return

    for idx, chart_info in enumerate(deferred_charts):
        row_offset = (idx // charts_per_row) * row_step
        col_offset = (idx % charts_per_row) * col_step
        worksheet.insert_chart(
            start_row + row_offset,
            start_col + col_offset,
            chart_info["chart"],
            chart_info["options"],
        )


def _ensure_manual_review_weight_columns(pod_review_df: pd.DataFrame, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if pod_review_df is None or pod_review_df.empty:
        return pod_review_df

    normalized_df = pod_review_df.copy()
    if source_df is not None and not source_df.empty and "tracking_id" in normalized_df.columns and "tracking_id" in source_df.columns:
        source_weight_df = source_df[["tracking_id"]].copy()
        source_weight_df["_resolved_weight"] = _resolve_weight_series(source_df)
        source_weight_df = source_weight_df.dropna(subset=["_resolved_weight"]).drop_duplicates(subset=["tracking_id"], keep="first")
        if not source_weight_df.empty:
            normalized_df = normalized_df.merge(source_weight_df, on="tracking_id", how="left")
            if "package_weight" not in normalized_df.columns:
                normalized_df["package_weight"] = normalized_df["_resolved_weight"]
            else:
                normalized_df["package_weight"] = pd.to_numeric(normalized_df["package_weight"], errors="coerce").fillna(normalized_df["_resolved_weight"])
            normalized_df = normalized_df.drop(columns=["_resolved_weight"], errors="ignore")

    if "package_weight" not in normalized_df.columns:
        normalized_df["package_weight"] = _resolve_weight_series(normalized_df)
    else:
        normalized_df["package_weight"] = pd.to_numeric(normalized_df["package_weight"], errors="coerce")

    if "Weight" not in normalized_df.columns:
        normalized_df["Weight"] = normalized_df["package_weight"]
    else:
        normalized_df["Weight"] = pd.to_numeric(normalized_df["Weight"], errors="coerce").fillna(normalized_df["package_weight"])

    return normalized_df



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

    attempt_level_df = build_attempt_kpi_detail_df(non_pickup_df)

    # Denominator for these metrics is all OFD attempts (including lost attempts),
    # not unique waybills and not only rows with parseable timestamps.
    attempt_base = attempt_level_df.copy() if not attempt_level_df.empty else pd.DataFrame()

    for threshold in [24, 48, 72]:
        within = attempt_base[
            (attempt_base["attempt_result"] == "success")
            & attempt_base["terminal_dt"].notna()
            & attempt_base["ofd_dt"].notna()
            & (attempt_base["ofd_to_terminal_hours"] >= 0)
            & (attempt_base["ofd_to_terminal_hours"] < threshold)
        ] if not attempt_base.empty else pd.DataFrame()
        metric_name = f"<{threshold}h delivery rate"
        hit_count = len(within)
        total_count = len(attempt_base)
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

    package_review_base = non_pickup_df[non_pickup_df["ofd_dt"].notna()].copy()
    package_review_base["ofd_to_attempted_hours"] = (package_review_base["attempted_dt"] - package_review_base["ofd_dt"]).dt.total_seconds() / 3600
    package_review_base["ofd_to_delivered_hours"] = (package_review_base["delivered_dt"] - package_review_base["ofd_dt"]).dt.total_seconds() / 3600
    package_failed_without_delivery_mask = package_review_base["attempted_dt"].notna() & package_review_base["delivered_dt"].isna()
    package_failed_attempt_within_24h_mask = (
        package_failed_without_delivery_mask
        & (package_review_base["ofd_to_attempted_hours"] >= 0)
        & (package_review_base["ofd_to_attempted_hours"] < 24)
    )
    package_delivered_within_24h_mask = (
        package_review_base["delivered_dt"].notna()
        & (package_review_base["ofd_to_delivered_hours"] >= 0)
        & (package_review_base["ofd_to_delivered_hours"] < 24)
    )

    review_base = package_review_base.copy()
    review_base["review_date"] = review_base["ofd_dt"].dt.date.astype(str).replace("NaT", "")
    review_base["driver_name"] = review_base.get("Driver", "")
    review_base["dsp"] = review_base.get("Contractor", "")
    review_base["route_code"] = review_base.get("Route_name", "")
    review_base["delivery_status"] = review_base["delivered_dt"].notna().map(lambda x: "Delivered" if x else "Not Delivered")
    review_base["stop_status"] = "No Attempt"
    review_base.loc[package_failed_without_delivery_mask, "stop_status"] = "Failed"
    review_base.loc[review_base["delivered_dt"].notna(), "stop_status"] = "Delivered"
    review_base["event_code"] = "NO_ATTEMPT"
    review_base.loc[package_failed_without_delivery_mask, "event_code"] = "FAILED_STOP"
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
    review_base["package_weight"] = _resolve_weight_series(review_base)
    review_base["Weight"] = review_base["package_weight"]
    review_base["out_for_delivery_time"] = review_base.get(ofd_col, "")
    review_base["manual_pod_review_status"] = "Pending"
    review_base["manual_review_note"] = ""
    review_base["attempt_validated"] = False
    review_base.loc[package_delivered_within_24h_mask, "attempt_validated"] = True
    review_base.loc[package_failed_attempt_within_24h_mask, "attempt_validated"] = True

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
        "out_for_delivery_time",
        "package_weight",
        "Weight",
        "attempted_time",
        "delivered_time",
        "beans_link",
        "manual_pod_review_status",
        "manual_review_note",
        "attempt_validated",
    ]
    for col in pod_review_export_columns:
        if col not in review_base.columns:
            review_base[col] = ""
    pod_review_export_df = review_base[pod_review_export_columns].copy()

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

    failed_without_delivery_count = int((attempt_base["attempt_result"] == "fail").sum()) if not attempt_base.empty else 0
    attempt_hit_mask = (
        attempt_base["terminal_dt"].notna()
        & attempt_base["ofd_dt"].notna()
        & (attempt_base["ofd_to_terminal_hours"] >= 0)
        & (attempt_base["ofd_to_terminal_hours"] < 24)
        & attempt_base["attempt_result"].isin(["success", "fail"])
    ) if not attempt_base.empty else pd.Series(dtype=bool)
    attempt_total_count = len(attempt_base)
    attempt_hit_count = int(attempt_hit_mask.sum()) if not attempt_base.empty else 0
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
                "hit": 0,
                "total": attempt_total_count,
                "rate": rate(0, attempt_total_count),
            },
        ]
    )

    dsp_lost_hit = int((attempt_base["attempt_result"] == "lost").sum()) if not attempt_base.empty else 0
    dsp_lost_total = attempt_total_count
    metrics.append(
        {
            "category": "dsp_assessment",
            "metric": "DSP lost rate",
            "hit": dsp_lost_hit,
            "total": dsp_lost_total,
            "rate": rate(dsp_lost_hit, dsp_lost_total),
        }
    )
    chart_rows.extend(
        [
            {
                "chart": "DSP lost rate",
                "category": "DSP lost",
                "count": dsp_lost_hit,
                "rate": rate(dsp_lost_hit, dsp_lost_total),
            },
            {
                "chart": "DSP lost rate",
                "category": "Not DSP lost",
                "count": max(dsp_lost_total - dsp_lost_hit, 0),
                "rate": rate(max(dsp_lost_total - dsp_lost_hit, 0), dsp_lost_total),
            },
        ]
    )

    lost_analysis = build_lost_package_analysis(df, fetch_reference_time=fetch_reference_time)
    scanned_base = lost_analysis["scanned_base"]
    scanned_base["lost"] = lost_analysis["lost_mask"].loc[scanned_base.index].astype(int)
    monthly_lost = scanned_base.groupby("month", as_index=False).agg(total=("tracking_id", "count"), lost=("lost", "sum"))

    warehouse_lost_hit = int(scanned_base["lost"].sum()) if not scanned_base.empty else 0
    warehouse_lost_total = len(scanned_base)
    metrics.append(
        {
            "category": "hub_assessment",
            "metric": "Warehouse lost rate",
            "hit": warehouse_lost_hit,
            "total": warehouse_lost_total,
            "rate": rate(warehouse_lost_hit, warehouse_lost_total),
        }
    )
    chart_rows.extend(
        [
            {
                "chart": "Warehouse lost rate",
                "category": "Warehouse lost",
                "count": warehouse_lost_hit,
                "rate": rate(warehouse_lost_hit, warehouse_lost_total),
            },
            {
                "chart": "Warehouse lost rate",
                "category": "Not warehouse lost",
                "count": max(warehouse_lost_total - warehouse_lost_hit, 0),
                "rate": rate(max(warehouse_lost_total - warehouse_lost_hit, 0), warehouse_lost_total),
            },
        ]
    )

    if "tracking_id" in df.columns and not df.empty:
        all_tracking_ids = set(df["tracking_id"].astype(str))
    else:
        all_tracking_ids = set(df.index.astype(str))

    dsp_lost_tracking_ids = set()
    if not attempt_base.empty and "tracking_id" in attempt_base.columns:
        dsp_lost_tracking_ids = set(
            attempt_base.loc[attempt_base["attempt_result"] == "lost", "tracking_id"].astype(str)
        )

    warehouse_lost_tracking_ids = set()
    if "tracking_id" in scanned_base.columns and not scanned_base.empty:
        warehouse_lost_tracking_ids = set(
            scanned_base.loc[scanned_base["lost"] == 1, "tracking_id"].astype(str)
        )

    combined_lost_tracking_ids = dsp_lost_tracking_ids | warehouse_lost_tracking_ids
    combined_lost_hit = len(combined_lost_tracking_ids)
    combined_lost_total = len(all_tracking_ids)

    metrics.append(
        {
            "category": "monthly_lost_rate_last_scan_120h",
            "metric": "lost rate",
            "hit": combined_lost_hit,
            "total": combined_lost_total,
            "rate": rate(combined_lost_hit, combined_lost_total),
        }
    )
    chart_rows.extend(
        [
            {
                "chart": "lost rate",
                "category": "Lost",
                "count": combined_lost_hit,
                "rate": rate(combined_lost_hit, combined_lost_total),
            },
            {
                "chart": "lost rate",
                "category": "Not lost",
                "count": max(combined_lost_total - combined_lost_hit, 0),
                "rate": rate(max(combined_lost_total - combined_lost_hit, 0), combined_lost_total),
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

    if not chart_df.empty and "chart" in chart_df.columns:
        chart_df["chart"] = chart_df["chart"].replace({"Manual POD qualified rate": "POD qualified rate"})
    if not metrics_df.empty and "metric" in metrics_df.columns:
        metrics_df["metric"] = metrics_df["metric"].replace({"Manual POD qualified rate": "POD qualified rate"})

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        detailed_layout_ready = layout_mode == "detailed" and detail_df is not None and not detail_df.empty
        if detailed_layout_ready:
            overview_ws = workbook.add_worksheet("overview")
            overview_table = _build_detailed_overview_table(detail_df, source_df=source_df)
            _style_overview_worksheet(overview_ws, overview_table, 0, workbook)
            next_row = max(len(overview_table) + 3, WEIGHT_DISTRIBUTION_START_ROW)
            if source_df is not None and not source_df.empty:
                overview_weight_charts: list[dict[str, Any]] = []
                next_row = _write_weight_distribution_section(
                    overview_ws,
                    workbook,
                    next_row,
                    "总体重量段",
                    _resolve_weight_distribution(source_df),
                    deferred_charts=overview_weight_charts,
                )
                warehouse_weight_df = _resolve_weight_distribution(source_df, "Hub")
                if not warehouse_weight_df.empty:
                    for _, warehouse_row in warehouse_weight_df.iterrows():
                        warehouse_name = str(warehouse_row.get("维度", "Unknown Hub"))
                        single_warehouse_df = pd.DataFrame([warehouse_row])
                        next_row = _write_weight_distribution_section(
                            overview_ws,
                            workbook,
                            next_row,
                            f"仓库重量段 - ({warehouse_name})",
                            single_warehouse_df,
                            deferred_charts=overview_weight_charts,
                        )
            overview_chart_col = max(len(overview_table.columns) + 2, 10)
            _insert_dashboard_charts(
                overview_ws,
                workbook,
                chart_df,
                metrics_df,
                chart_row=0,
                chart_col=overview_chart_col,
                data_col=max(len(overview_table.columns) + 28, 36),
            )
            _insert_deferred_charts_grid(
                overview_ws,
                overview_weight_charts,
                start_row=48,
                start_col=overview_chart_col,
                charts_per_row=3,
            )

            hub_series = detail_df["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
            hub_source = source_df if source_df is not None and not source_df.empty else detail_df
            hub_source_series = hub_source["Hub"].fillna("Unknown Hub").astype(str).str.strip().replace("", "Unknown Hub")
            for hub_name in sorted(hub_series.unique()):
                hub_table = _build_hub_table(detail_df, hub_name, source_df=hub_source)
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
                hub_next_row = max(len(hub_table) + 3, WEIGHT_DISTRIBUTION_START_ROW)
                hub_weight_charts: list[dict[str, Any]] = []
                hub_next_row = _write_weight_distribution_section(
                    hub_ws,
                    workbook,
                    hub_next_row,
                    "仓库总重量段",
                    _resolve_weight_distribution(hub_df),
                    deferred_charts=hub_weight_charts,
                )
                contractor_weight_df = _resolve_weight_distribution(hub_df, "Contractor")
                if not contractor_weight_df.empty:
                    for _, contractor_row in contractor_weight_df.iterrows():
                        contractor_name = str(contractor_row.get("维度", "Unknown Contractor"))
                        single_contractor_df = pd.DataFrame([contractor_row])
                        hub_next_row = _write_weight_distribution_section(
                            hub_ws,
                            workbook,
                            hub_next_row,
                            f"DSP重量段 - ({contractor_name})",
                            single_contractor_df,
                            deferred_charts=hub_weight_charts,
                        )

                hub_chart_col = max(len(hub_table.columns) + 2, 10)
                _insert_dashboard_charts(
                    hub_ws,
                    workbook,
                    hub_chart_df,
                    hub_metrics_df,
                    chart_row=0,
                    chart_col=hub_chart_col,
                    data_col=max(len(hub_table.columns) + 28, 36),
                )
                _insert_deferred_charts_grid(
                    hub_ws,
                    hub_weight_charts,
                    start_row=48,
                    start_col=hub_chart_col,
                    charts_per_row=3,
                )
        if not detailed_layout_ready:
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
                            if chart_name in {"Manual POD qualified rate", "POD qualified rate"}:
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
        if (not isinstance(pod_review_df, pd.DataFrame) or pod_review_df.empty) and source_df is not None and not source_df.empty:
            pod_review_df = build_kpi_report_payload(source_df).get("pod_review_df")
        if isinstance(pod_review_df, pd.DataFrame):
            pod_review_df = _ensure_manual_review_weight_columns(pod_review_df, source_df=source_df)
            pod_review_df.to_excel(writer, index=False, sheet_name="manual_review_data")

        if not detailed_layout_ready:
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
        if not detailed_layout_ready:
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
