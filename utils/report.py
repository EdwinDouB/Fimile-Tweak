from datetime import datetime
from utils.utils import to_datetime_series, rate
from utils.routes import split_pickup_routes, build_lost_package_analysis
from typing import Any
import pandas as pd
import io 

def build_kpi_report_payload(result_df: pd.DataFrame, fetch_reference_time: datetime | None = None) -> dict[str, Any]:
    df = result_df.copy()
    df["created_dt"] = to_datetime_series(df, "created_time")
    df["first_scanned_dt"] = to_datetime_series(df, "first_scanned_time")
    df["last_scanned_dt"] = to_datetime_series(df, "last_scanned_time")
    df["ofd_dt"] = to_datetime_series(df, "out_for_delivery_time")
    df["attempted_dt"] = to_datetime_series(df, "attempted_time")
    df["delivered_dt"] = to_datetime_series(df, "delivered_time")
    df["month"] = df["created_dt"].dt.to_period("M").astype(str)
    df.loc[df["month"] == "NaT", "month"] = "未知"
    non_pickup_df, _ = split_pickup_routes(df)

    metrics: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []

    non_pickup_df["ofd_to_delivered_hours"] = (non_pickup_df["delivered_dt"] - non_pickup_df["ofd_dt"]).dt.total_seconds() / 3600
    ofd_present_mask = non_pickup_df["out_for_delivery_time"].notna() & non_pickup_df["out_for_delivery_time"].astype(str).str.strip().ne("")
    ofd_base = non_pickup_df[ofd_present_mask].copy()

    for threshold in [24, 48, 72]:
        within = ofd_base[
            ofd_base["delivered_dt"].notna() & (ofd_base["ofd_to_delivered_hours"] >= 0) & (ofd_base["ofd_to_delivered_hours"] < threshold)
        ]
        metric_name = f"<{threshold}h 妥投率"
        hit_count = len(within)
        total_count = len(ofd_base)
        miss_count = max(total_count - hit_count, 0)
        metrics.append(
            {
                "分类": "24/48/72 小时妥投率（上网 -> 妥投）",
                "指标": metric_name,
                "命中": hit_count,
                "总数": total_count,
                "占比": rate(hit_count, total_count),
            }
        )
        chart_rows.extend(
            [
                {"图表": metric_name, "分类": f"<{threshold}h妥投", "数量": hit_count, "占比": rate(hit_count, total_count)},
                {"图表": metric_name, "分类": f">={threshold}h或未妥投", "数量": miss_count, "占比": rate(miss_count, total_count)},
            ]
        )

    df["created_to_scan_hours"] = (df["first_scanned_dt"] - df["created_dt"]).dt.total_seconds() / 3600
    total_count = len(df)
    for threshold in [12, 24, 48, 72]:
        within = df[
            df["first_scanned_dt"].notna() & (df["created_to_scan_hours"] >= 0) & (df["created_to_scan_hours"] < threshold)
        ]
        metric_name = f"<{threshold}h 上网率"
        hit_count = len(within)
        miss_count = max(total_count - hit_count, 0)
        metrics.append(
            {
                "分类": "12/24/48/72 小时上网率（提货 -> 上网）",
                "指标": metric_name,
                "命中": hit_count,
                "总数": total_count,
                "占比": rate(hit_count, total_count),
            }
        )
        chart_rows.extend(
            [
                {"图表": metric_name, "分类": f"<{threshold}h上网", "数量": hit_count, "占比": rate(hit_count, total_count)},
                {"图表": metric_name, "分类": f">={threshold}h或未上网", "数量": miss_count, "占比": rate(miss_count, total_count)},
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
            "分类": "月丢包率（Last Scan 120h 口径）",
            "指标": "整体月丢包率口径",
            "命中": lost_total,
            "总数": scanned_total,
            "占比": rate(lost_total, scanned_total),
        }
    )
    chart_rows.extend(
        [
            {"图表": "整体月丢包率口径", "分类": "丢包", "数量": lost_total, "占比": rate(lost_total, scanned_total)},
            {
                "图表": "整体月丢包率口径",
                "分类": "未丢包",
                "数量": max(scanned_total - lost_total, 0),
                "占比": rate(max(scanned_total - lost_total, 0), scanned_total),
            },
        ]
    )

    return {
        "metrics": metrics,
        "charts": chart_rows,
        "has_monthly_lost_data": not monthly_lost.empty,
        "monthly_lost": monthly_lost,
    }

def kpi_report_to_excel_bytes(kpi_payload: dict[str, Any], detail_df: pd.DataFrame | None = None) -> bytes:
    output = io.BytesIO()
    metrics_df = pd.DataFrame(kpi_payload["metrics"])
    chart_df = pd.DataFrame(kpi_payload["charts"])

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="kpi_summary")
        chart_df.to_excel(writer, index=False, sheet_name="kpi_chart_data")
        if detail_df is not None and not detail_df.empty:
            detail_df.to_excel(writer, index=False, sheet_name="detail_data")

        workbook = writer.book
        data_ws = writer.sheets["kpi_chart_data"]
        chart_ws = workbook.add_worksheet("kpi_charts")

        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        summary_ws = writer.sheets["kpi_summary"]
        summary_ws.set_column("A:B", 34)
        summary_ws.set_column("C:D", 12)
        summary_ws.set_column("E:E", 14, percent_fmt)
        data_ws.set_column("A:B", 32)
        data_ws.set_column("C:C", 12)
        data_ws.set_column("D:D", 14, percent_fmt)
        if detail_df is not None and not detail_df.empty:
            detail_ws = writer.sheets["detail_data"]
            detail_ws.set_column(0, max(len(detail_df.columns) - 1, 0), 20)

        row_cursor = 0
        for chart_name, group in chart_df.groupby("图表", sort=False):
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
            chart_ws.insert_chart(row_cursor, 0, pie, {"x_scale": 1.2, "y_scale": 1.2})
            row_cursor += 18

    return output.getvalue()
