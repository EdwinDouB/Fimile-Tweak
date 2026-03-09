from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from threading import local

from utils.utils import *
from utils.db import * 
from utils.routes import *
from utils.report import * 
from utils.api import * 
from utils.constants import * 

import pandas as pd
import requests
import streamlit as st


def render_compliance_section(title: str, delivered_df: pd.DataFrame, state_key_prefix: str) -> None:
    st.subheader(title)
    if delivered_df.empty:
        st.info(tr("no_delivered"))
        return

    header_cols = st.columns([2, 2, 5, 2])
    header_cols[0].markdown(f"**{tr('delivered_date')}**")
    header_cols[1].markdown(f"**{tr('tracking_id')}**")
    header_cols[2].markdown(f"**{tr('beans_link')}**")
    header_cols[3].markdown(f"**{tr('compliant')}**")

    for idx, row in delivered_df.iterrows():
        tracking_id = str(row.get("tracking_id") or "")
        delivered_time = str(row.get("delivered_time") or "")
        compliant = st.session_state["pod_compliance_map"].get(tracking_id, False)

        cols = st.columns([2, 2, 5, 2])
        cols[0].write(delivered_time)
        cols[1].write(tracking_id)
        cols[2].markdown(f"[{tr('open_beans')}]({build_beans_tracking_link(tracking_id)})")
        btn_label = "✅" if compliant else "❌"
        if cols[3].button(btn_label, key=f"{state_key_prefix}_toggle_{idx}_{tracking_id}", use_container_width=True):
            st.session_state["pod_compliance_map"][tracking_id] = not compliant
            st.rerun()

def render_percentage_pie(
    title: str,
    hit_count: int,
    total_count: int,
    hit_label: str = "达标",
    miss_label: str = "未达标",
    chart_key: str | None = None,
    container: Any | None = None,
) -> None:
    target = container or st
    if total_count <= 0:
        target.info(f"{title}：暂无可用数据")
        return

    miss_count = max(total_count - hit_count, 0)
    chart_df = pd.DataFrame({"分类": [hit_label, miss_label], "数量": [hit_count, miss_count]})
    chart_df = chart_df[chart_df["数量"] > 0]
    chart_df["占比"] = (chart_df["数量"] / total_count).map(lambda x: f"{x:.2%}")

    target.caption(title)
    target.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "arc", "outerRadius": 100},
            "encoding": {
                "theta": {"field": "数量", "type": "quantitative"},
                "color": {"field": "分类", "type": "nominal"},
                "tooltip": [
                    {"field": "分类", "type": "nominal"},
                    {"field": "数量", "type": "quantitative"},
                    {"field": "占比", "type": "nominal"},
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
    row: dict[str, Any] = {"维度": scope_name, "样本数": total_count}
    for threshold in thresholds:
        hit_col = f"within_{threshold}h"
        hit_count = int(source_df[hit_col].sum()) if hit_col in source_df.columns else 0
        row[f"<{threshold}h命中"] = hit_count
        row[f"<{threshold}h妥投率"] = rate(hit_count, total_count)

    sample_tracking_ids = (
        source_df["tracking_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().head(8).tolist()
        if "tracking_id" in source_df.columns
        else []
    )
    row["调试运单号示例"] = ", ".join(sample_tracking_ids)
    rows.append(row)

def style_breakdown_rows(table_df: pd.DataFrame) -> Any:
    if table_df.empty or "维度" not in table_df.columns:
        return table_df

    level_colors = {
        0: "#f3f4f6",  # 总体
        1: "#dbeafe",  # 美西/美东
        2: "#e0f2fe",  # HUB
        3: "#ecfccb",  # Contractor
    }

    def _style_row(row: pd.Series) -> list[str]:
        dimension_name = str(row.get("维度", ""))
        leading_spaces = len(dimension_name) - len(dimension_name.lstrip(" "))
        indent_level = min(3, leading_spaces // 2)
        color = level_colors.get(indent_level, "#ffffff")
        return [f"background-color: {color}"] * len(row)

    return table_df.style.apply(_style_row, axis=1)


def build_delivery_breakdown_table(delivered_detail_df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    if delivered_detail_df.empty:
        return pd.DataFrame(columns=["维度", "样本数"])

    source_df = delivered_detail_df.copy()
    source_df["region_norm"] = source_df["Region"].apply(normalize_region)

    rows: list[dict[str, Any]] = []
    _append_delivery_breakdown_rows(rows, "总体", source_df, thresholds)

    for region_code, region_name in [("WE", "美西"), ("EA", "美东")]:
        region_df = source_df[source_df["region_norm"] == region_code]
        _append_delivery_breakdown_rows(rows, region_name, region_df, thresholds)
        if region_df.empty:
            continue

        for hub_name in sorted(region_df["Hub"].fillna("未知Hub").astype(str).str.strip().replace("", "未知Hub").unique()):
            hub_df = region_df[region_df["Hub"].fillna("未知Hub").astype(str).str.strip().replace("", "未知Hub") == hub_name]
            hub_display_name = format_unknown_dimension_name(hub_name, hub_df)
            _append_delivery_breakdown_rows(rows, f"  {hub_display_name}", hub_df, thresholds)

            contractor_series = hub_df["Contractor"].fillna("未知Contractor").astype(str).str.strip().replace("", "未知Contractor")
            for contractor_name in sorted(contractor_series.unique()):
                contractor_df = hub_df[contractor_series == contractor_name]
                contractor_display_name = format_unknown_dimension_name(contractor_name, contractor_df)
                _append_delivery_breakdown_rows(rows, f"    {contractor_display_name}", contractor_df, thresholds)

    table_df = pd.DataFrame(rows)
    percent_cols = [f"<{threshold}h妥投率" for threshold in thresholds]
    for col in percent_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.2%}")
    return table_df


def render_compact_kpi_row(kpi_payload: dict[str, Any]) -> None:
    # scan time - created time and check how many hours 
    delivered_24h = next((m for m in kpi_payload["metrics"] if m.get("指标") == "<24h 妥投率"), None)
    scan_24h = next((m for m in kpi_payload["metrics"] if m.get("指标") == "<24h 上网率"), None)
    lost_metric = next((m for m in kpi_payload["metrics"] if m.get("指标") == "整体月丢包率口径"), None)

    st.markdown(f"#### {tr('compact_title')}")
    c1, c2, c3 = st.columns(3)
    if delivered_24h:
        c1.metric("24小时妥投率", f"{delivered_24h['占比']:.2%}", f"{delivered_24h['命中']}/{delivered_24h['总数']}")
        render_percentage_pie(
            title="24小时妥投占比",
            hit_count=int(delivered_24h["命中"]),
            total_count=int(delivered_24h["总数"]),
            hit_label="<24h妥投",
            miss_label=">=24h或未妥投",
            chart_key="compact_delivered_24h",
            container=c1,
        )
    else:
        c1.metric("24小时妥投率", "0.00%", "0/0")
        c1.info("24小时妥投占比：暂无可用数据")

    if scan_24h:
        c2.metric("24小时上网率", f"{scan_24h['占比']:.2%}", f"{scan_24h['命中']}/{scan_24h['总数']}")
        render_percentage_pie(
            title="24小时上网占比",
            hit_count=int(scan_24h["命中"]),
            total_count=int(scan_24h["总数"]),
            hit_label="<24h上网",
            miss_label=">=24h或未上网",
            chart_key="compact_scan_24h",
            container=c2,
        )
    else:
        c2.metric("24小时上网率", "0.00%", "0/0")
        c2.info("24小时上网占比：暂无可用数据")

    if lost_metric:
        c3.metric("丢包率", f"{lost_metric['占比']:.2%}", f"{lost_metric['命中']}/{lost_metric['总数']}")
        render_percentage_pie(
            title="丢包占比",
            hit_count=int(lost_metric["命中"]),
            total_count=int(lost_metric["总数"]),
            hit_label="丢包",
            miss_label="未丢包",
            chart_key="compact_lost_rate",
            container=c3,
        )
    else:
        c3.metric("丢包率", "0.00%", "0/0")
        c3.info("丢包占比：暂无可用数据")

def render_daily_kpi_charts(result_df: pd.DataFrame) -> None:
    chart_df = result_df.copy()
    chart_df["_created_date"] = pd.to_datetime(chart_df["created_time"], errors="coerce").dt.date
    chart_df["_delivered_date"] = pd.to_datetime(chart_df["delivered_time"], errors="coerce").dt.date
    chart_df["_evaluation_weight"] = calculate_package_evaluation_weight(chart_df)

    created_count_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")
        .size()
        .rename("包裹总数")
        .reset_index()
        .sort_values("_created_date")
    )
    delivered_count_df = (
        chart_df[chart_df["_delivered_date"].notna()]
        .groupby("_delivered_date")
        .size()
        .rename("包裹总数")
        .reset_index()
        .sort_values("_delivered_date")
    )
    evaluation_weight_df = (
        chart_df[chart_df["_created_date"].notna()]
        .groupby("_created_date")["_evaluation_weight"]
        .sum()
        .rename("评价重量")
        .reset_index()
        .sort_values("_created_date")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {tr('daily_created_chart')}")
        if created_count_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.line_chart(created_count_df.set_index("_created_date")["包裹总数"])

    with c2:
        st.markdown(f"#### {tr('daily_delivered_chart')}")
        if delivered_count_df.empty:
            st.info(tr("kpi_empty"))
        else:
            st.line_chart(delivered_count_df.set_index("_delivered_date")["包裹总数"])

    st.markdown(f"#### {tr('daily_eval_weight_chart')}")
    if evaluation_weight_df.empty:
        st.info(tr("eval_weight_empty"))
    else:
        st.line_chart(evaluation_weight_df.set_index("_created_date")["评价重量"])

def render_kpi_charts(result_df: pd.DataFrame, layout_mode: str, fetch_reference_time: datetime | None = None) -> dict[str, Any]:
    st.subheader(tr("kpi_title"))
    if result_df.empty:
        st.info(tr("kpi_empty"))
        return {"metrics": [], "charts": [], "has_monthly_lost_data": False, "monthly_lost": pd.DataFrame()}

    kpi_payload = build_kpi_report_payload(result_df, fetch_reference_time=fetch_reference_time)
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
        st.markdown("##### 24小时妥投率明细")
        compact_breakdown_df = build_delivery_breakdown_table(delivered_detail_df, thresholds=[24])
        st.dataframe(style_breakdown_rows(compact_breakdown_df), use_container_width=True)
        return kpi_payload

    render_daily_kpi_charts(result_df)

    st.markdown("#### 24/48/72 小时妥投率（上网 -> 妥投）")
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
    delivered_metrics = [m for m in kpi_payload["metrics"] if m.get("分类") == "24/48/72 小时妥投率（上网 -> 妥投）"]
    for i, metric in enumerate(delivered_metrics):
        threshold = metric["指标"].replace("<", "").replace("h 妥投率", "")
        delivered_cols[i].metric(
            metric["指标"],
            f"{metric['占比']:.2%}",
            f"{metric['命中']}/{metric['总数']}",
        )
        render_percentage_pie(
            title=f"<{threshold}h 妥投占比",
            hit_count=int(metric["命中"]),
            total_count=int(metric["总数"]),
            hit_label=f"<{threshold}h妥投",
            miss_label=f">={threshold}h或未妥投",
            chart_key=f"delivered_{threshold}_{refresh_key}",
            container=delivered_cols[i],
        )

    st.markdown("#### 12/24/48/72 小时上网率（提货 -> 上网）")
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
    scan_metrics = [m for m in kpi_payload["metrics"] if m.get("分类") == "12/24/48/72 小时上网率（提货 -> 上网）"]
    for i, metric in enumerate(scan_metrics):
        threshold = metric["指标"].replace("<", "").replace("h 上网率", "")
        scan_cols[i].metric(
            metric["指标"],
            f"{metric['占比']:.2%}",
            f"{metric['命中']}/{metric['总数']}",
        )
        render_percentage_pie(
            title=f"<{threshold}h 上网占比",
            hit_count=int(metric["命中"]),
            total_count=int(metric["总数"]),
            hit_label=f"<{threshold}h上网",
            miss_label=f">={threshold}h或未上网",
            chart_key=f"scan_{threshold}_{refresh_key}",
            container=scan_cols[i],
        )

    st.markdown("#### 月丢包率（Last Scan 后 120h 内无后续轨迹，且排除未满 120h 运单）")
    monthly_lost_metric = next((m for m in kpi_payload["metrics"] if m.get("指标") == "整体月丢包率口径"), None)

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
        metric_cols[0].metric("整体月丢包率口径", f"{monthly_lost_metric['占比']:.2%}")
        metric_cols[1].download_button(
            tr("download_lost"),
            data=lost_detail_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"lost_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=lost_detail_df.empty,
        )
        render_percentage_pie(
            "丢包占比",
            int(monthly_lost_metric["命中"]),
            int(monthly_lost_metric["总数"]),
            hit_label="丢包",
            miss_label="未丢包",
            chart_key=f"lost_{refresh_key}",
        )
        st.markdown(f"##### {tr('lost_detail')}")
        if lost_detail_df.empty:
            st.info(tr("lost_empty"))
        else:
            st.dataframe(lost_detail_df, use_container_width=True)
    else:
        st.info(tr("lost_no_scan"))

    st.markdown("#### 月破损率（预留）")
    st.info("预留区域：月破损率指标待后续开发。")

    st.markdown("#### 拦截成功率（预留）")
    st.info("预留区域：拦截成功率指标待后续开发。")

    return kpi_payload


def process_tracking_ids(
    dedup_ids: list[str],
    receive_province_map: dict[str, str],
    sender_info_map: dict[str, dict[str, str]],
    progress_bar,
    status_text,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    failures: list[dict[str, str]] = []

    if not dedup_ids:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), failures

    total = len(dedup_ids)
    completed = 0
    thread_local = local()
    headers = build_api_headers()

    def worker(tracking_id: str) -> tuple[str, dict[str, str], dict[str, str] | None]:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()

        try:
            payload = fetch_tracking_data(tracking_id, thread_local.session, headers)
            row = build_row(tracking_id, payload)
            return tracking_id, row, None
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "N/A"
            return tracking_id, empty_row(tracking_id), {"tracking_id": tracking_id, "reason": f"HTTP {code}"}
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
    return pd.DataFrame(ordered_rows, columns=OUTPUT_COLUMNS), failures
    

# ---- MySQL (read from env; DO NOT hardcode secrets) ----
MYSQL_HOST = read_config("MYSQL_HOST", "")
MYSQL_PORT = int(read_config("MYSQL_PORT", "3306"))
MYSQL_USERNAME = read_config("MYSQL_USERNAME", "")
MYSQL_PASSWORD = read_config("MYSQL_PASSWORD", "")
MYSQL_DATABASE = read_config("MYSQL_DATABASE", "")

DB_FETCH_BATCH_SIZE = max(100, int(read_config("DB_FETCH_BATCH_SIZE", "5000")))

def _require_db_env() -> None:
    missing = []
    if not MYSQL_HOST:
        missing.append("MYSQL_HOST")
    if not MYSQL_USERNAME:
        missing.append("MYSQL_USERNAME")
    if not MYSQL_PASSWORD:
        missing.append("MYSQL_PASSWORD")
    if not MYSQL_DATABASE:
        missing.append("MYSQL_DATABASE")
    if missing:
        raise RuntimeError(f"MySQL 环境变量未配置：{', '.join(missing)}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tracking_numbers_by_date(start_date: date, end_date: date) -> list[str]:
    """
    Query waybill_waybills for tracking_number where created_at is between [start_date, end_date] inclusive.
    """
    _require_db_env()

    # lazy import so the app can still run without DB deps until this mode is used
    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 pymysql。请先 pip install pymysql") from e

    if end_date < start_date:
        return []

    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            sql = """
                SELECT DISTINCT tracking_number
                FROM waybill_waybills
                WHERE created_at >= %s AND created_at <= %s
                AND tracking_number IS NOT NULL AND tracking_number <> ''
                ORDER BY tracking_number ASC
            """
            cur.execute(sql, (start_date, end_date))

            tracking_numbers: list[str] = []
            while True:
                rows = cur.fetchmany(DB_FETCH_BATCH_SIZE)
                if not rows:
                    break
                tracking_numbers.extend(str(r["tracking_number"]).strip() for r in rows if r.get("tracking_number"))
            return tracking_numbers
    finally:
        conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_receive_province_map(tracking_ids: tuple[str, ...]) -> dict[str, str]:
    """
    Query waybill_waybills.receive_province by tracking_number for given tracking_ids.
    """
    _require_db_env()

    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 pymysql。请先 pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    receive_province_map: dict[str, str] = {}
    try:
        with conn.cursor() as cur:
            chunk_size = 500
            for i in range(0, len(tracking_ids_clean), chunk_size):
                chunk = tracking_ids_clean[i : i + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                sql = f"""
                    SELECT tracking_number, receive_province
                    FROM waybill_waybills
                    WHERE tracking_number IN ({placeholders})
                """
                cur.execute(sql, chunk)
                while True:
                    rows = cur.fetchmany(DB_FETCH_BATCH_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        tracking_number = str(row.get("tracking_number") or "").strip()
                        if not tracking_number:
                            continue
                        receive_province_map[tracking_number] = str(row.get("receive_province") or "").strip()
    finally:
        conn.close()

    return receive_province_map


@st.cache_data(ttl=60, show_spinner=False)
def fetch_sender_info_map(tracking_ids: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """
    Query sender fields from waybill_waybills for given tracking_ids.
    """
    _require_db_env()

    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 pymysql。请先 pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    sender_info_map: dict[str, dict[str, str]] = {}
    try:
        with conn.cursor() as cur:
            chunk_size = 500
            for i in range(0, len(tracking_ids_clean), chunk_size):
                chunk = tracking_ids_clean[i : i + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                sql = f"""
                    SELECT tracking_number, sender_company, sender_province, sender_city, sender_address
                    FROM waybill_waybills
                    WHERE tracking_number IN ({placeholders})
                """
                cur.execute(sql, chunk)
                while True:
                    rows = cur.fetchmany(DB_FETCH_BATCH_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        tracking_number = str(row.get("tracking_number") or "").strip()
                        if not tracking_number:
                            continue
                        sender_info_map[tracking_number] = {
                            "sender_company": str(row.get("sender_company") or "").strip(),
                            "sender_province": str(row.get("sender_province") or "").strip(),
                            "sender_city": str(row.get("sender_city") or "").strip(),
                            "sender_address": str(row.get("sender_address") or "").strip(),
                        }
    finally:
        conn.close()

    return sender_info_map

def main() -> None:
    st.set_page_config(page_title="Fimile美区运单运营数据分析系统", layout="wide")
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
    if "pod_compliance_map" not in st.session_state:
        st.session_state["pod_compliance_map"] = {}
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
    if "ofd_filter_start" not in st.session_state:
        st.session_state["ofd_filter_start"] = date.today() - timedelta(days=7)
    if "ofd_filter_end" not in st.session_state:
        st.session_state["ofd_filter_end"] = date.today()
    if "hide_unknown_dimensions" not in st.session_state:
        st.session_state["hide_unknown_dimensions"] = False
    if "apply_ofd_filter" not in st.session_state:
        st.session_state["apply_ofd_filter"] = True
        
    st.selectbox(
        tr("language_label"),
        options=["zh", "en"],
        format_func=lambda x: "中文" if x == "zh" else "English",
        key="language",
    )

    st.subheader(tr("input_section"))
    mode = st.radio(
        tr("input_mode"),
        [tr("mode_db"), tr("mode_file"), tr("mode_text"), tr("mode_text_db")],
        horizontal=True,
    )

    raw_ids: list[str] = []

    if mode == tr("mode_db"):
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input(tr("start_date"), value=date.today() - timedelta(days=1))
        with c2:
            end_d = st.date_input(tr("end_date"), value=date.today())

        btn = st.button(tr("load_btn"), type="primary")
        if btn:
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

        if not btn:
            raw_ids = st.session_state.get("db_raw_ids", [])

        if raw_ids:
            with st.expander(tr("db_preview", count=len(raw_ids)), expanded=False):
                st.write(raw_ids[:50])

    elif mode == tr("mode_file"):
        file = st.file_uploader(tr("upload_file"), type=["csv", "xlsx"])
        raw_ids = read_uploaded_ids(file)

    elif mode == tr("mode_text"):
        text = st.text_area(tr("paste_ids"), height=180, key="paste_text_only")
        raw_ids = split_text_ids(text)

    else:
        text = st.text_area(tr("paste_ids"), height=180, key="paste_text_with_db")
        pasted_ids = split_text_ids(text)

        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input(tr("start_date"), value=date.today() - timedelta(days=1), key="text_db_start_date")
        with c2:
            end_d = st.date_input(tr("end_date"), value=date.today(), key="text_db_end_date")

        merge_btn = st.button(tr("load_merge_btn"), type="primary")
        if merge_btn:
            with st.spinner(tr("loading_db")):
                try:
                    db_raw_ids = fetch_tracking_numbers_by_date(start_d, end_d)
                    st.session_state["db_raw_ids"] = db_raw_ids
                    if not db_raw_ids:
                        st.warning(tr("no_tracking_found"))
                except Exception as e:
                    st.error(str(e))
                    st.session_state["db_raw_ids"] = []

        db_raw_ids = st.session_state.get("db_raw_ids", [])
        raw_ids = pasted_ids + db_raw_ids

        if db_raw_ids:
            with st.expander(tr("db_preview", count=len(db_raw_ids)), expanded=False):
                st.write(db_raw_ids[:50])


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

        try:
            receive_province_map = fetch_receive_province_map(tuple(dedup_ids))
        except Exception as e:
            st.warning(tr("state_region_fail", error=e))

        try:
            sender_info_map = fetch_sender_info_map(tuple(dedup_ids))
        except Exception:
            sender_info_map = {}

        progress = st.progress(0)
        status = st.empty()

        result_df, failures = process_tracking_ids(
            dedup_ids=dedup_ids,
            receive_province_map=receive_province_map,
            sender_info_map=sender_info_map,
            progress_bar=progress,
            status_text=status,
        )

        result_df = fill_route_identity_columns(result_df)

        st.session_state["result_df"] = result_df
        st.session_state["failures"] = failures

        compliance_map: dict[str, bool] = {}
        for _, row in result_df.iterrows():
            tracking_id = str(row.get("tracking_id") or "")
            if not tracking_id:
                continue
            compliance_map[tracking_id] = auto_is_pod_compliant(row)
        st.session_state["pod_compliance_map"] = compliance_map

        status.text(tr("done"))

    result_df: pd.DataFrame | None = st.session_state.get("result_df")
    failures: list[dict[str, str]] = st.session_state.get("failures", [])

    if result_df is not None:
        st.subheader(tr("filter_view"))
        toggle_label = tr("show_unknown_btn") if st.session_state.get("hide_unknown_dimensions", False) else tr("hide_unknown_btn")
        if st.button(toggle_label):
            st.session_state["hide_unknown_dimensions"] = not st.session_state.get("hide_unknown_dimensions", False)
            
        ofd_series = pd.to_datetime(result_df["out_for_delivery_time"], errors="coerce")
        ofd_valid_dates = ofd_series.dropna().dt.date
        if not ofd_valid_dates.empty:
            default_ofd_start = ofd_valid_dates.min()
            default_ofd_end = ofd_valid_dates.max()
        else:
            default_ofd_start = date.today() - timedelta(days=7)
            default_ofd_end = date.today()

        current_ofd_start = st.session_state.get("ofd_filter_start", default_ofd_start)
        current_ofd_end = st.session_state.get("ofd_filter_end", default_ofd_end)
        current_ofd_start = max(default_ofd_start, min(current_ofd_start, default_ofd_end))
        current_ofd_end = max(default_ofd_start, min(current_ofd_end, default_ofd_end))
        if current_ofd_start > current_ofd_end:
            current_ofd_start = current_ofd_end
        st.session_state["ofd_filter_start"] = current_ofd_start
        st.session_state["ofd_filter_end"] = current_ofd_end

        ofd_c1, ofd_c2 = st.columns(2)
        with ofd_c1:
            ofd_start_date = st.date_input(
                tr("ofd_filter_start"),
                value=current_ofd_start,
                min_value=default_ofd_start,
                max_value=default_ofd_end,
                key="ofd_filter_start",
            )
        with ofd_c2:
            ofd_end_date = st.date_input(
                tr("ofd_filter_end"),
                value=current_ofd_end,
                min_value=default_ofd_start,
                max_value=default_ofd_end,
                key="ofd_filter_end",
            )

        st.checkbox(tr("apply_ofd_filter"), key="apply_ofd_filter")

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

        if ofd_start_date > ofd_end_date:
            ofd_start_date, ofd_end_date = ofd_end_date, ofd_start_date

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

        
        if st.session_state.get("apply_ofd_filter", True):
            filtered_df["_ofd_dt"] = pd.to_datetime(filtered_df["out_for_delivery_time"], errors="coerce")
            ofd_start_ts = pd.Timestamp(ofd_start_date)
            ofd_end_exclusive_ts = pd.Timestamp(ofd_end_date)
            if ofd_end_exclusive_ts <= ofd_start_ts:
                ofd_end_exclusive_ts = ofd_start_ts + pd.Timedelta(days=1)
            filtered_df = filtered_df[
                filtered_df["_ofd_dt"].notna()
                & (filtered_df["_ofd_dt"] >= ofd_start_ts)
                & (filtered_df["_ofd_dt"] < ofd_end_exclusive_ts)
            ].drop(columns=["_ofd_dt"])

        non_pickup_filtered_df, pickup_filtered_df = split_pickup_routes(filtered_df)

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

        st.subheader(tr("pickup_section"))
        if pickup_filtered_df.empty:
            st.info(tr("pickup_empty"))
        else:
            pickup_display_cols = [
                "tracking_id",
                "Region",
                "State",
                "Driver",
                "Hub",
                "Contractor",
                "Route_name",
                "out_for_delivery_time",
                "delivered_time",
            ]
            pickup_display_df = pickup_filtered_df[[c for c in pickup_display_cols if c in pickup_filtered_df.columns]].copy()
            st.dataframe(pickup_display_df, use_container_width=True)
            st.download_button(
                tr("download_pickup"),
                data=pickup_display_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"pickup_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        delivered_df = non_pickup_filtered_df[non_pickup_filtered_df["delivered_time"].astype(str).str.strip() != ""].copy()
        if not delivered_df.empty:
            delivered_df["_delivered_dt"] = pd.to_datetime(delivered_df["delivered_time"], errors="coerce")
            delivered_df = delivered_df.sort_values(by=["_delivered_dt", "tracking_id"], ascending=[False, True]).drop(columns=["_delivered_dt"])

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_df = build_export_df(filtered_df)
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        kpi_report_data = None
        c_csv, c_report = st.columns(2)
        c_csv.download_button(
            tr("download_csv"),
            data=csv_data,
            file_name=f"export_{stamp}.csv",
            mime="text/csv",
        )
        try:
            kpi_report_data = kpi_report_to_excel_bytes(kpi_payload, export_df)
            c_report.download_button(
                tr("download_report"),
                data=kpi_report_data,
                file_name=f"kpi_report_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            c_report.warning(tr("report_dep_missing"))

        render_compliance_section(tr("pod_review"), delivered_df, "pod_review")
        render_compliance_section(tr("delivered"), delivered_df, "delivered")


if __name__ == "__main__":
    main()
