import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from threading import local
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st

# ---- MySQL (read from env; DO NOT hardcode secrets) ----
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")

# API configuration comes from code/env only (not exposed in UI).
API_URL_TEMPLATE = os.getenv(
    "KPI_API_URL_TEMPLATE",
    "https://isp.beans.ai/enterprise/v1/lists/status_logs"
    "?tracking_id={tracking_id}&readable=true"
    "&include_pod=true&include_item=true",
)
API_TOKEN = os.getenv("KPI_API_TOKEN", "")
API_TIMEOUT_SECONDS = int(os.getenv("KPI_API_TIMEOUT_SECONDS", "20"))
API_MAX_WORKERS = max(1, int(os.getenv("KPI_API_MAX_WORKERS", "12")))
DB_FETCH_BATCH_SIZE = max(100, int(os.getenv("DB_FETCH_BATCH_SIZE", "5000")))

# How many POD images to export per tracking_id (each image can have its own quality.feedback/score)
POD_IMAGE_EXPORT_N = int(os.getenv("POD_IMAGE_EXPORT_N", "5"))

POD_COLUMNS: list[str] = []
for i in range(1, POD_IMAGE_EXPORT_N + 1):
    POD_COLUMNS += [f"pod_feedback_{i}", f"pod_score_{i}"]

OUTPUT_COLUMNS = [
    "trakcing_id",
    "Region",
    "State",
    "shipperName",
    "created_time",
    "first_scanned_time",
    "last_scanned_time",
    "out_for_delivery_time",
    "attempted_time",
    "failed_route",
    "delivered_time",
    "success_route",
    *POD_COLUMNS,
    "创建到入库时间",
    "库内停留时间",
    "尝试配送时间",
    "送达时间",
    "整体配送时间",
]

REGION_BY_STATE = {
    "CA": "WE",
    "TX": "WE",
    "IL": "WE",
    "NJ": "EA",
    "GA": "EA",
    "FL": "EA",
}

STATE_ALIAS = {
    "NY": "NJ",
}

def count_pod_stats(row: dict[str, str] | pd.Series) -> tuple[int, int]:
    pod_count = 0
    scored_count = 0
    for i in range(1, POD_IMAGE_EXPORT_N + 1):
        feedback = str(row.get(f"pod_feedback_{i}") or "").strip()
        score = str(row.get(f"pod_score_{i}") or "").strip()
        if feedback or score:
            pod_count += 1
        if score:
            scored_count += 1
    return pod_count, scored_count


def auto_is_pod_compliant(row: dict[str, str] | pd.Series) -> bool:
    pod_count, scored_count = count_pod_stats(row)
    return pod_count >= 3 and scored_count >= 2


def build_beans_tracking_link(tracking_id: str) -> str:
    return f"https://www.beansroute.ai/3pl-manager/tabs.html#searchTrackingId/{tracking_id}"


def render_compliance_section(title: str, delivered_df: pd.DataFrame, state_key_prefix: str) -> None:
    st.subheader(title)
    if delivered_df.empty:
        st.info("暂无已妥投运单。")
        return

    header_cols = st.columns([2, 2, 5, 2])
    header_cols[0].markdown("**妥投日期（Delivered Date）**")
    header_cols[1].markdown("**运单号（Tracking ID）**")
    header_cols[2].markdown("**Beans 运单查询（查看 POD）**")
    header_cols[3].markdown("**是否标记为合规**")

    for idx, row in delivered_df.iterrows():
        tracking_id = str(row.get("trakcing_id") or "")
        delivered_time = str(row.get("delivered_time") or "")
        compliant = st.session_state["pod_compliance_map"].get(tracking_id, False)

        cols = st.columns([2, 2, 5, 2])
        cols[0].write(delivered_time)
        cols[1].write(tracking_id)
        cols[2].markdown(f"[打开 Beans 查看 POD]({build_beans_tracking_link(tracking_id)})")
        btn_label = "✅" if compliant else "❌"
        if cols[3].button(btn_label, key=f"{state_key_prefix}_toggle_{idx}_{tracking_id}", use_container_width=True):
            st.session_state["pod_compliance_map"][tracking_id] = not compliant
            st.rerun()


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


def read_uploaded_ids(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, dtype=str)
        else:
            return []
    except Exception:
        return []

    if df.empty:
        return []

    preferred = [c for c in df.columns if str(c).lower() in {"tracking_id", "trackingid", "trakcing_id"}]
    if preferred:
        series = df[preferred[0]].dropna()
        return series.astype(str).tolist()

    values: list[str] = []
    for col in df.columns:
        values.extend(df[col].dropna().astype(str).tolist())
    return values


def to_local_dt(ts_millis: Any) -> datetime | None:
    if ts_millis is None:
        return None
    try:
        millis = int(ts_millis)
        return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).astimezone()
    except (ValueError, TypeError, OSError):
        return None


def fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def diff_hours(end_dt: datetime | None, start_dt: datetime | None) -> str:
    if not end_dt or not start_dt:
        return ""
    return f"{(end_dt - start_dt).total_seconds() / 3600:.2f}"


def parse_route(description: Any) -> str:
    text = "" if description is None else str(description)
    match = re.search(r"route[:：\s-]*(.+)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


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


def event_type(event: dict[str, Any]) -> str:
    for key in ("type", "eventType", "status"):
        val = event.get(key)
        if val:
            return str(val).strip().lower().replace("_", "-")

    log_item = event.get("logItem")
    if isinstance(log_item, dict):
        for key in ("type", "eventType", "status"):
            val = log_item.get(key)
            if val:
                return str(val).strip().lower().replace("_", "-")

    log_obj = event.get("log")
    if isinstance(log_obj, dict):
        for key in ("type", "eventType", "status"):
            val = log_obj.get(key)
            if val:
                return str(val).strip().lower().replace("_", "-")
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


def first_event_by_predicate(events: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    filtered = [e for e in events if predicate(e)]
    if not filtered:
        return None

    with_ts = [(event_ts(e), idx, e) for idx, e in enumerate(filtered)]
    with_ts.sort(key=lambda x: (10**18 if x[0] is None else x[0], x[1]))
    return with_ts[0][2]


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


def extract_pod_images_from_success_event(success_evt: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not success_evt:
        return []

    pod_obj = success_evt.get("pod")
    if isinstance(pod_obj, dict):
        images = pod_obj.get("images")
        if isinstance(images, list):
            return [x for x in images if isinstance(x, dict)]

    pods_obj = success_evt.get("pods")
    if isinstance(pods_obj, dict):
        pod_list = pods_obj.get("pod")
        if isinstance(pod_list, list) and pod_list:
            first_pod = pod_list[0]
            if isinstance(first_pod, dict):
                images = first_pod.get("images")
                if isinstance(images, list):
                    return [x for x in images if isinstance(x, dict)]

    return []


def build_row(tracking_id: str, payload: dict[str, Any]) -> dict[str, str]:
    events = normalize_events(payload)
    shipper_name = extract_shipper_name_from_events(events)

    created_evt = first_event_by_predicate(events, lambda e: event_type(e) == "label")
    scanned_predicate = lambda e: (
        (desc := str(e.get("description", "")).strip().lower()).startswith("scan at")
        or desc.startswith("scanned at")
    )
    first_scanned_evt = first_event_by_predicate(events, scanned_predicate)
    last_scanned_evt = last_event_by_predicate(events, scanned_predicate)

    ofd_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"out-for-delivery", "ofd", "outfordelivery"})
    fail_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"fail", "failed", "failure"})
    success_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"success", "delivered"})

    created_time = to_local_dt(event_ts(created_evt) if created_evt else None)
    first_scanned_time = to_local_dt(event_ts(first_scanned_evt) if first_scanned_evt else None)
    last_scanned_time = to_local_dt(event_ts(last_scanned_evt) if last_scanned_evt else None)
    out_for_delivery_time = to_local_dt(event_ts(ofd_evt) if ofd_evt else None)
    attempted_time = to_local_dt(event_ts(fail_evt) if fail_evt else None)
    delivered_time = to_local_dt(event_ts(success_evt) if success_evt else None)

    row: dict[str, str] = {
        "trakcing_id": tracking_id,
        "shipperName": str(
            shipper_name
            or payload.get("shipperName")
            or payload.get("data", {}).get("shipperName")
            or payload.get("result", {}).get("shipperName")
            or payload.get("response", {}).get("shipperName")
            or ""
        ),
        "created_time": fmt_dt(created_time),
        "first_scanned_time": fmt_dt(first_scanned_time),
        "last_scanned_time": fmt_dt(last_scanned_time),
        "out_for_delivery_time": fmt_dt(out_for_delivery_time),
        "attempted_time": fmt_dt(attempted_time),
        "failed_route": parse_route(fail_evt.get("description")) if fail_evt else "",
        "delivered_time": fmt_dt(delivered_time),
        "success_route": parse_route(success_evt.get("description")) if success_evt else "",
        "创建到入库时间": diff_hours(first_scanned_time, created_time),
        "库内停留时间": diff_hours(out_for_delivery_time, first_scanned_time),
        "尝试配送时间": diff_hours(attempted_time, out_for_delivery_time),
        "送达时间": diff_hours(delivered_time, out_for_delivery_time),
        "整体配送时间": diff_hours(delivered_time, created_time),
    }

    for i in range(1, POD_IMAGE_EXPORT_N + 1):
        row[f"pod_feedback_{i}"] = ""
        row[f"pod_score_{i}"] = ""

    pod_images = extract_pod_images_from_success_event(success_evt)
    for i, img in enumerate(pod_images[:POD_IMAGE_EXPORT_N], start=1):
        q = img.get("quality")
        if not isinstance(q, dict):
            continue
        row[f"pod_feedback_{i}"] = str(q.get("feedback") or "")
        score_val = q.get("score")
        row[f"pod_score_{i}"] = "" if score_val is None else str(score_val)

    return row


def empty_row(tracking_id: str) -> dict[str, str]:
    row = {col: "" for col in OUTPUT_COLUMNS}
    row["trakcing_id"] = tracking_id
    return row

def normalize_state(state: str) -> str:
    normalized_state = str(state or "").strip().upper()
    if not normalized_state:
        return ""

    compact_state = re.sub(r"[^A-Z]", "", normalized_state)
    return STATE_ALIAS.get(normalized_state, STATE_ALIAS.get(compact_state, normalized_state))


def infer_region_from_state(state: str) -> str:
    normalized_state = normalize_state(state)
    return REGION_BY_STATE.get(normalized_state, "")



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


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="export")
    return output.getvalue()


def build_kpi_report_payload(result_df: pd.DataFrame) -> dict[str, Any]:
    df = result_df.copy()
    df["created_dt"] = to_datetime_series(df, "created_time")
    df["first_scanned_dt"] = to_datetime_series(df, "first_scanned_time")
    df["last_scanned_dt"] = to_datetime_series(df, "last_scanned_time")
    df["ofd_dt"] = to_datetime_series(df, "out_for_delivery_time")
    df["attempted_dt"] = to_datetime_series(df, "attempted_time")
    df["delivered_dt"] = to_datetime_series(df, "delivered_time")
    df["month"] = df["created_dt"].dt.to_period("M").astype(str)
    df.loc[df["month"] == "NaT", "month"] = "未知"

    metrics: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []

    df["ofd_to_delivered_hours"] = (df["delivered_dt"] - df["ofd_dt"]).dt.total_seconds() / 3600
    ofd_base = df[df["ofd_dt"].notna()].copy()

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

    has_followup = (
        (df["last_scanned_dt"].notna() & (df["last_scanned_dt"] > df["first_scanned_dt"]))
        | df["ofd_dt"].notna()
        | df["attempted_dt"].notna()
        | df["delivered_dt"].notna()
    )
    scanned_base = df[df["first_scanned_dt"].notna()].copy()
    scanned_base["lost"] = (~has_followup.loc[scanned_base.index]).astype(int)
    monthly_lost = scanned_base.groupby("month", as_index=False).agg(total=("trakcing_id", "count"), lost=("lost", "sum"))
    lost_total = int(monthly_lost["lost"].sum()) if not monthly_lost.empty else 0
    scanned_total = int(monthly_lost["total"].sum()) if not monthly_lost.empty else 0
    metrics.append(
        {
            "分类": "月丢包率（First Scan 后无后续轨迹）",
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


def kpi_report_to_excel_bytes(kpi_payload: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    metrics_df = pd.DataFrame(kpi_payload["metrics"])
    chart_df = pd.DataFrame(kpi_payload["charts"])

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="kpi_summary")
        chart_df.to_excel(writer, index=False, sheet_name="kpi_chart_data")

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


def to_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[column], errors="coerce")


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def render_percentage_pie(
    title: str,
    hit_count: int,
    total_count: int,
    hit_label: str = "达标",
    miss_label: str = "未达标",
) -> None:
    if total_count <= 0:
        st.info(f"{title}：暂无可用数据")
        return

    miss_count = max(total_count - hit_count, 0)
    chart_df = pd.DataFrame({"分类": [hit_label, miss_label], "数量": [hit_count, miss_count]})
    chart_df = chart_df[chart_df["数量"] > 0]
    chart_df["占比"] = (chart_df["数量"] / total_count).map(lambda x: f"{x:.2%}")

    st.caption(title)
    st.vega_lite_chart(
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
    )


def render_kpi_charts(result_df: pd.DataFrame) -> dict[str, Any]:
    st.subheader("3) 时效与质量 KPI 图表")
    if result_df.empty:
        st.info("暂无数据，无法计算 KPI。")
        return {"metrics": [], "charts": [], "has_monthly_lost_data": False, "monthly_lost": pd.DataFrame()}

    kpi_payload = build_kpi_report_payload(result_df)

    st.markdown("#### 24/48/72 小时妥投率（上网 -> 妥投）")
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
        )

    st.markdown("#### 12/24/48/72 小时上网率（提货 -> 上网）")
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
        )

    st.markdown("#### 月丢包率（First Scan 后无后续轨迹）")
    monthly_lost_metric = next((m for m in kpi_payload["metrics"] if m.get("指标") == "整体月丢包率口径"), None)
    if kpi_payload.get("has_monthly_lost_data") and monthly_lost_metric:
        st.metric("整体月丢包率口径", f"{monthly_lost_metric['占比']:.2%}")
        render_percentage_pie(
            "丢包占比",
            int(monthly_lost_metric["命中"]),
            int(monthly_lost_metric["总数"]),
            hit_label="丢包",
            miss_label="未丢包",
        )
    else:
        st.info("没有 First Scan 数据，无法计算月丢包率。")

    st.markdown("#### 月破损率（预留）")
    st.info("预留区域：月破损率指标待后续开发。")

    st.markdown("#### 拦截成功率（预留）")
    st.info("预留区域：拦截成功率指标待后续开发。")

    return kpi_payload


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


def process_tracking_ids(
    dedup_ids: list[str],
    receive_province_map: dict[str, str],
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
            rows_by_id[tracking_id] = row

            if failure:
                failures.append(failure)

            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"处理中：{completed}/{total} - {tracking_id}")

    ordered_rows = [rows_by_id[tid] for tid in dedup_ids]
    return pd.DataFrame(ordered_rows, columns=OUTPUT_COLUMNS), failures


def main() -> None:
    st.set_page_config(page_title="Tracking Export", layout="wide")
    st.title("Tracking Export")

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

    st.subheader("1) 输入 Tracking IDs")
    mode = st.radio("输入方式", ["数据库按日期", "上传文件", "文本粘贴"], horizontal=True)

    raw_ids: list[str] = []

    if mode == "数据库按日期":
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input("起始日期 (Created_at)", value=date.today() - timedelta(days=1))
        with c2:
            end_d = st.date_input("结束日期 (Created_at)", value=date.today())

        btn = st.button("从数据库加载运单号", type="primary")
        if btn:
            with st.spinner("查询数据库中..."):
                try:
                    raw_ids = fetch_tracking_numbers_by_date(start_d, end_d)
                    st.session_state["db_raw_ids"] = raw_ids
                    if not raw_ids:
                        st.warning("该日期范围内未找到任何 tracking_number")
                except Exception as e:
                    st.error(str(e))
                    raw_ids = []
                    st.session_state["db_raw_ids"] = []

        if not btn:
            raw_ids = st.session_state.get("db_raw_ids", [])

        if raw_ids:
            with st.expander(f"数据库返回运单号预览（前 50 / 共 {len(raw_ids)}）", expanded=False):
                st.write(raw_ids[:50])

    elif mode == "上传文件":
        file = st.file_uploader("上传 CSV 或 XLSX", type=["csv", "xlsx"])
        raw_ids = read_uploaded_ids(file)

    else:
        text = st.text_area("粘贴 Tracking IDs（支持换行/逗号/空格分隔）", height=180)
        raw_ids = split_text_ids(text)

    cleaned, dedup_ids, counter = normalize_tracking_ids(raw_ids, uppercase=False)
    duplicate_ids = [k for k, v in counter.items() if v > 1]
    st.session_state["dedup_ids"] = dedup_ids

    m1, m2, m3 = st.columns(3)
    m1.metric("input_count", len(cleaned))
    m2.metric("unique_count", len(dedup_ids))
    m3.metric("duplicate_count", len(cleaned) - len(dedup_ids))

    if duplicate_ids:
        with st.expander("重复 Tracking IDs"):
            st.write(duplicate_ids)

    st.subheader("2) 调用 API 并导出")
    if st.button("Fetch / Export", type="primary", disabled=not dedup_ids):
        failures: list[dict[str, str]] = []
        receive_province_map: dict[str, str] = {}

        try:
            receive_province_map = fetch_receive_province_map(tuple(dedup_ids))
        except Exception as e:
            st.warning(f"读取 State/Region 数据失败，将导出为空值：{e}")

        progress = st.progress(0)
        status = st.empty()

        result_df, failures = process_tracking_ids(
            dedup_ids=dedup_ids,
            receive_province_map=receive_province_map,
            progress_bar=progress,
            status_text=status,
        )

        st.session_state["result_df"] = result_df
        st.session_state["failures"] = failures

        compliance_map: dict[str, bool] = {}
        for _, row in result_df.iterrows():
            tracking_id = str(row.get("trakcing_id") or "")
            if not tracking_id:
                continue
            compliance_map[tracking_id] = auto_is_pod_compliant(row)
        st.session_state["pod_compliance_map"] = compliance_map

        status.text("处理完成")

    result_df: pd.DataFrame | None = st.session_state.get("result_df")
    failures: list[dict[str, str]] = st.session_state.get("failures", [])

    if result_df is not None:
        kpi_payload = render_kpi_charts(result_df)

        success_count = len(result_df) - len(failures)
        fail_count = len(failures)

        s1, s2 = st.columns(2)
        s1.metric("成功数量", success_count)
        s2.metric("失败数量", fail_count)

        if failures:
            st.error("以下 tracking_id 请求失败")
            fail_df = pd.DataFrame(failures)
            st.dataframe(fail_df, use_container_width=True)
            st.download_button(
                "下载失败列表 CSV",
                data=fail_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.subheader("结果预览")
        st.dataframe(result_df.head(50), use_container_width=True)

        delivered_df = result_df[result_df["delivered_time"].astype(str).str.strip() != ""].copy()
        if not delivered_df.empty:
            delivered_df["_delivered_dt"] = pd.to_datetime(delivered_df["delivered_time"], errors="coerce")
            delivered_df = delivered_df.sort_values(by=["_delivered_dt", "trakcing_id"], ascending=[False, True]).drop(columns=["_delivered_dt"])

        render_compliance_section("POD审核", delivered_df, "pod_review")
        render_compliance_section("已妥投", delivered_df, "delivered")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
        xlsx_data = None
        try:
            xlsx_data = df_to_excel_bytes(result_df)
        except Exception:
            st.warning("当前环境缺少 Excel 依赖，已提供 CSV 下载。")

        c_csv, c_xlsx, c_report = st.columns(3)
        c_csv.download_button(
            "下载 CSV",
            data=csv_data,
            file_name=f"export_{stamp}.csv",
            mime="text/csv",
        )
        if xlsx_data is not None:
            c_xlsx.download_button(
                "下载 Excel",
                data=xlsx_data,
                file_name=f"export_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        try:
            kpi_report_data = kpi_report_to_excel_bytes(kpi_payload)
            c_report.download_button(
                "下载数据报表（百分比+图表）",
                data=kpi_report_data,
                file_name=f"kpi_report_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            c_report.warning("当前环境缺少图表报表依赖，无法导出 KPI 报表。")


if __name__ == "__main__":
    main()
