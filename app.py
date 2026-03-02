import io
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st


# API configuration comes from code/env only (not exposed in UI).
API_URL_TEMPLATE = os.getenv(
    "KPI_API_URL_TEMPLATE",
    "https://isp.beans.ai/enterprise/v1/lists/status_logs"
    "?tracking_id={tracking_id}&readable=true"
    "&include_pod=true&include_item=true",
)
API_TOKEN = os.getenv("KPI_API_TOKEN", "")
API_TIMEOUT_SECONDS = int(os.getenv("KPI_API_TIMEOUT_SECONDS", "20"))

# How many POD images to export per tracking_id (each image can have its own quality.feedback/score)
POD_IMAGE_EXPORT_N = int(os.getenv("POD_IMAGE_EXPORT_N", "5"))

POD_COLUMNS: list[str] = []
for i in range(1, POD_IMAGE_EXPORT_N + 1):
    POD_COLUMNS += [f"pod_feedback_{i}", f"pod_score_{i}"]

OUTPUT_COLUMNS = [
    "trakcing_id",
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

    # Fallback: flatten all cells, keep non-empty values.
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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    """
    Try to extract POD images list from the "success/delivered" event.

    Common shapes (based on sample responses):
      success_evt["pod"]["images"] -> list
      success_evt["pods"]["pod"][0]["images"] -> list
    """
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

    # --- POD feedback / score columns ---
    # initialize columns to empty (important when there are fewer than N images)
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


def fetch_tracking_data(tracking_id: str, session: requests.Session) -> dict[str, Any]:
    if not API_URL_TEMPLATE:
        raise RuntimeError("KPI_API_URL_TEMPLATE 未配置")

    if "{tracking_id}" in API_URL_TEMPLATE:
        url = API_URL_TEMPLATE.format(tracking_id=tracking_id)
    else:
        parsed = urlparse(API_URL_TEMPLATE)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["tracking_id"] = tracking_id
        url = urlunparse(parsed._replace(query=urlencode(query)))

    headers = {"Accept": "application/json"}
    if API_TOKEN:
        token = API_TOKEN.strip()
        if token.lower().startswith("basic ") or token.lower().startswith("bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"

    response = session.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="export")
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Tracking Export", layout="wide")
    st.title("Tracking Export")

    if "dedup_ids" not in st.session_state:
        st.session_state["dedup_ids"] = []
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "failures" not in st.session_state:
        st.session_state["failures"] = []

    st.subheader("1) 输入 Tracking IDs")
    mode = st.radio("输入方式", ["上传文件", "文本粘贴"], horizontal=True)

    raw_ids: list[str] = []
    if mode == "上传文件":
        file = st.file_uploader("上传 CSV 或 XLSX", type=["csv", "xlsx"])
        raw_ids = read_uploaded_ids(file)
    else:
        text = st.text_area("粘贴 Tracking IDs（支持换行/逗号/空格分隔）", height=180)
        raw_ids = split_text_ids(text)

    cleaned, dedup_ids, counter = normalize_tracking_ids(raw_ids, uppercase=False)
    duplicate_ids = [k for k, v in counter.items() if v > 1]
    st.session_state["dedup_ids"] = dedup_ids

    c1, c2, c3 = st.columns(3)
    c1.metric("input_count", len(cleaned))
    c2.metric("unique_count", len(dedup_ids))
    c3.metric("duplicate_count", len(cleaned) - len(dedup_ids))

    if duplicate_ids:
        with st.expander("重复 Tracking IDs"):
            st.write(duplicate_ids)

    st.subheader("2) 调用 API 并导出")
    if st.button("Fetch / Export", type="primary", disabled=not dedup_ids):
        rows_by_id: dict[str, dict[str, str]] = {}
        failures: list[dict[str, str]] = []

        progress = st.progress(0)
        status = st.empty()

        with requests.Session() as session:
            total = len(dedup_ids)
            for idx, tracking_id in enumerate(dedup_ids, start=1):
                status.text(f"处理中：{idx}/{total} - {tracking_id}")
                try:
                    payload = fetch_tracking_data(tracking_id, session)
                    rows_by_id[tracking_id] = build_row(tracking_id, payload)
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else "N/A"
                    reason = f"HTTP {code}"
                    failures.append({"tracking_id": tracking_id, "reason": reason})
                    rows_by_id[tracking_id] = empty_row(tracking_id)
                except Exception as e:  # noqa: BLE001
                    failures.append({"tracking_id": tracking_id, "reason": str(e)})
                    rows_by_id[tracking_id] = empty_row(tracking_id)

                progress.progress(idx / total)

        ordered_rows = [rows_by_id[tid] for tid in dedup_ids]
        result_df = pd.DataFrame(ordered_rows, columns=OUTPUT_COLUMNS)

        st.session_state["result_df"] = result_df
        st.session_state["failures"] = failures
        status.text("处理完成")

    result_df: pd.DataFrame | None = st.session_state.get("result_df")
    failures: list[dict[str, str]] = st.session_state.get("failures", [])

    if result_df is not None:
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

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
        xlsx_data = None
        try:
            xlsx_data = df_to_excel_bytes(result_df)
        except Exception:
            st.warning("当前环境缺少 Excel 依赖，已提供 CSV 下载。")

        c_csv, c_xlsx = st.columns(2)
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


if __name__ == "__main__":
    main()
