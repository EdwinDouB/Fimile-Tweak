import io
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config (env only)
# -----------------------------
# Per-tracking API (status logs)
API_URL_TEMPLATE = os.getenv(
    "KPI_API_URL_TEMPLATE",
    "https://isp.beans.ai/enterprise/v1/lists/status_logs"
    "?tracking_id={tracking_id}&readable=true"
    "&include_pod=true&include_item=true",
)
API_TOKEN = os.getenv("KPI_API_TOKEN", "")
API_TIMEOUT_SECONDS = int(os.getenv("KPI_API_TIMEOUT_SECONDS", "20"))

# List APIs for mapping (warehouses / DSP companies)
LIST_BASE_URL = os.getenv("KPI_LIST_BASE_URL", "https://isp.beans.ai/enterprise/v1/lists").rstrip("/")

WAREHOUSES_URL = os.getenv("KPI_WAREHOUSES_URL", f"{LIST_BASE_URL}/warehouses?updatedAfter=0")
COMPANIES_URL = os.getenv("KPI_COMPANIES_URL", f"{LIST_BASE_URL}/thirdparty_companies?updatedAfter=0")

# If your API requires additional params/cookies, you can extend headers or envs as needed.

OUTPUT_COLUMNS = [
    "trakcing_id",
    "shipperName",
    "dsp_name",
    "warehouse_name",
    "created_time",
    "first_scanned_time",
    "last_scanned_time",
    "out_for_delivery_time",
    "attempted_time",
    "failed_route",
    "delivered_time",
    "success_route",
    "创建到入库时间",
    "库内停留时间",
    "尝试配送时间",
    "送达时间",
    "整体配送时间",
]


# -----------------------------
# Helpers: input parsing
# -----------------------------
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


# -----------------------------
# Helpers: time parsing
# -----------------------------
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


# -----------------------------
# Helpers: JSON traversal (for DSP/warehouse ID extraction)
# -----------------------------
def iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield all dict nodes in a nested structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_dicts(x)


def find_first_key(obj: Any, keys: set[str]) -> Any:
    """Find first occurrence of any key in keys (case-sensitive) in nested dict/list."""
    for d in iter_dicts(obj):
        for k in keys:
            if k in d and d.get(k) not in (None, "", []):
                return d.get(k)
    return None


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


# -----------------------------
# Events normalization (existing logic)
# -----------------------------
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


# -----------------------------
# Auth / HTTP
# -----------------------------
def build_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if API_TOKEN:
        token = API_TOKEN.strip()
        if token.lower().startswith(("basic ", "bearer ")):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str, session: requests.Session) -> Any:
    resp = session.get(url, headers=build_headers(), timeout=API_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


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

    payload = fetch_json(url, session)
    if isinstance(payload, dict):
        return payload
    return {"raw": payload}


# -----------------------------
# Build mapping tables: warehouse / DSP company
# -----------------------------
def normalize_list_root(payload: Any) -> Any:
    """Try to unwrap common envelopes."""
    if not isinstance(payload, dict):
        return payload
    root = payload
    for key in ("data", "result", "response"):
        if isinstance(root.get(key), (dict, list)):
            root = root[key]
            break
    return root


def list_candidates(root: Any) -> list[Any] | None:
    if isinstance(root, list):
        return root
    if isinstance(root, dict):
        # common list keys
        for k in (
            "warehouses",
            "thirdparty_companies",
            "companies",
            "items",
            "list",
            "rows",
            "data",
            "result",
        ):
            v = root.get(k)
            if isinstance(v, list):
                return v
    return None


def build_warehouse_map(payload: Any) -> dict[str, str]:
    root = normalize_list_root(payload)
    items = list_candidates(root) or []
    out: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        wid = safe_str(it.get("listWarehouseId") or it.get("warehouseId") or it.get("id"))
        name = safe_str(it.get("name") or it.get("warehouseName") or it.get("formattedAddress") or it.get("address"))
        if wid and name:
            out[wid] = name
    return out


def build_company_map(payload: Any) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      - company_id -> companyName
      - service_id -> companyName (if present)
    """
    root = normalize_list_root(payload)
    items = list_candidates(root) or []
    by_company_id: dict[str, str] = {}
    by_service_id: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = safe_str(it.get("companyName") or it.get("name"))
        cid = safe_str(it.get("companyId") or it.get("thirdpartyCompanyId") or it.get("id"))
        sid = safe_str(it.get("serviceId"))
        if name and cid:
            by_company_id[cid] = name
        if name and sid:
            by_service_id[sid] = name
    return by_company_id, by_service_id


# -----------------------------
# Extract DSP / warehouse IDs from status_logs payload
# -----------------------------
WAREHOUSE_ID_KEYS = {
    "listWarehouseId",
    "warehouseId",
    "defaultWarehouseId",
}
COMPANY_ID_KEYS = {
    "companyId",
    "thirdpartyCompanyId",
    "thirdPartyCompanyId",
}
SERVICE_ID_KEYS = {
    "serviceId",
}


def extract_warehouse_id(payload: dict[str, Any], events: list[dict[str, Any]]) -> str:
    # Try events first (often richer)
    wid = find_first_key(events, WAREHOUSE_ID_KEYS)
    if wid is None:
        wid = find_first_key(payload, WAREHOUSE_ID_KEYS)
    return safe_str(wid)


def extract_company_or_service_id(payload: dict[str, Any], events: list[dict[str, Any]]) -> tuple[str, str]:
    cid = find_first_key(events, COMPANY_ID_KEYS)
    sid = find_first_key(events, SERVICE_ID_KEYS)

    if cid is None:
        cid = find_first_key(payload, COMPANY_ID_KEYS)
    if sid is None:
        sid = find_first_key(payload, SERVICE_ID_KEYS)

    return safe_str(cid), safe_str(sid)


# -----------------------------
# Row building
# -----------------------------
def build_row(
    tracking_id: str,
    payload: dict[str, Any],
    warehouse_map: dict[str, str],
    company_map_by_company_id: dict[str, str],
    company_map_by_service_id: dict[str, str],
) -> dict[str, str]:
    events = normalize_events(payload)
    shipper_name = extract_shipper_name_from_events(events)

    created_evt = first_event_by_predicate(events, lambda e: event_type(e) == "label")
    scanned_predicate = lambda e: (
        (desc := str(e.get("description", "")).strip().lower()).startswith("scan at")
        or desc.startswith("scanned at")
    )
    first_scanned_evt = first_event_by_predicate(events, scanned_predicate)
    last_scanned_evt = last_event_by_predicate(events, scanned_predicate)
    ofd_evt = first_event_by_predicate(
        events, lambda e: event_type(e) in {"out-for-delivery", "ofd", "outfordelivery"}
    )
    fail_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"fail", "failed", "failure"})
    success_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"success", "delivered"})

    created_time = to_local_dt(event_ts(created_evt) if created_evt else None)
    first_scanned_time = to_local_dt(event_ts(first_scanned_evt) if first_scanned_evt else None)
    last_scanned_time = to_local_dt(event_ts(last_scanned_evt) if last_scanned_evt else None)
    out_for_delivery_time = to_local_dt(event_ts(ofd_evt) if ofd_evt else None)
    attempted_time = to_local_dt(event_ts(fail_evt) if fail_evt else None)
    delivered_time = to_local_dt(event_ts(success_evt) if success_evt else None)

    # New: warehouse + DSP
    warehouse_id = extract_warehouse_id(payload, events)
    company_id, service_id = extract_company_or_service_id(payload, events)

    warehouse_name = warehouse_map.get(warehouse_id, "") if warehouse_id else ""
    dsp_name = ""
    if company_id:
        dsp_name = company_map_by_company_id.get(company_id, "")
    if not dsp_name and service_id:
        dsp_name = company_map_by_service_id.get(service_id, "")

    row: dict[str, str] = {
        "trakcing_id": tracking_id,
        "shipperName": str(
            shipper_name
            or payload.get("shipperName")
            or (payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}).get("shipperName")
            or (payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}).get("shipperName")
            or (payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}).get("shipperName")
            or ""
        ),
        "dsp_name": dsp_name,
        "warehouse_name": warehouse_name,
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
    return row


def empty_row(tracking_id: str) -> dict[str, str]:
    row = {col: "" for col in OUTPUT_COLUMNS}
    row["trakcing_id"] = tracking_id
    return row


# -----------------------------
# Export helpers
# -----------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="export")
    return output.getvalue()


# -----------------------------
# Streamlit app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Tracking Export", layout="wide")
    st.title("Tracking Export")

    if "dedup_ids" not in st.session_state:
        st.session_state["dedup_ids"] = []
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "failures" not in st.session_state:
        st.session_state["failures"] = []

    st.subheader("0) 配置检查（环境变量）")
    with st.expander("查看当前配置（不显示 token）", expanded=False):
        st.write(
            {
                "KPI_API_URL_TEMPLATE": API_URL_TEMPLATE,
                "KPI_LIST_BASE_URL": LIST_BASE_URL,
                "KPI_WAREHOUSES_URL": WAREHOUSES_URL,
                "KPI_COMPANIES_URL": COMPANIES_URL,
                "KPI_API_TIMEOUT_SECONDS": API_TIMEOUT_SECONDS,
                "KPI_API_TOKEN_set": bool(API_TOKEN),
            }
        )

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

    st.subheader("2) 调用 API 并导出（新增：DSP/仓库映射）")
    if st.button("Fetch / Export", type="primary", disabled=not dedup_ids):
        rows_by_id: dict[str, dict[str, str]] = {}
        failures: list[dict[str, str]] = []

        progress = st.progress(0)
        status = st.empty()

        with requests.Session() as session:
            # 1) Preload mapping tables (once)
            status.text("预加载 warehouses / thirdparty_companies 映射表…")
            warehouse_map: dict[str, str] = {}
            company_map_by_company_id: dict[str, str] = {}
            company_map_by_service_id: dict[str, str] = {}

            try:
                wh_payload = fetch_json(WAREHOUSES_URL, session)
                warehouse_map = build_warehouse_map(wh_payload)
            except Exception as e:  # noqa: BLE001
                st.warning(f"warehouses 映射表加载失败：{e}")

            try:
                co_payload = fetch_json(COMPANIES_URL, session)
                company_map_by_company_id, company_map_by_service_id = build_company_map(co_payload)
            except Exception as e:  # noqa: BLE001
                st.warning(f"thirdparty_companies 映射表加载失败：{e}")

            # 2) Per tracking fetch
            total = len(dedup_ids)
            for idx, tracking_id in enumerate(dedup_ids, start=1):
                status.text(f"处理中：{idx}/{total} - {tracking_id}")
                try:
                    payload = fetch_tracking_data(tracking_id, session)
                    rows_by_id[tracking_id] = build_row(
                        tracking_id=tracking_id,
                        payload=payload,
                        warehouse_map=warehouse_map,
                        company_map_by_company_id=company_map_by_company_id,
                        company_map_by_service_id=company_map_by_service_id,
                    )
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

        with st.expander("字段命中情况（排查 DSP/仓库为空）", expanded=False):
            st.write("如果 dsp_name / warehouse_name 为空，通常意味着 status_logs 返回里没带对应 ID，或映射表接口没拉到数据。")
            st.write("你可以把某个 tracking_id 的原始 JSON 打出来看里面有没有 companyId/serviceId/listWarehouseId。")


if __name__ == "__main__":
    main()
