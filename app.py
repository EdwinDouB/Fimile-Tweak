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
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# 你同事已有：normalize_events(payload) / event_type(e) / event_ts(e)

OFD_TYPES = {"out-for-delivery", "ofd", "outfordelivery"}
SUCCESS_TYPES = {"success", "delivered"}
FAIL_TYPES = {"fail", "failed", "failure"}

# API configuration comes from code/env only (not exposed in UI).
API_URL_TEMPLATE = os.getenv(
    "KPI_API_URL_TEMPLATE",
    "https://isp.beans.ai/enterprise/v1/lists/status_logs"
    "?tracking_id={tracking_id}&readable=true" \
    "&include_pod=true&include_item=true",
)
API_TOKEN = os.getenv("KPI_API_TOKEN", "")
API_TIMEOUT_SECONDS = int(os.getenv("KPI_API_TIMEOUT_SECONDS", "20"))
# 1) 先全自动拿到某天 OFD 的 tracking_id 列表（你需要填对这个接口模板）
OFD_LIST_URL_TEMPLATE = os.getenv(
    "KPI_OFD_LIST_URL_TEMPLATE",
    # 下面这个只是示例格式：你要替换成你们真实可用的 endpoint
    # 例如：https://isp.beans.ai/enterprise/v1/lists/items?event=out-for-delivery&date=YYYY-MM-DD
    "https://isp.beans.ai/enterprise/v1/REPLACE_ME?date={date}",
)

# 2) 这个接口可能会分页（如果你们返回 next_page_token / page / limit）
OFD_LIST_PAGE_SIZE = int(os.getenv("KPI_OFD_LIST_PAGE_SIZE", "500"))


OUTPUT_COLUMNS = [
    "trakcing_id",
    "shipperName",
    "created_time",
    "scanned_time",
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

def pick_first_event(events, types:set[str], start_ms:int|None=None, end_ms:int|None=None):
    """在给定时间窗内找最早出现的某类事件"""
    candidates = []
    for e in events:
        t = event_type(e)
        if t not in types:
            continue
        ts = event_ts(e)
        if ts is None:
            continue
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts > end_ms:
            continue
        candidates.append((ts, e))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def get_item_meta(payload: dict) -> dict:
    """
    从 include_item=true 的 payload 里尽量提取 Region/Hub/DSP/Driver
    由于字段名你们内部可能不同，这里写成“多候选字段兜底”
    """
    # 常见情况：payload里会有 item / listItem / data.item 等
    item = None
    for k in ("item", "listItem", "list_item"):
        if isinstance(payload.get(k), dict):
            item = payload[k]
            break
    if item is None and isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("item"), dict):
        item = payload["data"]["item"]

    item = item or {}

    def pick(*keys):
        for key in keys:
            v = item.get(key)
            if v not in (None, ""):
                return str(v)
        return ""

    region = pick("region", "Region")
    hub = pick("hub", "Hub", "station", "Station")
    dsp = pick("dsp", "DSP", "carrier", "Carrier")
    driver = pick("driverName", "DriverName", "assigneeName", "AssigneeName")

    # 有些系统 driver 在 route/assignee 对象里
    assignee = item.get("assignee")
    if not driver and isinstance(assignee, dict):
        driver = str(assignee.get("name") or assignee.get("email") or "")

    return {"Region": region, "Hub": hub, "DSP": dsp, "Driver_Name": driver}

def classify_24h(payload: dict, selected_date: datetime) -> dict:
    """
    返回：
      - ofd_time_ms (用于判断是否属于选定日期)
      - status: successful/failed/unfinished/skip
      - bad_pod: 0/1
      - meta: Region/Hub/DSP/Driver_Name
    """
    events = normalize_events(payload)

    # 1) 找 OFD（最早那一次）
    ofd_evt = pick_first_event(events, OFD_TYPES)
    if not ofd_evt:
        return {"status": "skip"}  # 没有OFD，不参与当天KPI

    ofd_ms = event_ts(ofd_evt)
    if ofd_ms is None:
        return {"status": "skip"}

    ofd_dt = datetime.fromtimestamp(ofd_ms/1000, tz=timezone.utc).astimezone()

    # 2) 判断 OFD 是否落在 selected_date 当天
    # selected_date 传入当天 00:00 本地时间
    day_start = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    if not (day_start <= ofd_dt < day_end):
        return {"status": "skip"}  # 不是这天OFD

    # 3) 在 24h 窗口内找 delivered/failed
    win_start = ofd_ms
    win_end = ofd_ms + int(24*3600*1000)

    success_evt = pick_first_event(events, SUCCESS_TYPES, start_ms=win_start, end_ms=win_end)
    fail_evt = pick_first_event(events, FAIL_TYPES, start_ms=win_start, end_ms=win_end)

    status = "unfinished"
    if success_evt:
        status = "successful"
    elif fail_evt:
        status = "failed"

    # 4) bad POD（你按业务改这个判断）
    bad_pod = 0
    if status == "successful":
        # 通用兜底：看看事件里有没有 pod/images/tags 等
        # 你同事的 event_ts 里已经考虑 podTimestampEpoch，说明 payload 里有 pod
        # 这里写一个宽松判断：没有任何 pod 信息就算 bad_pod
        pod_ok = False
        for e in events:
            pod = e.get("pod")
            if isinstance(pod, dict) and (pod.get("podTimestampEpoch") or pod.get("images") or pod.get("signature")):
                pod_ok = True
                break
            # 有些字段在 e["logItem"]["pod"]
            logItem = e.get("logItem")
            if isinstance(logItem, dict):
                pod2 = logItem.get("pod")
                if isinstance(pod2, dict) and (pod2.get("podTimestampEpoch") or pod2.get("images") or pod2.get("signature")):
                    pod_ok = True
                    break
        if not pod_ok:
            bad_pod = 1

    meta = get_item_meta(payload)
    return {
        "status": status,
        "bad_pod": bad_pod,
        "ofd_time_ms": ofd_ms,
        **meta,
    }

def aggregate_kpi(rows: list[dict], selected_date_str: str) -> pd.DataFrame:
    """
    rows: 每个 tracking_id 的分类结果（已过滤 skip）
    输出你要的列：
    Date Region Hub DSP Driver_Name total_packages total_successful total_failed total_unfinished bad_POD
    """
    agg = defaultdict(lambda: {
        "total_packages": 0,
        "total_successful": 0,
        "total_failed": 0,
        "total_unfinished": 0,
        "bad_POD": 0,
    })

    for r in rows:
        key = (selected_date_str, r.get("Region",""), r.get("Hub",""), r.get("DSP",""), r.get("Driver_Name",""))
        agg[key]["total_packages"] += 1
        if r["status"] == "successful":
            agg[key]["total_successful"] += 1
        elif r["status"] == "failed":
            agg[key]["total_failed"] += 1
        else:
            agg[key]["total_unfinished"] += 1
        agg[key]["bad_POD"] += int(r.get("bad_pod", 0))

    out_rows = []
    for (date, region, hub, dsp, driver), v in agg.items():
        out_rows.append({
            "Date": date,
            "Region": region,
            "Hub": hub,
            "DSP": dsp,
            "Driver_Name": driver,
            **v
        })

    return pd.DataFrame(out_rows, columns=[
        "Date","Region","Hub","DSP","Driver_Name",
        "total_packages","total_successful","total_failed","total_unfinished","bad_POD"
    ])

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


def extract_shipper_name_from_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        item = event.get("item")
        if isinstance(item, dict):
            name = item.get("shipperName")
            if name:
                return str(name)
    return ""


def build_row(tracking_id: str, payload: dict[str, Any]) -> dict[str, str]:
    events = normalize_events(payload)
    shipper_name = extract_shipper_name_from_events(events)

    created_evt = first_event_by_predicate(events, lambda e: event_type(e) == "label")
    scanned_evt = first_event_by_predicate(
        events,
        lambda e: str(e.get("description", "")).strip().lower().startswith("scan at"),
    )
    ofd_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"out-for-delivery", "ofd", "outfordelivery"})
    fail_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"fail", "failed", "failure"})
    success_evt = first_event_by_predicate(events, lambda e: event_type(e) in {"success", "delivered"})

    created_time = to_local_dt(event_ts(created_evt) if created_evt else None)
    scanned_time = to_local_dt(event_ts(scanned_evt) if scanned_evt else None)
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
        "scanned_time": fmt_dt(scanned_time),
        "out_for_delivery_time": fmt_dt(out_for_delivery_time),
        "attempted_time": fmt_dt(attempted_time),
        "failed_route": parse_route(fail_evt.get("description")) if fail_evt else "",
        "delivered_time": fmt_dt(delivered_time),
        "success_route": parse_route(success_evt.get("description")) if success_evt else "",
        "创建到入库时间": diff_hours(scanned_time, created_time),
        "库内停留时间": diff_hours(out_for_delivery_time, scanned_time),
        "尝试配送时间": diff_hours(attempted_time, out_for_delivery_time),
        "送达时间": diff_hours(delivered_time, out_for_delivery_time),
        "整体配送时间": diff_hours(delivered_time, created_time),
    }
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

def fetch_ofd_tracking_ids(date_str: str, session: requests.Session) -> list[str]:
    """
    date_str: 'YYYY-MM-DD'
    目标：返回当天所有发生过 OFD 的 tracking_id 列表（去重）
    注意：这里依赖 KPI_OFD_LIST_URL_TEMPLATE 指向一个“能按日期返回 items/tracking_id”的接口。
    """

    url = OFD_LIST_URL_TEMPLATE.format(date=date_str)

    headers = {"Accept": "application/json"}
    if API_TOKEN:
        token = API_TOKEN.strip()
        if token.lower().startswith("basic ") or token.lower().startswith("bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"

    # ====== 下面是“通用兼容写法” ======
    # 你们可能是一次返回全部，也可能需要分页。
    # 我先写成：循环拉取直到没有 next_page_token / nextPageToken / has_more
    tracking_ids: list[str] = []
    next_token = None
    safety = 0

    while True:
        safety += 1
        if safety > 200:
            break

        req_url = url
        if next_token:
            # 兼容常见分页参数名（你们如果不是这个规则，改这里就行）
            sep = "&" if ("?" in req_url) else "?"
            req_url = f"{req_url}{sep}page_token={next_token}&limit={OFD_LIST_PAGE_SIZE}"
        else:
            sep = "&" if ("?" in req_url) else "?"
            req_url = f"{req_url}{sep}limit={OFD_LIST_PAGE_SIZE}"

        resp = session.get(req_url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()

        # 兼容常见数据字段名：items / listItems / data.items / result.items
        items = []
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                items = payload["items"]
            elif isinstance(payload.get("listItems"), list):
                items = payload["listItems"]
            elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("items"), list):
                items = payload["data"]["items"]
            elif isinstance(payload.get("result"), dict) and isinstance(payload["result"].get("items"), list):
                items = payload["result"]["items"]

        for it in items:
            if not isinstance(it, dict):
                continue
            tid = it.get("tracking_id") or it.get("trackingId") or it.get("trackingID")
            if tid:
                tracking_ids.append(str(tid).strip())

        # 兼容分页 token
        next_token = None
        for k in ("next_page_token", "nextPageToken", "nextToken", "pageToken"):
            if payload.get(k):
                next_token = str(payload.get(k))
                break

        # 兼容 has_more
        has_more = payload.get("has_more") or payload.get("hasMore")
        if next_token:
            continue
        if has_more is True:
            # 有些接口只给 has_more，不给 token，这种你要按 page=1,2.. 改写
            # 先直接 break，避免死循环
            break

        break

    # 去重但保序
    seen = set()
    dedup = []
    for x in tracking_ids:
        if x and x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="export")
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="OFD 24h KPI", layout="wide")
    st.title("OFD 24h KPI (Auto)")

    selected_date = st.date_input("选择日期（以 Out for delivery 当天统计）")
    date_str = selected_date.strftime("%Y-%m-%d")

    st.caption("会自动获取当天所有 OFD 的 tracking_id，然后按 24h 规则聚合到 Driver 维度输出。")

    if st.button("Run KPI", type="primary"):
        if "REPLACE_ME" in OFD_LIST_URL_TEMPLATE:
            st.error("你还没配置 KPI_OFD_LIST_URL_TEMPLATE（全自动拉 OFD 列表的接口模板）")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        with requests.Session() as session:
            status.text("Step 1/2: 拉取当天 OFD tracking_ids ...")
            tracking_ids = fetch_ofd_tracking_ids(date_str, session)
            st.write(f"当天 OFD tracking_ids 数量（去重后）：{len(tracking_ids)}")

            rows = []
            total = len(tracking_ids)
            status.text("Step 2/2: 拉取 status_logs 并分类聚合 ...")

            for idx, tracking_id in enumerate(tracking_ids, start=1):
                try:
                    payload = fetch_tracking_data(tracking_id, session)
                    r = classify_24h(payload, datetime.combine(selected_date, datetime.min.time()).astimezone())
                    if r.get("status") != "skip":
                        rows.append(r)
                except Exception:
                    # 你也可以像你同事那样记录失败列表
                    pass

                if total > 0:
                    progress.progress(idx / total)

        df = aggregate_kpi(rows, date_str)

        st.subheader("KPI 输出（按 Date/Region/Hub/DSP/Driver 聚合）")
        st.dataframe(df, use_container_width=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "下载 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"kpi_{date_str}_{stamp}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
