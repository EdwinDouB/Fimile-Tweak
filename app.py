import io
import os
import re
from datetime import datetime, date
from typing import Any

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config (prefer env vars)
# -----------------------------
def _get_config(name: str, default: str = "") -> str:
    """Read config from env first, then Streamlit secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        secret_val = st.secrets.get(name)
    except Exception:  # noqa: BLE001
        secret_val = None
    if secret_val is None:
        return default
    return str(secret_val)


BASE_URL = _get_config("BEANS_BASE_URL", "https://isp.beans.ai/enterprise/v1/lists")
API_TIMEOUT_SECONDS = int(_get_config("BEANS_TIMEOUT_SECONDS", "30"))

# Use ONE of these auth methods:
# 1) Basic auth:  export BEANS_BASIC_AUTH="Basic <base64>"
#    (You can copy the whole 'Basic ...' string from your teammate / DevTools)
# 2) Bearer token: export BEANS_TOKEN="..."
# 3) Cookie session: export BEANS_COOKIE="_session_id=...; other_cookie=..."
BEANS_BASIC_AUTH = _get_config("BEANS_BASIC_AUTH", "")
BEANS_TOKEN = _get_config("BEANS_TOKEN", "")
# Backward-compatible aliases used by earlier internal scripts.␊
KPI_API_TOKEN = _get_config("KPI_API_TOKEN", "")
KPI_API_BASIC_AUTH = _get_config("KPI_API_BASIC_AUTH", "")
BEANS_COOKIE = _get_config("BEANS_COOKIE", "")

# Account buid list for routes_metrics (comma separated)
# Example: export BEANS_ACCOUNT_BUIDS="2c1e8ad1e59945579fa2a992e93932d6"
BEANS_ACCOUNT_BUIDS = _get_config("BEANS_ACCOUNT_BUIDS", "")

OUTPUT_COLUMNS = [
    "route_code",
    "date",
    "start_address",
    "dsp",
    "driver",
    "dsp_driver",
    "delivery_count",
    "pickup_count",
    "failed_count",
    "listRouteId",
    "listWarehouseId",
    "listAssigneeId",
]

REPORT_COLUMNS = [
    "date",
    "route_name",
    "address",
    "package_number",
    "assignee_name",
    "list_route_id",
]

# -----------------------------
# Helpers
# -----------------------------
def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "Fimile-Routes-Export/1.0",
    }
    auth_header = (
        BEANS_BASIC_AUTH.strip()
        or KPI_API_BASIC_AUTH.strip()
        or BEANS_TOKEN.strip()
        or KPI_API_TOKEN.strip()
    )
    if auth_header:
        if auth_header.lower().startswith(("basic ", "bearer ")):
            h["Authorization"] = auth_header
        else:
            h["Authorization"] = f"Bearer {auth_header}"
    if BEANS_COOKIE:
        h["Cookie"] = BEANS_COOKIE
    return h


def _auth_status_summary() -> str:
    has_auth = bool(_headers().get("Authorization"))
    has_cookie = bool(BEANS_COOKIE.strip())
    if has_auth and has_cookie:
        return "Authorization + Cookie"
    if has_auth:
        return "Authorization"
    if has_cookie:
        return "Cookie"
    return "none"


def _get_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = session.get(url, headers=_headers(), params=params, timeout=API_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def _humanize_error(e: Exception) -> str:
    if isinstance(e, requests.HTTPError):
        res = e.response
        if res is not None:
            status = res.status_code
            url = res.url
            hint = ""
            if status in {401, 403}:
                hint = "（鉴权失败或账号无权限：请检查 BEANS_TOKEN / BEANS_BASIC_AUTH / BEANS_COOKIE）"
            elif status == 404:
                hint = "（接口不存在或当前账号不可见）"
            return f"HTTP {status} @ {url}{hint}"
    return str(e)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_dsp(route_name: str) -> str:
    """
    Infer DSP code from route name.
    Examples:
      - "CA-IE01-02/28-DX-ALEX" -> "DX"
      - "TX-0227 - 1 - CBC-Andres" -> "CBC"
    """
    s = route_name.strip()

    # Pattern like "...-DX-ALEX" at end
    m = re.search(r"-([A-Z]{2,6})-[A-Z0-9]+$", s)
    if m:
        return m.group(1)

    # Pattern like "... CBC-Andres"
    m = re.search(r"\b([A-Z]{2,6})-[A-Za-z]", s)
    if m:
        return m.group(1)

    return ""


def _normalize_date_str(raw: Any) -> str:
    s = _safe_str(raw).strip()
    if not s:
        return ""

    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        return datetime.strptime(s, "%m/%d/%Y").strftime("%Y-%m-%d")
    except Exception:
        pass

    if s.isdigit():
        ts = int(s)
        if ts > 10_000_000_000:
            ts = ts / 1000
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    return ""


def _parse_date(route: dict[str, Any]) -> str:
    for k in [
        "dateStr",
        "date",
        "routeDate",
        "routeDateStr",
        "startDate",
        "scheduledDate",
        "startTsMillis",
        "routeTsMillis",
    ]:
        ds = _normalize_date_str(route.get(k))
        if ds:
            return ds

    name = _safe_str(route.get("name"))
    m = re.search(r"(\d{2})/(\d{2})", name)
    if m:
        mm = int(m.group(1))
        dd = int(m.group(2))
        year = int(os.getenv("BEANS_DEFAULT_YEAR", str(datetime.now().year)))
        return f"{year:04d}-{mm:02d}-{dd:02d}"
    return ""


def _compute_planned_actual(stop_metrics: list[dict[str, Any]] | None, metric_type: str) -> tuple[int, int]:
    planned = 0
    actual = 0
    for x in stop_metrics or []:
        metric_kind = _safe_str(x.get("type") or x.get("metricType") or x.get("stopType")).lower()
        if metric_kind != metric_type.lower():
            continue
        pc = int(x.get("packageCount") or x.get("count") or x.get("plannedCount") or 0)
        planned += pc
        actual_raw = x.get("actualCount")
        if actual_raw is not None:
            actual += int(actual_raw or 0)
            continue

        if _safe_str(x.get("status") or x.get("metricStatus")).lower() in {
            "finished",
            "success",
            "succeeded",
            "completed",
        }:
            actual += pc
    return planned, actual


def _compute_failed_count(stop_metrics: list[dict[str, Any]] | None) -> int:
    total = 0
    failed_types = {"failed", "fail", "attempted", "attempt"}
    for x in stop_metrics or []:
        tp = _safe_str(x.get("type") or x.get("metricType") or x.get("stopType")).lower()
        status = _safe_str(x.get("status") or x.get("metricStatus")).lower()
        if tp in failed_types or status in failed_types:
            total += int(x.get("packageCount") or x.get("count") or x.get("plannedCount") or 0)
    return total


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="routes")
    buf.seek(0)
    return buf.read()


def _parse_csv_extra_buids(raw: str) -> str:
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    return ",".join(items)


def _date_in_range(ds: str, start: date | None, end: date | None) -> bool:
    if not (start or end):
        return True
    if not ds:
        return False
    try:
        d = datetime.strptime(ds, "%Y-%m-%d").date()
    except Exception:
        return False
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


# -----------------------------
# Fetchers
# -----------------------------
def fetch_routes(
    session: requests.Session,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "updatedAfter": 0,
        "includeToday": "true",
        "includePast": "true",
        "includeFuture": "true",
    }
    if start_date:
        sd = start_date.strftime("%Y-%m-%d")
        params.update(
            {
                "startDate": sd,
                "dateFrom": sd,
                "fromDate": sd,
                "routeDateFrom": sd,
            }
        )
    if end_date:
        ed = end_date.strftime("%Y-%m-%d")
        params.update(
            {
                "endDate": ed,
                "dateTo": ed,
                "toDate": ed,
                "routeDateTo": ed,
            }
        )

    # Some tenants return only recent items unless pagination params are provided.
    # We request pages and merge unique routes by listRouteId.
    page_size = 500
    all_routes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for page in range(1, 51):
        paged_params = {
            **params,
            "page": page,
            "pageSize": page_size,
            "limit": page_size,
            "offset": (page - 1) * page_size,
        }
        payload = _get_json(session, "routes", params=paged_params)
        items = payload.get("route") or payload.get("routes") or []
        if not items:
            break

        appended = 0
        for r in items:
            rid = _safe_str(r.get("listRouteId"))
            key = rid or _safe_str(r.get("name"))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            all_routes.append(r)
            appended += 1

        if appended == 0 or len(items) < page_size:
            break

    return all_routes


def _extract_reference_from_routes(
    routes: list[dict[str, Any]], key: str, id_key: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in routes:
        ref = r.get(key) or {}
        if not isinstance(ref, dict):
            continue
        ref_id = ref.get(id_key)
        if not ref_id:
            continue
        current = out.get(ref_id, {})
        merged = {**current, **ref}
        out[ref_id] = merged
    return out


def fetch_warehouses(session: requests.Session) -> dict[str, dict[str, Any]]:
    try:
        payload = _get_json(session, "warehouses", params={"updatedAfter": 0})
        items = payload.get("warehouse") or payload.get("warehouses") or []
        return {w.get("listWarehouseId"): w for w in items if w.get("listWarehouseId")}
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        # Some accounts are forbidden to read /warehouses directly.
        # Fall back to route payload, which usually embeds warehouse info.
        if status in {401, 403, 404}:
            routes = fetch_routes(session)
            return _extract_reference_from_routes(routes, key="warehouse", id_key="listWarehouseId")
        raise


def fetch_assignees(session: requests.Session) -> dict[str, dict[str, Any]]:
    try:
        payload = _get_json(session, "assignees", params={"updatedAfter": 0})
        items = payload.get("assignee") or payload.get("assignees") or []
        return {a.get("listAssigneeId"): a for a in items if a.get("listAssigneeId")}
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        # Some accounts are forbidden to read /assignees directly.
        # Fall back to route payload, which usually embeds assignee info.
        if status in {401, 403, 404}:
            routes = fetch_routes(session)
            return _extract_reference_from_routes(routes, key="assignee", id_key="listAssigneeId")
        raise


def fetch_routes_metrics(session: requests.Session, csv_extra_buids: str) -> dict[str, dict[str, Any]]:
    params: dict[str, Any] = {}
    if csv_extra_buids:
        params["csvExtraAccountBuidsList"] = csv_extra_buids

    payload = _get_json(session, "routes_metrics", params=params)
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("routesMetrics") or payload.get("routeMetrics") or payload.get("metrics") or []

    out: dict[str, dict[str, Any]] = {}
    for m in items:
        rid = _safe_str(m.get("listRouteId") or m.get("routeId") or m.get("id"))
        route_code = _safe_str(m.get("routeCode") or m.get("name"))
        if rid:
            out[rid] = m
        if route_code:
            out[route_code] = m
    return out


def fetch_routes_report(
    session: requests.Session,
    from_date: date,
    to_date: date,
    route_name: str = "",
    address: str = "",
    package_number: str = "",
    assignee_name: str = "",
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "date_from": from_date.strftime("%Y-%m-%d"),
        "date_to": to_date.strftime("%Y-%m-%d"),
    }
    if route_name.strip():
        params["route_name"] = route_name.strip()
    if address.strip():
        params["address"] = address.strip()
    if package_number.strip():
        params["package_number"] = package_number.strip()
    if assignee_name.strip():
        params["assignee_name"] = assignee_name.strip()

    payload = _get_json(session, "items/do/report", params=params)
    if isinstance(payload, list):
        return payload
    return payload.get("items") or payload.get("data") or payload.get("report") or []


def build_report_rows(report_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report_items:
        rows.append(
            {
                "date": _normalize_date_str(item.get("date") or item.get("dateStr") or item.get("routeDate")),
                "route_name": _safe_str(item.get("route_name") or item.get("routeName") or item.get("name")),
                "address": _safe_str(item.get("address") or item.get("formattedAddress")),
                "package_number": _safe_str(
                    item.get("package_number") or item.get("packageNumber") or item.get("trackingId")
                ),
                "assignee_name": _safe_str(
                    item.get("assignee_name") or item.get("assigneeName") or item.get("driver")
                ),
                "list_route_id": _safe_str(item.get("list_route_id") or item.get("listRouteId") or item.get("routeId")),
            }
        )
    return rows


# -----------------------------
# Assembly
# -----------------------------
def build_rows(
    routes: list[dict[str, Any]],
    metrics_by_route: dict[str, dict[str, Any]],
    wh_by_id: dict[str, dict[str, Any]],
    asg_by_id: dict[str, dict[str, Any]],
    start_date: date | None,
    end_date: date | None,
    warehouse_filter: str,
    name_contains: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name_contains_lc = (name_contains or "").strip().lower()

    for r in routes:
        route_id = r.get("listRouteId")
        route_code = _safe_str(r.get("name"))

        ds = _parse_date(r)
        if not _date_in_range(ds, start_date, end_date):
            continue

        wh_id = (r.get("warehouse") or {}).get("listWarehouseId") or ""
        if warehouse_filter and wh_id != warehouse_filter:
            continue

        if name_contains_lc and name_contains_lc not in route_code.lower():
            continue

        start_addr = ""
        if wh_id and wh_id in wh_by_id:
            w = wh_by_id[wh_id]
            start_addr = _safe_str(w.get("formattedAddress") or w.get("address") or w.get("name") or "")

        asg_id = (r.get("assignee") or {}).get("listAssigneeId") or ""
        driver = ""
        if asg_id and asg_id in asg_by_id:
            driver = _safe_str(asg_by_id[asg_id].get("name") or "")

        dsp = _extract_dsp(route_code)
        dsp_driver = "-".join([x for x in [dsp, driver] if x])

        m = metrics_by_route.get(_safe_str(route_id), {}) or metrics_by_route.get(route_code, {})
        stop_metrics = (
            m.get("stopMetrics")
            or m.get("metrics")
            or m.get("routeMetrics")
            or m.get("stopLevelMetrics")
            or []
        )
        d_plan, d_act = _compute_planned_actual(stop_metrics, "dropoff")
        if d_plan == 0 and d_act == 0:
            d_plan, d_act = _compute_planned_actual(stop_metrics, "delivery")
        p_plan, p_act = _compute_planned_actual(stop_metrics, "pickup")
        failed_count = _compute_failed_count(stop_metrics)

        rows.append(
            {
                "route_code": route_code,
                "date": ds,
                "start_address": start_addr,
                "dsp": dsp,
                "driver": driver,
                "dsp_driver": dsp_driver,
                "delivery_count": f"{d_plan}/{d_act}",
                "pickup_count": f"{p_plan}/{p_act}",
                "failed_count": failed_count,
                "listRouteId": _safe_str(route_id),
                "listWarehouseId": _safe_str(wh_id),
                "listAssigneeId": _safe_str(asg_id),
            }
        )
    return rows


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Routes Ops Export", layout="wide")
    st.title("Routes Ops Export")

    with st.expander("运行前准备（鉴权放环境变量）"):
        st.markdown(
            """
- Bearer：`BEANS_TOKEN`
- Cookie：`BEANS_COOKIE`（例如 `_session_id=...`）
- routes_metrics：`BEANS_ACCOUNT_BUIDS`（逗号分隔）
            """.strip()
        )
        st.caption(f"当前检测到鉴权方式：{_auth_status_summary()}")

    # Session state
    if "warehouses" not in st.session_state:
        st.session_state["warehouses"] = {}
    if "assignees" not in st.session_state:
        st.session_state["assignees"] = {}
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "report_df" not in st.session_state:
        st.session_state["report_df"] = None
    if "failures" not in st.session_state:
        st.session_state["failures"] = []

    st.subheader("1) 过滤条件")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start date（可空）", value=None)
    end = c2.date_input("End date（可空）", value=None)
    name_contains = c3.text_input("Route name contains（可空）", value="")

    st.subheader("2) 参考表（仓库/司机）")
    b1, b2 = st.columns(2)
    if b1.button("刷新仓库列表"):
        with requests.Session() as session:
            try:
                st.session_state["warehouses"] = fetch_warehouses(session)
                st.success(f"仓库数量：{len(st.session_state['warehouses'])}")
            except Exception as e:  # noqa: BLE001
                st.error(f"刷新仓库失败：{e}")

    if b2.button("刷新司机列表"):
        with requests.Session() as session:
            try:
                st.session_state["assignees"] = fetch_assignees(session)
                st.success(f"司机数量：{len(st.session_state['assignees'])}")
            except Exception as e:  # noqa: BLE001
                st.error(f"刷新司机失败：{e}")

    wh_by_id: dict[str, dict[str, Any]] = st.session_state.get("warehouses", {})
    warehouse_ids = sorted([k for k in wh_by_id.keys() if k])
    warehouse_filter = st.selectbox("Warehouse（可选）", options=[""] + warehouse_ids, index=0)

    st.subheader("3) Fetch / Export")
    if st.button("Fetch / Export", type="primary"):
        failures: list[dict[str, str]] = []
        progress = st.progress(0)
        status = st.empty()

        csv_extra_buids = _parse_csv_extra_buids(BEANS_ACCOUNT_BUIDS)

        with requests.Session() as session:
            # Use cached reference tables if present; if empty, try fetch once.
            if not wh_by_id:
                status.text("Fetching warehouses ...")
                try:
                    wh_by_id = fetch_warehouses(session)
                    st.session_state["warehouses"] = wh_by_id
                except Exception as e:  # noqa: BLE001
                    wh_by_id = {}
                    failures.append({
                        "step": "warehouses",
                        "reason": f"{_humanize_error(e)}；且回退 routes 也失败",
                    })

            asg_by_id: dict[str, dict[str, Any]] = st.session_state.get("assignees", {})
            if not asg_by_id:
                status.text("Fetching assignees ...")
                try:
                    asg_by_id = fetch_assignees(session)
                    st.session_state["assignees"] = asg_by_id
                except Exception as e:  # noqa: BLE001
                    asg_by_id = {}
                    failures.append({
                        "step": "assignees",
                        "reason": f"{_humanize_error(e)}；且回退 routes 也失败",
                    })

            status.text("Fetching routes ...")
            try:
                routes = fetch_routes(session, start_date=start, end_date=end)
            except Exception as e:  # noqa: BLE001
                routes = []
                failures.append({"step": "routes", "reason": _humanize_error(e)})

            status.text("Fetching routes_metrics ...")
            try:
                metrics_by_route = fetch_routes_metrics(session, csv_extra_buids=csv_extra_buids)
            except Exception as e:  # noqa: BLE001
                metrics_by_route = {}
                failures.append({"step": "routes_metrics", "reason": _humanize_error(e)})

            status.text("Assembling rows ...")
            rows = build_rows(
                routes=routes,
                metrics_by_route=metrics_by_route,
                wh_by_id=wh_by_id,
                asg_by_id=asg_by_id,
                start_date=start,
                end_date=end,
                warehouse_filter=warehouse_filter,
                name_contains=name_contains,
            )

            progress.progress(1.0)

        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        st.session_state["result_df"] = df
        st.session_state["failures"] = failures
        status.text("Done")

    st.subheader("4) Routes Report (items/do/report)")
    r1, r2, r3 = st.columns(3)
    report_from = r1.date_input("Report from date", value=date.today(), key="report_from")
    report_to = r2.date_input("Report to date", value=date.today(), key="report_to")
    report_route_name = r3.text_input("Report route_name（可空）", value="")
    rr1, rr2, rr3 = st.columns(3)
    report_address = rr1.text_input("Report address（可空）", value="")
    report_package_number = rr2.text_input("Report package_number（可空）", value="")
    report_assignee_name = rr3.text_input("Report assignee_name（可空）", value="")

    if st.button("Fetch Routes Report"):
        if report_from > report_to:
            st.error("Report from date 不能大于 to date")
        else:
            with requests.Session() as session:
                try:
                    report_items = fetch_routes_report(
                        session=session,
                        from_date=report_from,
                        to_date=report_to,
                        route_name=report_route_name,
                        address=report_address,
                        package_number=report_package_number,
                        assignee_name=report_assignee_name,
                    )
                    report_rows = build_report_rows(report_items)
                    st.session_state["report_df"] = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
                    st.success(f"Routes Report 获取成功：{len(report_rows)} 条")
                except Exception as e:  # noqa: BLE001
                    st.session_state["report_df"] = None
                    st.error(f"Routes Report 获取失败：{_humanize_error(e)}")

    failures = st.session_state.get("failures", [])
    df: pd.DataFrame | None = st.session_state.get("result_df")

    if failures:
        st.warning("有步骤失败（但仍可能导出部分数据）")
        st.dataframe(pd.DataFrame(failures), use_container_width=True)

    if df is not None:
        st.subheader("结果预览")
        st.dataframe(df.head(100), use_container_width=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载 CSV", data=csv_data, file_name=f"routes_ops_{stamp}.csv", mime="text/csv")

        try:
            xlsx_data = _to_excel_bytes(df)
            st.download_button(
                "下载 Excel",
                data=xlsx_data,
                file_name=f"routes_ops_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.info("当前环境未能生成 Excel（已提供 CSV）。")

    report_df: pd.DataFrame | None = st.session_state.get("report_df")
    if report_df is not None:
        st.subheader("Routes Report 预览")
        st.dataframe(report_df.head(100), use_container_width=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_csv_data = report_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载 Routes Report CSV",
            data=report_csv_data,
            file_name=f"routes_report_{stamp}.csv",
            mime="text/csv",
        )

        try:
            report_xlsx_data = _to_excel_bytes(report_df)
            st.download_button(
                "下载 Routes Report Excel",
                data=report_xlsx_data,
                file_name=f"routes_report_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.info("当前环境未能生成 Routes Report Excel（已提供 CSV）。")



if __name__ == "__main__":
    main()


