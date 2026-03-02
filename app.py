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
BASE_URL = os.getenv("BEANS_BASE_URL", "https://isp.beans.ai/enterprise/v1/lists")
API_TIMEOUT_SECONDS = int(os.getenv("BEANS_TIMEOUT_SECONDS", "30"))

# Use ONE of these auth methods:
# 1) Basic auth:  export BEANS_BASIC_AUTH="Basic <base64>"
#    (You can copy the whole 'Basic ...' string from your teammate / DevTools)
# 2) Bearer token: export BEANS_TOKEN="..."
# 3) Cookie session: export BEANS_COOKIE="_session_id=...; other_cookie=..."
BEANS_BASIC_AUTH = os.getenv("BEANS_BASIC_AUTH", "")
BEANS_TOKEN = os.getenv("BEANS_TOKEN", "")
# Backward-compatible aliases used by earlier internal scripts.
KPI_API_TOKEN = os.getenv("KPI_API_TOKEN", "")
KPI_API_BASIC_AUTH = os.getenv("KPI_API_BASIC_AUTH", "")
BEANS_COOKIE = os.getenv("BEANS_COOKIE", "")

# Account buid list for routes_metrics (comma separated)
# Example: export BEANS_ACCOUNT_BUIDS="2c1e8ad1e59945579fa2a992e93932d6"
BEANS_ACCOUNT_BUIDS = os.getenv("BEANS_ACCOUNT_BUIDS", "")

OUTPUT_COLUMNS = [
    "route_code",
    "date",
    "start_address",
    "dsp",
    "driver",
    "dsp_driver",
    "delivery_count",
    "pickup_count",
    "listRouteId",
    "listWarehouseId",
    "listAssigneeId",
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


def _parse_date(route: dict[str, Any]) -> str:
    ds = route.get("dateStr")
    if ds:
        return _safe_str(ds)

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
        if x.get("type") != metric_type:
            continue
        pc = int(x.get("packageCount") or 0)
        planned += pc
        if x.get("status") == "finished":
            actual += pc
    return planned, actual


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
def fetch_routes(session: requests.Session) -> list[dict[str, Any]]:
    payload = _get_json(session, "routes", params={"updatedAfter": 0, "includeToday": "true"})
    return payload.get("route") or payload.get("routes") or []


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
        rid = m.get("listRouteId")
        if rid:
            out[rid] = m
    return out


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

        m = metrics_by_route.get(route_id or "", {})
        stop_metrics = m.get("stopMetrics") or []
        d_plan, d_act = _compute_planned_actual(stop_metrics, "dropoff")
        p_plan, p_act = _compute_planned_actual(stop_metrics, "pickup")

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

    # Session state
    if "warehouses" not in st.session_state:
        st.session_state["warehouses"] = {}
    if "assignees" not in st.session_state:
        st.session_state["assignees"] = {}
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
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
                routes = fetch_routes(session)
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


if __name__ == "__main__":
    main()





