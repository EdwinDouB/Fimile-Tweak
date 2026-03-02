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
    if BEANS_BASIC_AUTH:
        h["Authorization"] = BEANS_BASIC_AUTH
    if BEANS_TOKEN:
        h["Authorization"] = f"Bearer {BEANS_TOKEN}"
    if BEANS_COOKIE:
        h["Cookie"] = BEANS_COOKIE
    return h


def _get_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = session.get(url, headers=_headers(), params=params, timeout=API_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


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


def fetch_warehouses(session: requests.Session) -> dict[str, dict[str, Any]]:␊
    payload = _get_json(session, "warehouses", params={"updatedAfter": 0})␊
    items = payload.get("warehouse") or payload.get("warehouses") or []
    return {w.get("listWarehouseId"): w for w in items if w.get("listWarehouseId")}


def fetch_assignees(session: requests.Session) -> dict[str, dict[str, Any]]:
    payload = _get_json(session, "assignees", params={"updatedAfter": 0})
    items = payload.get("assignee") or payload.get("assignees") or []
    return {a.get("listAssigneeId"): a for a in items if a.get("listAssigneeId")}


def fetch_warehouses_from_routes(session: requests.Session) -> dict[str, dict[str, Any]]:␊
    """Fallback: derive warehouse options from routes payload when warehouses API is forbidden."""
    routes = fetch_routes(session)
    out: dict[str, dict[str, Any]] = {}
    for r in routes:
        wh = r.get("warehouse") or {}
        wh_id = wh.get("listWarehouseId")
        if not wh_id:
            continue
        out[wh_id] = {
            "listWarehouseId": wh_id,
            "name": wh.get("name") or "",
            "formattedAddress": wh.get("formattedAddress") or wh.get("address") or "",
            "address": wh.get("address") or "",
        }
    return out


def _is_403_error(err: Exception) -> bool:
    if isinstance(err, requests.HTTPError) and err.response is not None:
        return err.response.status_code == 403
    return "403" in str(err)


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
@@ -285,88 +291,110 @@ def main() -> None:
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
                if _is_403_error(e):
                    try:
                        st.session_state["warehouses"] = fetch_warehouses_from_routes(session)
                        st.warning("warehouses 接口无权限，已改用 routes 数据推导仓库列表。")
                        st.success(f"仓库数量：{len(st.session_state['warehouses'])}")
                    except Exception as fallback_e:  # noqa: BLE001
                        st.error(f"刷新仓库失败：{e}；回退 routes 也失败：{fallback_e}")
                else:
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
                    if _is_403_error(e):
                        try:
                            wh_by_id = fetch_warehouses_from_routes(session)
                            st.session_state["warehouses"] = wh_by_id
                            failures.append(
                                {
                                    "step": "warehouses",
                                    "reason": "warehouses 接口 403，已自动回退为 routes 推导仓库列表",
                                }
                            )
                        except Exception as fallback_e:  # noqa: BLE001
                            wh_by_id = {}
                            failures.append({"step": "warehouses", "reason": f"{e}; fallback failed: {fallback_e}"})
                    else:
                        wh_by_id = {}
                        failures.append({"step": "warehouses", "reason": str(e)})

            asg_by_id: dict[str, dict[str, Any]] = st.session_state.get("assignees", {})
            if not asg_by_id:
                status.text("Fetching assignees ...")
                try:
                    asg_by_id = fetch_assignees(session)
                    st.session_state["assignees"] = asg_by_id
                except Exception as e:  # noqa: BLE001
                    asg_by_id = {}
                    failures.append({"step": "assignees", "reason": str(e)})

            status.text("Fetching routes ...")
            try:
                routes = fetch_routes(session)
            except Exception as e:  # noqa: BLE001
                routes = []
                failures.append({"step": "routes", "reason": str(e)})

            status.text("Fetching routes_metrics ...")
            try:
                metrics_by_route = fetch_routes_metrics(session, csv_extra_buids=csv_extra_buids)
            except Exception as e:  # noqa: BLE001
                metrics_by_route = {}
                failures.append({"step": "routes_metrics", "reason": str(e)})

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




