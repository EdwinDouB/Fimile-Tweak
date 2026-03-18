from utils.utils import *
import streamlit as st
from datetime import date, datetime, time, timedelta
import json
from typing import Any
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()

def _read_with_aliases(*names: str, default: str = "") -> str:
    for name in names:
        value = read_config(name, "")
        if value:
            return value
    return default


def _load_mysql_config() -> dict[str, str | int]:
    return {
        "host": _read_with_aliases("MYSQL_HOST", "DB_HOST"),
        "port": int(_read_with_aliases("MYSQL_PORT", "DB_PORT", default="3306")),
        "username": _read_with_aliases("MYSQL_USERNAME", "MYSQL_USER", "DB_USERNAME", "DB_USER"),
        "password": _read_with_aliases("MYSQL_PASSWORD", "MYSQL_PASS", "DB_PASSWORD"),
        "database": _read_with_aliases("MYSQL_DATABASE", "MYSQL_DB", "DB_DATABASE", "DB_NAME"),
        "ssl_ca": _read_with_aliases("MYSQL_SSL_CA", "DB_SSL_CA"),
    }


def _resolve_path(path_value: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""

    candidates = []
    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                Path(__file__).resolve().parent.parent / raw_path,
            ]
        )

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate.resolve())
        except Exception:
            continue

    default_ca_candidates = (
        Path("/etc/ssl/certs/ca-certificates.crt"),
        Path("/etc/pki/tls/certs/ca-bundle.crt"),
        Path("/etc/ssl/cert.pem"),
    )
    for candidate in default_ca_candidates:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate.resolve())
        except Exception:
            continue

    return ""


def _build_mysql_connect_kwargs(pymysql_module: Any) -> dict[str, Any]:
    config = _load_mysql_config()
    connect_kwargs: dict[str, Any] = {
        "host": str(config["host"]),
        "port": int(config["port"]),
        "user": str(config["username"]),
        "password": str(config["password"]),
        "database": str(config["database"]),
        "charset": "utf8mb4",
        "cursorclass": pymysql_module.cursors.DictCursor,
        "autocommit": True,
    }

    ssl_ca = _resolve_path(str(config.get("ssl_ca") or ""))
    if ssl_ca:
        connect_kwargs["ssl"] = {"ca": ssl_ca}

    return connect_kwargs


def _connect_mysql(pymysql_module: Any) -> Any:
    return pymysql_module.connect(**_build_mysql_connect_kwargs(pymysql_module))


def _discover_table_candidates(conn: Any, explicit_name: str, candidates: list[str]) -> tuple[str, set[str]]:
    requested = [explicit_name, *candidates]
    requested = [name.strip() for name in requested if str(name).strip()]

    with conn.cursor() as cur:
        existing_tables: set[str] = set()

        try:
            cur.execute("SHOW TABLES")
            for row in cur.fetchall() or []:
                if not isinstance(row, dict):
                    continue
                for value in row.values():
                    table_name = str(value or "").strip()
                    if table_name:
                        existing_tables.add(table_name)
        except Exception:
            existing_tables = set()

        if not existing_tables:
            schema = str(_load_mysql_config()["database"])
            try:
                if requested:
                    placeholders = ", ".join(["%s"] * len(requested))
                    cur.execute(
                        f"""
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = %s
                            AND table_name IN ({placeholders})
                        """,
                        [schema, *requested],
                    )
                    existing_tables = {
                        str(row.get("table_name") or "").strip()
                        for row in cur.fetchall() or []
                        if isinstance(row, dict)
                    }
            except Exception:
                existing_tables = set()

        for candidate in requested:
            if candidate in existing_tables:
                return candidate, existing_tables

    return "", existing_tables


def _resolve_tracking_source_table(conn: Any) -> str:
    preferred = _read_with_aliases("TRACKING_SOURCE_TABLE", "MYSQL_TRACKING_SOURCE_TABLE")
    table_name, existing_tables = _discover_table_candidates(
        conn,
        preferred,
        [
            "waybill_waybills",
            "biz_delivery_order",
        ],
    )
    if table_name:
        return table_name

    if existing_tables:
        fuzzy_candidates = sorted(
            table
            for table in existing_tables
            if "delivery" in table.casefold() and "order" in table.casefold()
        )
        if fuzzy_candidates:
            return fuzzy_candidates[0]

    return ""


def _resolve_tracking_number_column(columns: set[str]) -> str:
    preferred = _read_with_aliases("TRACKING_NUMBER_COLUMN", "MYSQL_TRACKING_NUMBER_COLUMN")
    if preferred and preferred in columns:
        return preferred

    for candidate in (
        "tracking_number",
        "tracking_id",
        "waybill_no",
        "waybill_number",
        "mail_no",
        "express_no",
        "logistics_no",
        "order_no",
    ):
        if candidate in columns:
            return candidate
    return ""


def _resolve_tracking_created_at_column(columns: set[str]) -> str:
    preferred = _read_with_aliases("TRACKING_CREATED_AT_COLUMN", "MYSQL_TRACKING_CREATED_AT_COLUMN")
    if preferred and preferred in columns:
        return preferred

    for candidate in (
        "created_at",
        "create_time",
        "created_time",
        "gmt_create",
        "order_time",
        "delivery_time",
        "first_out_for_delivery_date",
    ):
        if candidate in columns:
            return candidate
    return ""


def _resolve_router_messages_table(conn: Any) -> str:
    """Pick an existing router_messages cache table, with env override support."""
    preferred = _read_with_aliases("ROUTER_MESSAGES_TABLE", "MYSQL_ROUTER_MESSAGES_TABLE")
    candidates = [
        "third_party_transit_cache",
        "transit_third_party_cache",
        "third_party_cache",
        "transit_router_messages_cache",
        "biz_delivery_dimension_item",
    ]
    table_name, existing_tables = _discover_table_candidates(conn, preferred, candidates)
    if table_name:
        return table_name

    if existing_tables:
        fuzzy_candidates = sorted(
            table
            for table in existing_tables
            if "third_party_cache" in table or ("dimension" in table and "item" in table)
        )
        if fuzzy_candidates:
            return fuzzy_candidates[0]

    with conn.cursor() as cur:
        # Last-resort lookup when metadata queries are restricted.
        for candidate in [preferred, *candidates]:
            candidate = str(candidate or "").strip()
            if not candidate or not candidate.replace("_", "").isalnum():
                continue
            try:
                cur.execute(f"SELECT 1 FROM {candidate} LIMIT 1")
                return candidate
            except Exception:
                continue

    return ""


def _load_table_columns(conn: Any, table_name: str) -> set[str]:
    """Best-effort column discovery for a table."""
    columns: set[str] = set()
    if not table_name or not table_name.replace("_", "").isalnum():
        return columns

    with conn.cursor() as cur:
        try:
            cur.execute(f"SHOW COLUMNS FROM {table_name}")
            for row in cur.fetchall() or []:
                if not isinstance(row, dict):
                    continue
                field_name = str(row.get("Field") or "").strip()
                if field_name:
                    columns.add(field_name)
            if columns:
                return columns
        except Exception:
            columns = set()

        try:
            cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
            if hasattr(cur, "description") and cur.description:
                columns = {str(desc[0]).strip() for desc in cur.description if desc and str(desc[0]).strip()}
        except Exception:
            columns = set()

    return columns


def _resolve_router_messages_order_column(columns: set[str]) -> str:
    """Pick the best ordering column from router-message cache table columns."""
    preferred = (
        "created_at",
        "updated_at",
        "event_time",
        "sync_time",
        "id",
    )
    for candidate in preferred:
        if candidate in columns:
            return candidate
    return ""


def _resolve_router_messages_tracking_column(columns: set[str]) -> str:
    preferred = _read_with_aliases("ROUTER_MESSAGES_TRACKING_COLUMN", "MYSQL_ROUTER_MESSAGES_TRACKING_COLUMN")
    if preferred and preferred in columns:
        return preferred
    return _resolve_tracking_number_column(columns)


def _resolve_router_messages_payload_column(columns: set[str]) -> str:
    preferred = _read_with_aliases("ROUTER_MESSAGES_PAYLOAD_COLUMN", "MYSQL_ROUTER_MESSAGES_PAYLOAD_COLUMN")
    if preferred and preferred in columns:
        return preferred

    for candidate in (
        "router_messages",
        "route_messages",
        "route_payload",
        "payload",
        "raw_payload",
        "raw_data",
        "content",
        "message_body",
    ):
        if candidate in columns:
            return candidate
    return ""

DB_FETCH_BATCH_SIZE = max(100, int(read_config("DB_FETCH_BATCH_SIZE", "5000")))

def _require_db_env() -> None:
    config = _load_mysql_config()
    missing = []
    if not config["host"]:
        missing.append("MYSQL_HOST")
    if not config["username"]:
        missing.append("MYSQL_USERNAME")
    if not config["password"]:
        missing.append("MYSQL_PASSWORD")
    if not config["database"]:
        missing.append("MYSQL_DATABASE")
    if missing:
        raise RuntimeError(f"MySQL 环境变量未配置：{', '.join(missing)}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tracking_numbers_by_date(start_date: date, end_date: date) -> list[str]:
    # fake tracking number for testing
    # return ["ZX34043383"]

    """
    Query waybill_waybills for tracking_number where created_at is between
    [start_date 00:00:00, end_date 23:59:59.999999] (inclusive by date).
    """
    _require_db_env()

    # lazy import so the app can still run without DB deps until this mode is used
    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 pymysql。请先 pip install pymysql") from e

    if end_date < start_date:
        return []

    start_dt = datetime.combine(start_date, time.min)
    # Use an exclusive upper-bound at next-day 00:00:00 to avoid dropping rows on end_date.
    end_exclusive_dt = datetime.combine(end_date + timedelta(days=1), time.min)

    conn = _connect_mysql(pymysql)

    try:
        table_name = _resolve_tracking_source_table(conn)
        if not table_name:
            raise RuntimeError("未找到运单主表，请配置 TRACKING_SOURCE_TABLE / MYSQL_TRACKING_SOURCE_TABLE。")
        if not table_name.replace("_", "").isalnum():
            raise RuntimeError(f"运单主表名不合法：{table_name}")

        table_columns = _load_table_columns(conn, table_name)
        tracking_number_column = _resolve_tracking_number_column(table_columns)
        if not tracking_number_column:
            raise RuntimeError(
                f"表 {table_name} 中未找到运单号字段，请配置 TRACKING_NUMBER_COLUMN / MYSQL_TRACKING_NUMBER_COLUMN。"
            )

        created_at_column = _resolve_tracking_created_at_column(table_columns)

        with conn.cursor() as cur:
            sql = f"""
                SELECT DISTINCT {tracking_number_column} AS tracking_number
                FROM {table_name}
                WHERE {tracking_number_column} IS NOT NULL AND {tracking_number_column} <> ''
            """
            params: list[Any] = []
            if created_at_column:
                sql += f" AND {created_at_column} >= %s AND {created_at_column} < %s"
                params.extend([start_dt, end_exclusive_dt])
            sql += " ORDER BY tracking_number ASC"

            cur.execute(sql, params)

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
def fetch_tracking_numbers_by_delivery_window(start_date: date, end_date: date) -> list[str]:
    """
    Query waybill_waybills for tracking_number where created_at is between
    [start_date - 7 days, end_date] inclusive.
    """
    shifted_start = start_date - timedelta(days=7)
    return fetch_tracking_numbers_by_date(shifted_start, end_date)



@st.cache_data(ttl=60, show_spinner=False)
def fetch_receive_province_map(tracking_ids: tuple[str, ...]) -> dict[str, str]:
    """
    Query waybill_waybills.receive_province by tracking_number for given tracking_ids.
    """
    _require_db_env()

    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pymysql. Please run: pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    conn = _connect_mysql(pymysql)

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
        raise RuntimeError("Missing dependency: pymysql. Please run: pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    conn = _connect_mysql(pymysql)

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


@st.cache_data(ttl=60, show_spinner=False)
def fetch_router_messages_map(tracking_ids: tuple[str, ...]) -> dict[str, Any]:
    """Load latest cached router_messages JSON by tracking_number."""
    _require_db_env()

    try:
        import pymysql  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pymysql. Please run: pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    conn = _connect_mysql(pymysql)

    payload_map: dict[str, Any] = {}
    try:
        table_name = _resolve_router_messages_table(conn)
        if not table_name:
            return {}
        if not table_name.replace("_", "").isalnum():
            return {}

        table_columns = _load_table_columns(conn, table_name)
        if not table_columns:
            return {}

        tracking_column = _resolve_router_messages_tracking_column(table_columns)
        payload_column = _resolve_router_messages_payload_column(table_columns)
        if not tracking_column or not payload_column:
            return {}

        order_column = _resolve_router_messages_order_column(table_columns)
        order_sql = f"ORDER BY {tracking_column} ASC, {order_column} DESC" if order_column else f"ORDER BY {tracking_column} ASC"

        with conn.cursor() as cur:
            chunk_size = 500
            for i in range(0, len(tracking_ids_clean), chunk_size):
                chunk = tracking_ids_clean[i : i + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                sql = f"""
                    SELECT {tracking_column} AS tracking_number, {payload_column} AS router_messages
                    FROM {table_name}
                    WHERE {tracking_column} IN ({placeholders})
                    AND {payload_column} IS NOT NULL
                    {order_sql}
                """
                cur.execute(sql, chunk)

                while True:
                    rows = cur.fetchmany(DB_FETCH_BATCH_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        tracking_number = str(row.get("tracking_number") or "").strip()
                        if not tracking_number or tracking_number in payload_map:
                            continue

                        raw_payload = row.get("router_messages")
                        if raw_payload is None:
                            continue

                        if isinstance(raw_payload, (dict, list)):
                            payload_map[tracking_number] = raw_payload
                            continue

                        text_payload = str(raw_payload).strip()
                        if not text_payload:
                            continue

                        try:
                            payload_map[tracking_number] = json.loads(text_payload)
                        except json.JSONDecodeError:
                            # Keep as raw text so upper layer can report parse failure.
                            payload_map[tracking_number] = text_payload
    finally:
        conn.close()

    return payload_map


def clear_query_caches() -> None:
    """Clear all DB query caches so users can fetch the latest updated records immediately."""
    for fn in (
        fetch_tracking_numbers_by_date,
        fetch_tracking_numbers_by_delivery_window,
        fetch_receive_province_map,
        fetch_sender_info_map,
        fetch_router_messages_map,
    ):
        fn.clear()
