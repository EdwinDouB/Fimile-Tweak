from utils.utils import *
import streamlit as st
from datetime import date, datetime, time, timedelta
import json
from typing import Any


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
    }


def _resolve_router_messages_table(conn: Any) -> str:
    """Pick an existing router_messages cache table, with env override support."""
    config = _load_mysql_config()
    schema = str(config["database"])
    preferred = _read_with_aliases("ROUTER_MESSAGES_TABLE", "MYSQL_ROUTER_MESSAGES_TABLE")

    candidates = [
        preferred,
        "transit_third_party_cache",
        "third_party_cache",
        "transit_router_messages_cache",
    ]
    candidates = [name.strip() for name in candidates if str(name).strip()]

    with conn.cursor() as cur:
        existing_tables: set[str] = set()

        # Prefer SHOW TABLES: some DB users have restricted access to information_schema.
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
            try:
                if candidates:
                    placeholders = ", ".join(["%s"] * len(candidates))
                    cur.execute(
                        f"""
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = %s
                            AND table_name IN ({placeholders})
                        """,
                        [schema, *candidates],
                    )
                    existing_tables = {str(row.get("table_name") or "") for row in cur.fetchall()}
            except Exception:
                existing_tables = set()

        for candidate in candidates:
            if candidate in existing_tables:
                return candidate

        if existing_tables:
            fuzzy_candidates = sorted(table for table in existing_tables if "third_party_cache" in table)
            if fuzzy_candidates:
                return fuzzy_candidates[0]

        # Last-resort lookup when metadata queries are restricted.
        for candidate in candidates:
            if not candidate.replace("_", "").isalnum():
                continue
            try:
                cur.execute(f"SELECT 1 FROM {candidate} LIMIT 1")
                return candidate
            except Exception:
                continue

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

    config = _load_mysql_config()
    conn = pymysql.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        user=str(config["username"]),
        password=str(config["password"]),
        database=str(config["database"]),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            sql = """
                SELECT DISTINCT tracking_number
                FROM waybill_waybills
                WHERE created_at >= %s AND created_at < %s
                AND tracking_number IS NOT NULL AND tracking_number <> ''
                ORDER BY tracking_number ASC
            """
            cur.execute(sql, (start_dt, end_exclusive_dt))

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

    config = _load_mysql_config()
    conn = pymysql.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        user=str(config["username"]),
        password=str(config["password"]),
        database=str(config["database"]),
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
        raise RuntimeError("Missing dependency: pymysql. Please run: pip install pymysql") from e

    if not tracking_ids:
        return {}

    tracking_ids_clean = tuple(str(tid).strip() for tid in tracking_ids if str(tid).strip())
    if not tracking_ids_clean:
        return {}

    config = _load_mysql_config()
    conn = pymysql.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        user=str(config["username"]),
        password=str(config["password"]),
        database=str(config["database"]),
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

    config = _load_mysql_config()
    conn = pymysql.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        user=str(config["username"]),
        password=str(config["password"]),
        database=str(config["database"]),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    payload_map: dict[str, Any] = {}
    try:
        table_name = _resolve_router_messages_table(conn)
        if not table_name:
            return {}
        if not table_name.replace("_", "").isalnum():
            return {}

        with conn.cursor() as cur:
            chunk_size = 500
            for i in range(0, len(tracking_ids_clean), chunk_size):
                chunk = tracking_ids_clean[i : i + chunk_size]
                placeholders = ", ".join(["%s"] * len(chunk))
                sql = f"""
                    SELECT tracking_number, router_messages, created_at
                    FROM {table_name}
                    WHERE tracking_number IN ({placeholders})
                    AND router_messages IS NOT NULL
                    ORDER BY tracking_number ASC, created_at DESC
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
