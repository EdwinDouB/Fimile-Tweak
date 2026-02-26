import pandas as pd
import pymysql
import streamlit as st


def get_conn():
    host = st.secrets.get("MYSQL_HOST", "")
    port = int(st.secrets.get("MYSQL_PORT", 3306))
    user = st.secrets.get("MYSQL_USERNAME", "")
    password = st.secrets.get("MYSQL_PASSWORD", "")
    db = st.secrets.get("MYSQL_DATABASE", "")

    ssl = None
    if st.secrets.get("MYSQL_SSL", "false").lower() == "true":
        ssl = {}

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        ssl=ssl,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
        autocommit=True,
    )


def query_df(sql: str, params=None) -> pd.DataFrame:
    try:
        conn = get_conn()
    except Exception as e:
        code = getattr(e, "args", [None])[0]
        st.error(f"MySQL 连接失败。错误码={code}（去 Streamlit Cloud logs 看原始错误原因）")
        raise

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
        return pd.DataFrame(rows)
    finally:
        conn.close()


def safe_ident(name: str) -> str:
    if not name or any(c in name for c in [" ", ";", "--", "/*", "*/", "`", '"', "'"]):
        raise ValueError(f"非法标识符: {name}")
    return name


def list_tables(schema: str) -> list[str]:
    sql = """
    SELECT TABLE_NAME
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY TABLE_NAME
    """
    df = query_df(sql, (schema,))
    if df.empty:
        return []
    return df["TABLE_NAME"].tolist()


def list_columns(schema: str, TABLE_NAME: str) -> list[str]:
    sql = """
    SELECT COLUMN_NAME
    FROM information_schema.columns
    WHERE table_schema = %s AND TABLE_NAME = %s
    ORDER BY ordinal_position
    """
    df = query_df(sql, (schema, TABLE_NAME))
    if df.empty:
        return []
    return df["COLUMN_NAME"].tolist()


def maybe_pick(default: str, options: list[str]) -> str:
    return default if default in options else options[0]


def build_and_run_ofd_report(
    schema: str,
    TABLE_NAME: str,
    selected_date,
    package_col: str,
    status_col: str,
    time_col: str,
    region_col: str,
    hub_col: str,
    dsp_col: str,
    driver_col: str,
    bad_pod_col: str | None,
    ofd_keyword: str,
    failed_keyword: str,
    success_keywords: list[str],
) -> pd.DataFrame:
    schema_i = safe_ident(schema)
    table_i = safe_ident(TABLE_NAME)

    package_i = safe_ident(package_col)
    status_i = safe_ident(status_col)
    time_i = safe_ident(time_col)
    region_i = safe_ident(region_col)
    hub_i = safe_ident(hub_col)
    dsp_i = safe_ident(dsp_col)
    driver_i = safe_ident(driver_col)

    bad_pod_sql = "0 AS bad_pod"
    if bad_pod_col and bad_pod_col != "(None)":
        bad_pod_i = safe_ident(bad_pod_col)
        bad_pod_sql = f"COALESCE(t.`{bad_pod_i}`, 0) AS bad_pod"

    success_clause = " OR ".join(["LOWER(t2.`" + status_i + "`) LIKE %s" for _ in success_keywords])

    sql = f"""
    WITH ofd AS (
        SELECT
            t.`{package_i}` AS package_id,
            t.`{region_i}` AS region,
            t.`{hub_i}` AS hub,
            t.`{dsp_i}` AS dsp,
            t.`{driver_i}` AS driver_name,
            {bad_pod_sql},
            MIN(t.`{time_i}`) AS ofd_time
        FROM `{schema_i}`.`{table_i}` t
        WHERE DATE(t.`{time_i}`) = %s
          AND LOWER(t.`{status_i}`) LIKE %s
        GROUP BY
            t.`{package_i}`,
            t.`{region_i}`,
            t.`{hub_i}`,
            t.`{dsp_i}`,
            t.`{driver_i}`,
            bad_pod
    ),
    classified AS (
        SELECT
            o.*,
            MAX(
                CASE
                    WHEN LOWER(t2.`{status_i}`) LIKE %s
                     AND t2.`{time_i}` >= o.ofd_time
                     AND t2.`{time_i}` <= DATE_ADD(o.ofd_time, INTERVAL 24 HOUR)
                    THEN 1 ELSE 0
                END
            ) AS has_failed,
            MAX(
                CASE
                    WHEN ({success_clause})
                     AND t2.`{time_i}` >= o.ofd_time
                     AND t2.`{time_i}` <= DATE_ADD(o.ofd_time, INTERVAL 24 HOUR)
                    THEN 1 ELSE 0
                END
            ) AS has_success
        FROM ofd o
        LEFT JOIN `{schema_i}`.`{table_i}` t2
          ON t2.`{package_i}` = o.package_id
        GROUP BY
            o.package_id,
            o.region,
            o.hub,
            o.dsp,
            o.driver_name,
            o.bad_pod,
            o.ofd_time
    )
    SELECT
        DATE(%s) AS `Date`,
        region AS Region,
        hub AS Hub,
        dsp AS DSP,
        driver_name AS Driver_Name,
        COUNT(*) AS total_packages,
        SUM(CASE WHEN has_failed = 0 AND has_success = 1 THEN 1 ELSE 0 END) AS total_successful,
        SUM(CASE WHEN has_failed = 1 THEN 1 ELSE 0 END) AS total_failed,
        SUM(CASE WHEN has_failed = 0 AND has_success = 0 THEN 1 ELSE 0 END) AS total_unfinished,
        SUM(CASE WHEN bad_pod IN (1, '1', 'Y', 'y', 'true', 'TRUE', 'bad', 'BAD') THEN 1 ELSE 0 END) AS bad_POD
    FROM classified
    GROUP BY region, hub, dsp, driver_name
    ORDER BY total_packages DESC, Driver_Name ASC
    """

    params = [
        str(selected_date),
        f"%{ofd_keyword.lower()}%",
        f"%{failed_keyword.lower()}%",
    ]
    params.extend([f"%{k.lower()}%" for k in success_keywords])
    params.append(str(selected_date))

    return query_df(sql, tuple(params))


def main():
    st.set_page_config(page_title="OFD 24h Outcome Analyzer", layout="wide")
    st.title("OFD 24小时结果分析")

    default_db = st.secrets.get("MYSQL_DATABASE", "")
    st.caption(f"当前默认数据库：{default_db}")

    schema = st.text_input("Schema / Database", value=default_db)
    if not schema:
        st.warning("请先输入数据库名。")
        st.stop()

    try:
        tables = list_tables(schema)
    except Exception as e:
        st.error(f"读取表列表失败：{e}")
        st.stop()

    if not tables:
        st.warning("当前 schema 下没有可见的表。")
        st.stop()

    default_table = "station_delivery_details" if "station_delivery_details" in tables else tables[0]
    TABLE_NAME = st.selectbox("轨迹/派送事件表", tables, index=tables.index(default_table))

    try:
        columns = list_columns(schema, TABLE_NAME)
    except Exception as e:
        st.error(f"读取字段失败：{e}")
        st.stop()

    if not columns:
        st.warning("选中的表没有字段。")
        st.stop()

    st.subheader("1) 规则与字段映射")
    selected_date = st.date_input("选择日期（查找当天 Out for delivery）")

    c1, c2, c3 = st.columns(3)
    with c1:
        package_col = st.selectbox("包裹唯一ID字段", columns, index=columns.index(maybe_pick("waybill_no", columns)))
        status_col = st.selectbox("状态字段", columns, index=columns.index(maybe_pick("status", columns)))
        time_col = st.selectbox("状态时间字段", columns, index=columns.index(maybe_pick("created_at", columns)))
    with c2:
        region_col = st.selectbox("Region字段", columns, index=columns.index(maybe_pick("region", columns)))
        hub_col = st.selectbox("Hub字段", columns, index=columns.index(maybe_pick("hub", columns)))
        dsp_col = st.selectbox("DSP字段", columns, index=columns.index(maybe_pick("dsp", columns)))
    with c3:
        driver_col = st.selectbox("Driver_Name字段", columns, index=columns.index(maybe_pick("driver_name", columns)))
        bad_pod_options = ["(None)"] + columns
        bad_pod_col = st.selectbox("bad_POD字段（可选）", bad_pod_options)

    st.subheader("2) 状态关键词")
    k1, k2, k3 = st.columns(3)
    with k1:
        ofd_keyword = st.text_input("Out for delivery 关键词", value="out for delivery")
    with k2:
        failed_keyword = st.text_input("Failed 关键词", value="failed")
    with k3:
        success_text = st.text_input("Successful 关键词（逗号分隔）", value="successful,delivered")

    success_keywords = [x.strip() for x in success_text.split(",") if x.strip()]
    if not success_keywords:
        st.error("Successful 关键词不能为空。")
        st.stop()

    if st.button("生成汇总表", type="primary"):
        try:
            result_df = build_and_run_ofd_report(
                schema=schema,
                TABLE_NAME=TABLE_NAME,
                selected_date=selected_date,
                package_col=package_col,
                status_col=status_col,
                time_col=time_col,
                region_col=region_col,
                hub_col=hub_col,
                dsp_col=dsp_col,
                driver_col=driver_col,
                bad_pod_col=bad_pod_col if bad_pod_col != "(None)" else None,
                ofd_keyword=ofd_keyword,
                failed_keyword=failed_keyword,
                success_keywords=success_keywords,
            )
            st.subheader("结果")
            st.dataframe(result_df, use_container_width=True, height=500)
            st.download_button(
                "下载 CSV",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"ofd_outcome_{selected_date}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"生成失败：{e}")


if __name__ == "__main__":
    main()



