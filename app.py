import pandas as pd
import pymysql
import streamlit as st


import pymysql
import streamlit as st
import pandas as pd

def get_conn():
    host = st.secrets.get("MYSQL_HOST", "")
    port = int(st.secrets.get("MYSQL_PORT", 3306))
    user = st.secrets.get("MYSQL_USERNAME", "")
    password = st.secrets.get("MYSQL_PASSWORD", "")
    db = st.secrets.get("MYSQL_DATABASE", "")

    # 可选 SSL（有些公司库强制 TLS）
    ssl = None
    if st.secrets.get("MYSQL_SSL", "false").lower() == "true":
        # 最简单的“启用 TLS”方式：ssl={}
        # 如果你们有 CA 文件，再扩展为 ssl={"ca": "..."} 等
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
        # 关键：把错误码显示出来（不显示敏感内容）
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
def main():
    st.set_page_config(page_title="DB Table Explorer", layout="wide")
    st.title("MySQL 表名/字段快速查看")

    # 选库（默认用 secrets 里的库）
    default_db = st.secrets.get("MYSQL_DATABASE", "")
    st.caption(f"当前默认数据库：{default_db}")

    col1, col2 = st.columns([2, 3])
    with col1:
        schema = st.text_input("Schema / Database", value=default_db, help="一般不用改，除非你有权限看多个库")
        keyword = st.text_input("表名关键词筛选（可选）", value="", placeholder="例如：status, tracking, event, pod ...")
        refresh = st.button("刷新表列表", type="primary")

    # 表列表
    if refresh or True:
        sql = """
        SELECT table_name, table_rows, data_length, index_length, create_time, update_time
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY (data_length + index_length) DESC, table_name ASC
        """
        tables_df = query_df(sql, (schema,))
        if keyword.strip():
            tables_df = tables_df[tables_df["table_name"].str.contains(keyword.strip(), case=False, na=False)]

        st.subheader("Tables")
        st.dataframe(tables_df, use_container_width=True, height=380)

        if tables_df.empty:
            st.warning("没有查到表。可能是 schema 名不对，或你的账号没有权限查看 information_schema。")
            return

        with col2:
            table_names = tables_df["table_name"].tolist()
            picked = st.selectbox("选择一个表查看字段 & 预览", table_names)

    st.divider()

    # 字段信息
    st.subheader(f"Columns — `{schema}.{picked}`")
    cols_sql = """
    SELECT
        ordinal_position,
        column_name,
        column_type,
        is_nullable,
        column_key,
        extra,
        column_default
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position
    """
    cols_df = query_df(cols_sql, (schema, picked))
    st.dataframe(cols_df, use_container_width=True, height=320)

    # 预览数据
    st.subheader("Preview (Top N)")
    n = st.slider("预览行数", 5, 200, 30)

    # 注意：表名/库名不能用参数占位符，必须拼接；所以做简单校验防注入
    def safe_ident(x: str) -> str:
        if not x or any(c in x for c in [" ", ";", "--", "/*", "*/", "`", '"', "'"]):
            raise ValueError("非法表名/库名字符")
        return x

    try:
        safe_schema = safe_ident(schema)
        safe_table = safe_ident(picked)
        preview_sql = f"SELECT * FROM `{safe_schema}`.`{safe_table}` LIMIT {int(n)}"
        preview_df = query_df(preview_sql)
        st.dataframe(preview_df, use_container_width=True, height=420)
    except Exception as e:
        st.error(f"预览失败：{e}")

    st.divider()

    # 快速定位“轨迹/状态”相关表
    st.subheader("快速搜索：可能的轨迹表（按字段名匹配）")
    hint = st.text_input("字段关键词（例如 tracking_id / status / event / ofd / pod / ts ）", value="tracking")
    if st.button("按字段关键词搜索"):
        search_sql = """
        SELECT table_name, GROUP_CONCAT(column_name ORDER BY ordinal_position SEPARATOR ', ') AS matched_columns
        FROM information_schema.columns
        WHERE table_schema = %s
          AND (LOWER(column_name) LIKE %s OR LOWER(column_type) LIKE %s)
        GROUP BY table_name
        ORDER BY table_name
        """
        like = f"%{hint.lower()}%"
        hit_df = query_df(search_sql, (schema, like, like))
        st.dataframe(hit_df, use_container_width=True, height=420)


if __name__ == "__main__":
    main()

