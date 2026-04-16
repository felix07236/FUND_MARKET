import oracledb
import pandas as pd

# -----------------------------
# 1. 数据库连接
# -----------------------------
def get_oracle_conn():
    dsn = oracledb.makedsn(
        host="10.150.0.73",
        port=1521,
        service_name="demo"
    )
    return oracledb.connect(
        user="data_mart_04",
        password="8EFmdaej2JeKwt",
        dsn=dsn
    )

# -----------------------------
# 2. 获取基金规模数据（单位：元）
# -----------------------------
def fetch_fund_scale() -> pd.DataFrame:
    sql = """
    SELECT
        PRD_CODE,
        PRD_TYP,
        NAV_DT,
        TTL_NAVL
    FROM DATA_MART_04.FIN_PRD_NAV
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df

# -----------------------------
# 3. 计算规模变动率（环比）
# -----------------------------
def calc_scale_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PRD_CODE", "NAV_DT"])

    df["规模变动率"] = (
        df.groupby("PRD_CODE")["TTL_NAVL"]
          .pct_change()
          .fillna(0)   # ← 关键
    )
    return df

# -----------------------------
# 4. 主流程
# -----------------------------
if __name__ == "__main__":
    df = fetch_fund_scale()
    df = calc_scale_change(df)

    # 中文列名
    result = df.rename(columns={
        "PRD_CODE": "产品代码",
        "PRD_TYP": "产品类型",
        "NAV_DT": "净值日期",
        "TTL_NAVL": "基金规模（元）"
    })


    result["基金规模（元）"] = result["基金规模（元）"].round(2)
    result["规模变动率"] = (
        result["规模变动率"]
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    result.to_csv(
        "基金规模变动率.csv",
        index=False,
        encoding="utf-8-sig"
    )
