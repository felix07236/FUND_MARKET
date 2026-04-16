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
# 2. 历史净值明细
# -----------------------------
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
    SELECT
        PRD_CODE,
        NAV_DT,
        UNIT_NVAL,
        AGGR_UNIT_NVAL
    FROM DATA_MART_04.FIN_PRD_NAV
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(
        df["NAV_DT"].astype(str), format="%Y%m%d"
    )

    return df.rename(columns={
        "PRD_CODE": "产品代码",
        "NAV_DT": "净值日期",
        "UNIT_NVAL": "单位净值",
        "AGGR_UNIT_NVAL": "累计净值"
    })

# -----------------------------
# 3. 分红拆分记录
# -----------------------------
def fetch_fin_prd_bons() -> pd.DataFrame:
    sql = """
    SELECT
        PRD_CODE,
        RT_REG_DT,
        EX_RD_DT,
        UNIT_SHR_BONS,
        SPLI_CNV_RAT
    FROM DATA_MART_04.FIN_PRD_BONS
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["RT_REG_DT"] = pd.to_datetime(
        df["RT_REG_DT"].astype(str), format="%Y%m%d"
    )
    df["EX_RD_DT"] = pd.to_datetime(
        df["EX_RD_DT"].astype(str), format="%Y%m%d"
    )

    return df.rename(columns={
        "PRD_CODE": "产品代码",
        "RT_REG_DT": "权益登记日",
        "EX_RD_DT": "拆分生效日",
        "UNIT_SHR_BONS": "每份分红",
        "SPLI_CNV_RAT": "拆分比例"
    })

# -----------------------------
# 4. 主流程
# -----------------------------
if __name__ == "__main__":

    # 历史净值明细
    nav_df = fetch_fin_prd_nav()
    nav_df.to_csv(
        "历史净值明细.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 分红拆分记录
    bons_df = fetch_fin_prd_bons()
    bons_df.to_csv(
        "分红拆分记录.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 历史净值与分红拆分数据已生成")