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
# 2. 获取数据
# -----------------------------
def fetch_fin_prd_nav():
    sql = """
    SELECT
        PRD_CODE,
        PRD_TYP,
        NAV_DT,
        AGGR_UNIT_NVAL,
        UNIT_YLD
    FROM DATA_MART_04.FIN_PRD_NAV
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


def fetch_hs300_quot():
    sql = """
    SELECT
        TRD_DT,
        CLS_PRC,
        PREV_CLS_PRC
    FROM DATA_MART_04.VAR_SECU_DQUOT
    WHERE SECU_ID = '000300.IDX.CSIDX'
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["TRD_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
    return df

# -----------------------------
# 3. 区间收益 & 年化（月 / 季 / 年）
# -----------------------------
def calc_period_return(df_nav, df_idx, freq):
    df_nav = df_nav.copy()
    df_nav["周期"] = df_nav["NAV_DT"].dt.to_period(freq)

    records = []

    for (prd_code, prd_typ, period), g in df_nav.groupby(
        ["PRD_CODE", "PRD_TYP", "周期"]
    ):
        if len(g) < 2:
            continue

        # ---------- 基金收益（累计净值） ----------
        start_nav = g["AGGR_UNIT_NVAL"].iloc[0]
        end_nav = g["AGGR_UNIT_NVAL"].iloc[-1]
        fund_ret = end_nav / start_nav - 1

        start_dt = g["NAV_DT"].iloc[0]
        end_dt = g["NAV_DT"].iloc[-1]
        days = max((end_dt - start_dt).days, 1)

        # ---------- 基金年化 ----------

        fund_ann = (
                    g["UNIT_YLD"].sum() / len(g) * 365 / 10000
            )

        # ---------- 沪深300 ----------
        idx = df_idx[
            (df_idx["TRD_DT"] >= start_dt) &
            (df_idx["TRD_DT"] <= end_dt)
        ].sort_values("TRD_DT")

        if len(idx) < 2:
            idx_ret = None
            idx_ann = None
        else:
            idx_ret = idx["CLS_PRC"].iloc[-1] / idx["CLS_PRC"].iloc[0] - 1
            idx_ann = (1 + idx_ret) ** (365 / days) - 1

        records.append({
            "产品代码": prd_code,
            "产品类型": prd_typ,
            "周期": period.strftime("%Y-%m"),
            "基金收益": fund_ret,
            "沪深300收益": idx_ret,
            "基金年化收益": fund_ann,
            "沪深300年化收益": idx_ann
        })

    return pd.DataFrame(records)

# -----------------------------
# 4. 基金最大回撤
# -----------------------------
def calc_max_drawdown(df_nav):
    df = df_nav.copy()
    df = df.sort_values(["PRD_CODE", "NAV_DT"])

    df["MAX_NAV"] = df.groupby("PRD_CODE")["AGGR_UNIT_NVAL"].cummax()
    df["回撤"] = 1 - df["AGGR_UNIT_NVAL"] / df["MAX_NAV"]

    dd = (
        df.groupby(["PRD_CODE", "PRD_TYP"])["回撤"]
        .max()
        .reset_index(name="基金最大回撤")
    )
    dd = dd.rename(columns={
        "PRD_CODE": "产品代码",
        "PRD_TYP": "产品类型"
    })
    return dd

# -----------------------------
# 5. 同类平均
# -----------------------------
def calc_peer_avg(df):
    peer = (
        df.groupby(["产品类型", "周期"])
        .agg(
            同类平均收益=("基金收益", "mean"),
            同类平均年化收益=("基金年化收益", "mean")
        )
        .reset_index()
    )
    return peer

# -----------------------------
# 6. 主流程
# -----------------------------
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_idx = fetch_hs300_quot()

    result = pd.concat([
        calc_period_return(df_nav, df_idx, "M"),
        calc_period_return(df_nav, df_idx, "Q"),
        calc_period_return(df_nav, df_idx, "Y")
    ], ignore_index=True)

    peer_df = calc_peer_avg(result)
    result = result.merge(peer_df, on=["产品类型", "周期"], how="left")

    dd_df = calc_max_drawdown(df_nav)
    result = result.merge(dd_df, on=["产品代码", "产品类型"], how="left")

    pct_cols = [
        "基金收益", "沪深300收益",
        "基金年化收益", "沪深300年化收益",
        "同类平均收益", "同类平均年化收益",
        "基金最大回撤"
    ]

    for col in pct_cols:
        if col in result.columns:
            result[col] = (result[col] * 100).round(2).astype(str) + "%"

    result.to_csv(
        "最大回撤数据.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 基金业绩对比数据已生成")