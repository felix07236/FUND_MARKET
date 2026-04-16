import numpy as np
import oracledb
import pandas as pd

# -----------------------------# 1. 数据库连接# -----------------------------
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

# -----------------------------# 2. 获取数据# -----------------------------
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
    SELECT
        PRD_CODE,
        PRD_TYP,
        NAV_ADD_RAT,
        NAV_DT,
        AGGR_UNIT_NVAL
    FROM DATA_MART_04.FIN_PRD_NAV
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


def fetch_hs300_quot() -> pd.DataFrame:
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


def fetch_pty_prd_base_info() -> pd.DataFrame:
    """获取产品基础信息表"""
    sql = """
    SELECT
        PRD_CODE,
        FOUND_DT,
        PRD_NAME,
        PRD_FULL_NAME,
        PRD_TYP
    FROM DATA_MART_04.PTY_PRD_BASE_INFO
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["FOUND_DT"] = pd.to_datetime(
        df["FOUND_DT"],
        format="%Y%m%d",
        errors="coerce"
    )
    return df


# =========================================================
# PRD_CODE 的产品类型填充
# =========================================================
def fill_special_prd_typ(df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:

    df = df_nav.copy()

    # 检查基础信息表中是否有 PRD_TYP 字段
    if "PRD_TYP" in df_base_info.columns:
        # 从基础信息表获取产品类型映射
        base_prd_typ_map = (
            df_base_info[df_base_info["PRD_TYP"].notna()]
            .groupby("PRD_CODE")["PRD_TYP"]
            .first()
            .to_dict()
        )

        # 优先使用基础信息表中的产品类型
        mask_base = df["PRD_TYP"].isna() & df["PRD_CODE"].isin(base_prd_typ_map)
        df.loc[mask_base, "PRD_TYP"] = df.loc[mask_base, "PRD_CODE"].map(base_prd_typ_map)

    # 按产品代码分组，获取每个产品的非空产品类型
    prd_typ_map = (
        df[df["PRD_TYP"].notna()]
        .groupby("PRD_CODE")["PRD_TYP"]
        .first()
        .to_dict()
    )
    mask = df["PRD_TYP"].isna() & df["PRD_CODE"].isin(prd_typ_map)
    df.loc[mask, "PRD_TYP"] = df.loc[mask, "PRD_CODE"].map(prd_typ_map)
    df["PRD_TYP"] = df["PRD_TYP"].fillna("__UNKNOWN__")
    return df


# -----------------------------
# 3. 收益走势（日频）
# -----------------------------
def calc_return_series(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    df_nav = df_nav.sort_values(["PRD_CODE", "NAV_DT"]).copy()

    # ========== 数据清洗：只保留成立日之后的净值数据 ==========
    df_nav_clean = pd.DataFrame()

    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取该产品的成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            # 如果没有成立日信息，保留所有数据
            df_nav_clean = pd.concat([df_nav_clean, grp])
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 只保留成立日之后的数据
        grp_clean = grp[grp["NAV_DT"] >= fund_established_dt].copy()
        if not grp_clean.empty:
            df_nav_clean = pd.concat([df_nav_clean, grp_clean])

    df_nav = df_nav_clean

    # ========== 计算基金收益 ==========
    df_nav["基金收益"] = df_nav.groupby("PRD_CODE")["AGGR_UNIT_NVAL"].pct_change()

    # ========== 处理成立日的收益：将成立日前一天净值设为 1.0 计算成立日收益 ==========
    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取该产品的成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]
        
        # 找到成立日在该产品数据中的位置
        prd_mask = (df_nav["PRD_CODE"] == prd_code)
        established_mask = (df_nav["NAV_DT"] == fund_established_dt)
        
        # 获取成立日的净值
        established_nav = df_nav.loc[prd_mask & established_mask, "AGGR_UNIT_NVAL"].values
        if len(established_nav) > 0:
            # 成立日收益 = (成立日净值 / 1.0) - 1
            fund_ret = established_nav[0] / 1.0 - 1
            # 设置成立日的收益
            df_nav.loc[prd_mask & established_mask, "基金收益"] = fund_ret

    df_idx = df_idx.sort_values("TRD_DT").copy()
    df_idx["沪深 300 收益"] = df_idx["CLS_PRC"] / df_idx["PREV_CLS_PRC"] - 1

    df = df_nav.merge(
        df_idx[["TRD_DT", "沪深 300 收益"]],
        left_on="NAV_DT",
        right_on="TRD_DT",
        how="left"
    )

    peer_avg = (
        df.groupby(["PRD_TYP", "NAV_DT"])["基金收益"]
          .mean()
          .reset_index(name="同类平均收益")
    )

    df = df.merge(peer_avg, on=["PRD_TYP", "NAV_DT"], how="left")

    return df[[
        "PRD_CODE",
        "PRD_TYP",
        "NAV_DT",
        "基金收益",
        "沪深 300 收益",
        "同类平均收益"
    ]].rename(columns={
        "PRD_CODE": "产品代码",
        "PRD_TYP": "产品类型",
        "NAV_DT": "日期"
    })

# -----------------------------
# 4. 区间收益（月度 / 季度 / 年度）
# -----------------------------
def calc_interval_return_by_nav(
        df: pd.DataFrame,
        start_nav: float,
        end_nav: float
) -> float:

    if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
        return np.nan

    return (end_nav / start_nav) - 1


def calc_periodic_return(
        df_nav: pd.DataFrame,
        df_idx: pd.DataFrame,
        df_base_info: pd.DataFrame,
        freq: str
) -> pd.DataFrame:
    df_nav = df_nav.copy()

    # ========== 数据清洗：只保留成立日之后的净值数据 ==========
    df_nav_clean = pd.DataFrame()

    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取该产品的成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            # 如果没有成立日信息，保留所有数据
            df_nav_clean = pd.concat([df_nav_clean, grp])
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 只保留成立日之后的数据
        grp_clean = grp[grp["NAV_DT"] >= fund_established_dt].copy()
        if not grp_clean.empty:
            df_nav_clean = pd.concat([df_nav_clean, grp_clean])

    df_nav = df_nav_clean

    df_nav["周期"] = df_nav["NAV_DT"].dt.to_period(freq)

    all_periods = sorted(df_nav["周期"].unique())

    period_end_nav = {}
    for period in all_periods:
        period_str = str(period)
        period_data = df_nav[df_nav["周期"] == period]
        if not period_data.empty:
            last_day = period_data["NAV_DT"].max()
            last_nav = period_data.loc[
                period_data["NAV_DT"] == last_day,
                "AGGR_UNIT_NVAL"
            ].values[0]
            period_end_nav[period_str] = last_nav

    period_idx_ret = {}
    for period, g_idx in df_idx.groupby(df_idx["TRD_DT"].dt.to_period(freq)):
        if len(g_idx) < 2:
            period_idx_ret[str(period)] = 0
            continue

        g_idx = g_idx.sort_values("TRD_DT")
        ret = g_idx["CLS_PRC"].iloc[-1] / g_idx["CLS_PRC"].iloc[0] - 1
        period_idx_ret[str(period)] = ret

    records = []

    prev_period_end_nav = {}

    for i, period in enumerate(all_periods):
        period_str = str(period)

        for (prd_code, prd_typ), g in df_nav[df_nav["周期"] == period].groupby(
                ["PRD_CODE", "PRD_TYP"]
        ):
            if len(g) < 1:
                continue

            end_dt = g["NAV_DT"].max()
            end_nav = g.loc[g["NAV_DT"] == end_dt, "AGGR_UNIT_NVAL"].values[0]

            # 从基础信息表获取该产品的成立日
            base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
            fund_established_dt = base_info.iloc[0]["FOUND_DT"] if len(base_info) > 0 and pd.notna(base_info.iloc[0]["FOUND_DT"]) else None

            # 检查当前周期是否包含成立日
            period_start_date = period.start_time
            period_end_date = period.end_time
            contains_established_dt = False
            if fund_established_dt is not None:
                contains_established_dt = (fund_established_dt >= period_start_date) and (fund_established_dt <= period_end_date)

            if contains_established_dt:
                # 周期包含成立日：使用成立日前一天作为起始点，净值设为1.0
                virtual_start_dt = fund_established_dt - pd.Timedelta(days=1)
                
                # 基金收益：使用虚拟净值 1.0
                start_nav = 1.0
                
                # 指数收益：使用虚拟起始点或之前最近的交易日数据，与基金收益时间严格对齐
                idx_before = df_idx[df_idx["TRD_DT"] <= virtual_start_dt]
                if len(idx_before) > 0:
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
                else:
                    # 如果找不到之前的数据，使用指数最早数据
                    idx_start = df_idx["TRD_DT"].min()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
            elif i == 0 and fund_established_dt is not None:
                # 第一个周期但不包含成立日（成立日在周期之前）：使用虚拟起始点
                virtual_start_dt = fund_established_dt - pd.Timedelta(days=1)
                
                # 基金收益：使用虚拟净值 1.0
                start_nav = 1.0
                
                # 指数收益：使用虚拟起始点或之前最近的交易日数据
                idx_before = df_idx[df_idx["TRD_DT"] <= virtual_start_dt]
                if len(idx_before) > 0:
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
                else:
                    # 如果找不到之前的数据，使用指数最早数据
                    idx_start = df_idx["TRD_DT"].min()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
            elif i == 0:
                # 第一个周期且没有成立日信息：使用实际数据
                period_start_dt = g["NAV_DT"].min()
                start_nav = g.loc[g["NAV_DT"] == period_start_dt, "AGGR_UNIT_NVAL"].values[0]
                
                # 指数收益：使用与基金收益时间严格对齐的起始点
                idx_before = df_idx[df_idx["TRD_DT"] <= period_start_dt]
                if len(idx_before) > 0:
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
                else:
                    idx_ret = 0
            else:
                # 非第一个周期：使用上一周期末的净值
                prev_period = str(all_periods[i - 1])
                key = (prd_code, prev_period)
                
                if key in prev_period_end_nav:
                    start_nav = prev_period_end_nav[key]
                    # 需要获取 prev_last_day 用于指数收益计算
                    prev_period_data = df_nav[
                        (df_nav["PRD_CODE"] == prd_code) &
                        (df_nav["周期"] == all_periods[i - 1])
                    ]
                    if not prev_period_data.empty:
                        prev_last_day = prev_period_data["NAV_DT"].max()
                else:
                    prev_period_data = df_nav[
                        (df_nav["PRD_CODE"] == prd_code) &
                        (df_nav["周期"] == all_periods[i - 1])
                        ]
                    if not prev_period_data.empty:
                        prev_last_day = prev_period_data["NAV_DT"].max()
                        start_nav = prev_period_data.loc[
                            prev_period_data["NAV_DT"] == prev_last_day,
                            "AGGR_UNIT_NVAL"
                        ].values[0]
                        prev_period_end_nav[key] = start_nav
                    else:
                        start_nav = end_nav
                        prev_last_day = end_dt
                
                # 指数收益：使用与基金收益时间严格对齐的起始点（上一周期末最后一个交易日）
                idx_before = df_idx[df_idx["TRD_DT"] <= prev_last_day]
                if len(idx_before) > 0:
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= end_dt)
                    ].copy()
                    
                    if len(idx_sub) >= 2:
                        idx_sub = idx_sub.sort_values("TRD_DT")
                        idx_ret = idx_sub["CLS_PRC"].iloc[-1] / idx_sub["CLS_PRC"].iloc[0] - 1
                    else:
                        idx_ret = 0
                else:
                    idx_ret = 0

            fund_ret = calc_interval_return_by_nav(
                df=None,
                start_nav=start_nav,
                end_nav=end_nav
            )

            prev_period_end_nav[(prd_code, period_str)] = end_nav

            records.append({
                "产品代码": prd_code,
                "产品类型": prd_typ,
                "周期": period_str,
                "基金收益": fund_ret,
                "沪深 300 收益": idx_ret
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_idx = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()

    # 使用基础信息表填充产品类型
    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 收益走势
    series_df = calc_return_series(df_nav, df_idx, df_base_info)

    # 月度 / 季度 / 年度
    periodic_df = pd.concat([
        calc_periodic_return(df_nav, df_idx, df_base_info, "M"),
        calc_periodic_return(df_nav, df_idx, df_base_info, "Q"),
        calc_periodic_return(df_nav, df_idx, df_base_info, "Y")
    ], ignore_index=True)

    # 同类平均
    peer_avg = (
        periodic_df.groupby(["产品类型", "周期"])["基金收益"]
        .mean()
        .reset_index(name="同类平均收益")
    )

    peer_avg["同类平均收益"] = peer_avg["同类平均收益"].fillna(0)

    periodic_df = periodic_df.merge(
        peer_avg, on=["产品类型", "周期"], how="left"
    )

    def to_pct_str(s):
        return (s * 100).round(2).astype(str) + "%"

    # 处理 series_df 中的列
    for col in ["基金收益", "沪深 300 收益", "同类平均收益"]:
        if col in series_df.columns:
            series_df[col] = series_df[col].apply(lambda x: f"{float(x) * 100:.2f}%" if pd.notna(x) else "0.00%")

    # 处理 periodic_df 中的列
    for col in ["基金收益", "沪深 300 收益", "同类平均收益"]:
        if col in periodic_df.columns:
            periodic_df[col] = periodic_df[col].apply(lambda x: f"{float(x) * 100:.2f}%" if pd.notna(x) else "0.00%")

    # 确保所有包含百分比的列都是字符串类型，防止被误识别为日期或数字
    for col_name in ["基金收益", "沪深 300 收益", "同类平均收益"]:
        if col_name in series_df.columns:
            series_df[col_name] = series_df[col_name].astype(str)
    for col_name in ["基金收益", "沪深 300 收益", "同类平均收益"]:
        if col_name in periodic_df.columns:
            periodic_df[col_name] = periodic_df[col_name].astype(str)

    series_df.to_csv(
        "基金收益走势_产品沪深300同类平均.csv",
        index=False,
        encoding="utf-8-sig"
    )

    periodic_df.to_csv(
        "基金区间收益_月度季度年度.csv",
        index=False,
        encoding="utf-8-sig"
    )