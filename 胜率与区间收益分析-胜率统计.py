import oracledb
import pandas as pd

# =========================================================
# 1. 数据库连接
# =========================================================
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


# =========================================================
# 2. 获取数据
# =========================================================
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
          SELECT PRD_TYP, 
                 UNIT_NVAL, 
                 AGGR_UNIT_NVAL, 
                 NAV_DT, 
                 PRD_CODE
          FROM DATA_MART_04.FIN_PRD_NAV 
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(
        df["NAV_DT"].astype(str), format="%Y%m%d"
    )
    return df


def fetch_hs300_quot() -> pd.DataFrame:
    sql = """
          SELECT SECU_ID, 
                 TRD_DT, 
                 CLS_PRC, 
                 PREV_CLS_PRC
          FROM DATA_MART_04.VAR_SECU_DQUOT
          WHERE SECU_ID = '000300.IDX.CSIDX' 
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["TRD_DT"] = pd.to_datetime(
        df["TRD_DT"].astype(str), format="%Y%m%d"
    )
    return df


def fetch_pty_prd_base_info() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE, 
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

    return df


# =========================================================
# 3. 产品级计算基准日
# =========================================================
def get_product_end_dt(df_nav: pd.DataFrame) -> pd.DataFrame:
    return (
        df_nav.groupby("PRD_CODE", as_index=False)["NAV_DT"]
        .max()
        .rename(columns={"NAV_DT": "END_DT"})
    )


# =========================================================
# 4. 周度收益计算
# =========================================================
def calc_weekly_return(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算周度收益（基金和沪深300）
    """
    results = []

    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 数据清洗：过滤掉成立日之前的脏数据
        grp_clean = grp[grp["NAV_DT"] >= fund_established_dt].copy()

        if grp_clean.empty:
            continue

        # 按周分组
        grp_clean["WEEK"] = grp_clean["NAV_DT"].dt.to_period("W")

        for week, week_data in grp_clean.groupby("WEEK"):
            week_data = week_data.sort_values("NAV_DT")

            if len(week_data) < 2:
                continue

            # 计算基金周度收益
            start_nav = week_data.iloc[0]["AGGR_UNIT_NVAL"]
            end_nav = week_data.iloc[-1]["AGGR_UNIT_NVAL"]

            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
                continue

            fund_return = (end_nav - start_nav) / start_nav

            # 计算沪深300周度收益
            week_start = week_data.iloc[0]["NAV_DT"]
            week_end = week_data.iloc[-1]["NAV_DT"]

            idx_data = df_idx[
                (df_idx["TRD_DT"] >= week_start) &
                (df_idx["TRD_DT"] <= week_end)
            ].copy()

            if len(idx_data) < 2:
                continue

            idx_data = idx_data.sort_values("TRD_DT")
            idx_start_price = idx_data.iloc[0]["CLS_PRC"]
            idx_end_price = idx_data.iloc[-1]["CLS_PRC"]

            if pd.isna(idx_start_price) or pd.isna(idx_end_price) or idx_start_price == 0:
                continue

            idx_return = (idx_end_price - idx_start_price) / idx_start_price

            results.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": week_data.iloc[0]["PRD_TYP"],
                "WEEK": str(week),
                "WEEK_START": week_start,
                "WEEK_END": week_end,
                "FUND_RETURN": fund_return,
                "IDX_RETURN": idx_return,
                "EXCESS_RETURN": fund_return - idx_return
            })

    return pd.DataFrame(results)


# =========================================================
# 5. 胜率统计
# =========================================================
def calc_win_rate(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    计算周度超额胜率和绝对胜率
    """
    results = []

    for (prd_code, prd_typ), grp in df_weekly.groupby(["PRD_CODE", "PRD_TYP"]):
        # 统计周期数
        total_periods = len(grp)

        # 统计跑赢次数（超额收益 > 0）
        win_count = len(grp[grp["EXCESS_RETURN"] > 0])

        # 统计跑输次数（超额收益 <= 0）
        lose_count = len(grp[grp["EXCESS_RETURN"] <= 0])

        # 统计绝对胜率次数（基金收益 > 0）
        positive_count = len(grp[grp["FUND_RETURN"] > 0])

        # 计算超额胜率
        excess_win_rate = win_count / total_periods if total_periods > 0 else 0

        # 计算绝对胜率
        absolute_win_rate = positive_count / total_periods if total_periods > 0 else 0

        results.append({
            "PRD_CODE": prd_code,
            "PRD_TYP": prd_typ,
            "统计周期数": total_periods,
            "跑赢次数": win_count,
            "跑输次数": lose_count,
            "超额胜率": excess_win_rate,
            "绝对胜率": absolute_win_rate
        })

    return pd.DataFrame(results)


# =========================================================
# 6. 主流程
# =========================================================
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_hs300 = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()
    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 计算周度收益
    df_weekly = calc_weekly_return(df_nav, df_hs300, df_base_info)

    if df_weekly.empty:
        print("没有可用的周度收益数据")
        exit()

    # 计算胜率
    df_win_rate = calc_win_rate(df_weekly)

    # 格式化输出
    df_win_rate["超额胜率"] = df_win_rate["超额胜率"].apply(lambda x: f"{x:.2%}")
    df_win_rate["绝对胜率"] = df_win_rate["绝对胜率"].apply(lambda x: f"{x:.2%}")

    # 输出结果
    output_file = "胜率统计结果.txt"
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("📊 周度胜率统计结果")
    output_lines.append(f"✅ 共统计产品数量：{len(df_win_rate)}")
    output_lines.append("=" * 80)
    output_lines.append("")

    for _, row in df_win_rate.iterrows():
        output_lines.append("-" * 80)
        for col in df_win_rate.columns:
            output_lines.append(f"{col:20s}: {row[col]}")

    output_lines.append("=" * 80)
    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n结果已保存至：{output_file}")
