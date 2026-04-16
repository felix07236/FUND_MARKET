import oracledb
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. 数据库连接
# --------------------------------------------------
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


# --------------------------------------------------
# 2. 数据获取
# --------------------------------------------------
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE, \
                 PRD_TYP, \
                 NAV_DT, \
                 AGGR_UNIT_NVAL, \
                 NAV_ADD_RAT, \
                 AGGR_NAV_ADD_RAT
          FROM DATA_MART_04.FIN_PRD_NAV
          WHERE AGGR_UNIT_NVAL IS NOT NULL \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, con=conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


def fetch_index_quote(index_secu_id: str) -> pd.DataFrame:
    sql = f"""
    SELECT TRD_DT, CLS_PRC
    FROM VAR_SECU_DQUOT
    WHERE SECU_ID = '{index_secu_id}'
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, con=conn)

    df["NAV_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
    return df.rename(columns={"CLS_PRC": "INDEX_CLOSE"})


# --------------------------------------------------
# 3. 工具函数
# --------------------------------------------------
def get_period_start(end_dt: pd.Timestamp, period: str) -> pd.Timestamp:
    if period == "今年以来":
        return pd.Timestamp(year=end_dt.year, month=1, day=1)

    if period.startswith("近") and period.endswith("年"):
        try:
            n_years = int(period[1:-1])
        except ValueError:
            raise ValueError(f"无法解析周期：{period}")

        target_year = end_dt.year - n_years
        target_month = end_dt.month
        target_day = end_dt.day

        # 处理月末情况（如 2 月 30 日不存在）
        try:
            result = pd.Timestamp(
                year=target_year,
                month=target_month,
                day=target_day
            )
        except ValueError:
            # 如果日期不存在（如 2 月 30 日），使用该月最后一天
            import calendar
            last_day = calendar.monthrange(target_year, target_month)[1]
            result = pd.Timestamp(
                year=target_year,
                month=target_month,
                day=last_day
            )

        return result

    elif period == "成立以来":
        return None

    else:
        raise ValueError(f"未知周期：{period}")


# --------------------------------------------------
# 4. 周期 NAV 天数（仅用于排名资格）
# --------------------------------------------------
def calc_period_nav_days(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    records = []

    for prd_code, g in df.groupby("PRD_CODE"):
        g = g.sort_values("NAV_DT")
        base_dt = g["NAV_DT"].max()

        for period in periods:
            start_dt = (
                g["NAV_DT"].min()
                if period == "成立以来"
                else get_period_start(base_dt, period)
            )

            nav_days = g[
                (g["NAV_DT"] >= start_dt) &
                (g["NAV_DT"] <= base_dt)
                ]["NAV_DT"].nunique()

            records.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": g["PRD_TYP"].iloc[0] if pd.notna(g["PRD_TYP"].iloc[0]) else "未分类",
                "周期": period,
                "nav_days": nav_days
            })

    return pd.DataFrame(records)


# --------------------------------------------------
# 5. 风险控制指标计算
# --------------------------------------------------
def calc_risk_metrics(prd_df: pd.DataFrame, idx_df: pd.DataFrame, period: str) -> dict:
    """
    计算单个产品在指定周期内的风险控制指标

    Parameters:
    -----------
    prd_df : DataFrame
        单只产品的净值数据
    idx_df : DataFrame
        指数数据
    period : str
        周期名称

    Returns:
    --------
    dict : 包含所有风险指标的字典
    """
    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    est_dt = prd_df["NAV_DT"].min()

    # 确定起始日期
    start_dt = est_dt if period == "成立以来" else get_period_start(base_dt, period)
    sub = prd_df[prd_df["NAV_DT"] >= start_dt] if start_dt else prd_df

    if len(sub) < 2:
        return {
            "周期": period,
            "最大回撤": np.nan,
            "贝塔": np.nan,
            "回撤修复": np.nan,
            "年化波动率": np.nan,
            "下行风险": np.nan,
            "防守能力": np.nan
        }

    # 计算日收益率
    sub = sub.copy()
    sub["日收益"] = sub["AGGR_UNIT_NVAL"].pct_change()

    # ========== 1. 最大回撤 (Maximum Drawdown) ==========
    # 计算累计净值曲线
    sub["累计净值"] = sub["AGGR_UNIT_NVAL"]
    # 计算历史最高点（滚动最大值）
    sub["历史最高"] = sub["累计净值"].cummax()
    # 计算回撤
    sub["回撤"] = (sub["累计净值"] - sub["历史最高"]) / sub["历史最高"]
    # 最大回撤
    max_drawdown = sub["回撤"].min()

    # ========== 2. 贝塔系数 (Beta) ==========
    # 准备指数数据
    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= sub["NAV_DT"].min()) &
        (idx_df["NAV_DT"] <= base_dt)
        ].copy()

    if len(idx_sub) < 2:
        beta = np.nan
    else:
        # 计算指数日收益
        idx_sub = idx_sub.sort_values("NAV_DT")
        idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()

        # 合并基金和指数数据
        merged = sub.merge(idx_sub[["NAV_DT", "指数日收益"]], on="NAV_DT", how="inner")

        if len(merged) < 10:  # 至少需要 10 个交易日
            beta = np.nan
        else:
            # 去除 NaN 值
            merged = merged.dropna(subset=["日收益", "指数日收益"])

            if len(merged) < 10:
                beta = np.nan
            else:
                # 计算 Beta = Cov(Rf, Rm) / Var(Rm)
                cov = np.cov(merged["日收益"], merged["指数日收益"], ddof=1)[0, 1]
                var = np.var(merged["指数日收益"], ddof=1)
                beta = cov / var if var != 0 else 0

    # ========== 3. 回撤修复时间==========
    # 找出所有显著回撤（超过 1%）
    significant_drawdowns = sub[sub["回撤"] < -0.01].copy()

    if len(significant_drawdowns) == 0:
        recovery_time = np.nan
    else:
        # 对每个回撤点，计算到恢复的时间
        recovery_days = []
        for idx, row in significant_drawdowns.iterrows():
            # 找到该时点之后的所有日期
            future = sub[sub["NAV_DT"] >= row["NAV_DT"]].copy()
            # 找到首次恢复到或超过历史高点的日期
            recovered = future[future["累计净值"] >= row["历史最高"]]
            if len(recovered) > 0:
                days = (recovered.iloc[0]["NAV_DT"] - row["NAV_DT"]).days
                recovery_days.append(days)

        # 取平均修复时间
        recovery_time = np.mean(recovery_days) if recovery_days else np.nan

    # ========== 4. 年化波动率 (Annualized Volatility) ==========
    # 计算日收益率的标准差
    daily_vol = sub["日收益"].std()
    # 年化处理（假设一年 252 个交易日）
    ann_vol = daily_vol * np.sqrt(252) if pd.notna(daily_vol) else np.nan

    # ========== 5. 下行风险 (Downside Risk) ==========
    # 只考虑负收益的波动
    negative_returns = sub[sub["日收益"] < 0]["日收益"]

    if len(negative_returns) < 2:
        downside_risk = np.nan
    else:
        # 计算下行标准差
        downside_std = negative_returns.std()
        # 年化处理
        downside_risk = downside_std * np.sqrt(252) if pd.notna(downside_std) else np.nan

    # ========== 6. 防守能力 (Defensive Ability) ==========
    # 标准定义：1 - (基金跌幅 / 市场跌幅)，越大越好
    if "指数日收益" not in locals() or len(merged) < 10:
        defensive_ability = np.nan
    else:
        # 筛选市场下跌的交易日
        down_days = merged[merged["指数日收益"] < 0].copy()

        if len(down_days) < 5:  # 至少需要 5 个下跌交易日
            defensive_ability = np.nan
        else:
            # 计算平均跌幅（取绝对值）
            avg_fund_drop = abs(down_days["日收益"].mean())
            avg_market_drop = abs(down_days["指数日收益"].mean())

            if avg_market_drop != 0:
                # 下跌保护比率 = 1 - (基金跌幅 / 市场跌幅)
                defensive_ability = 1 - (avg_fund_drop / avg_market_drop)
            else:
                defensive_ability = np.nan

    return {
        "周期": period,
        "最大回撤": max_drawdown,
        "贝塔": beta,
        "回撤修复": recovery_time,
        "年化波动率": ann_vol,
        "下行风险": downside_risk,
        "防守能力": defensive_ability
    }


# --------------------------------------------------
# 6. 单产品多周期指标计算
# --------------------------------------------------
def calc_product_risk_metrics(prd_df: pd.DataFrame, idx_df: pd.DataFrame, periods: list) -> list:
    """
    计算单个产品在多个周期内的所有风险指标
    """
    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    est_dt = prd_df["NAV_DT"].min()

    prd_typ = prd_df["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    results = []

    for period in periods:
        metrics = calc_risk_metrics(prd_df, idx_df, period)

        results.append({
            "产品代码": prd_df["PRD_CODE"].iloc[0],
            "产品类型": prd_typ,
            **metrics
        })

    return results


# --------------------------------------------------
# 7. 同类平均 & 沪深 300（基于全部产品）
# --------------------------------------------------
def build_benchmark_and_avg(df: pd.DataFrame, idx_df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    构建同类平均和沪深 300 基准数据
    """
    rows = []

    # 计算同类平均
    for (typ, period), g in df.groupby(["产品类型", "周期"]):
        if typ == "指数":
            continue

        risk_cols = ["最大回撤", "贝塔", "年化波动率", "下行风险", "防守能力"]

        row = {
            "产品代码": "同类平均",
            "产品类型": typ,
            "周期": period,
        }

        for col in risk_cols:
            row[col] = g[col].mean()

        # 回撤修复时间取中位数
        row["回撤修复"] = g["回撤修复"].median()

        rows.append(row)

    # 计算沪深 300 基准
    base_dt = pd.to_datetime(df["NAV_DT"].max()) if "NAV_DT" in df.columns else pd.Timestamp.now()

    for period in periods:
        start_dt = get_period_start(base_dt, period)
        if start_dt is None:
            start_dt = idx_df["NAV_DT"].min()

        idx_sub = idx_df[
            (idx_df["NAV_DT"] >= start_dt) &
            (idx_df["NAV_DT"] <= base_dt)
            ].sort_values("NAV_DT")

        if len(idx_sub) < 2:
            continue

        # 计算指数的风险指标
        idx_sub = idx_sub.copy()
        idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()

        # 最大回撤
        idx_sub["累计净值"] = idx_sub["INDEX_CLOSE"]
        idx_sub["历史最高"] = idx_sub["累计净值"].cummax()
        idx_sub["回撤"] = (idx_sub["累计净值"] - idx_sub["历史最高"]) / idx_sub["历史最高"]
        idx_max_dd = idx_sub["回撤"].min()

        # 年化波动率
        idx_vol = idx_sub["指数日收益"].std() * np.sqrt(252)

        # 下行风险
        idx_negative = idx_sub[idx_sub["指数日收益"] < 0]["指数日收益"]
        idx_down_risk = idx_negative.std() * np.sqrt(252) if len(idx_negative) > 0 else np.nan

        rows.append({
            "产品代码": "沪深 300",
            "产品类型": "指数",
            "周期": period,
            "最大回撤": idx_max_dd,
            "贝塔": 1.0,  # 指数的 Beta 为 1
            "回撤修复": np.nan,
            "年化波动率": idx_vol,
            "下行风险": idx_down_risk,
            "防守能力": 1.0  # 指数自身作为基准
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------
# 8. 排名
# --------------------------------------------------
def rank_risk_df(df: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    """
    对风险指标进行排名
    """
    # 风险指标：越小越好（除了防守能力越大越好）
    smaller_is_better = ["最大回撤", "贝塔", "回撤修复", "年化波动率", "下行风险"]
    larger_is_better = ["防守能力"]

    # 初始化排名列为空
    for col in smaller_is_better + larger_is_better:
        df[f"{col}排名"] = np.nan

    # 创建合格性索引
    idx = df.set_index(["产品代码", "周期"]).index
    elig_idx = eligibility.set_index(["PRD_CODE", "周期"]).index
    eligible = idx.isin(elig_idx)

    # 对每个指标进行排名
    for col in smaller_is_better:
        if not eligible.any():
            continue

        rank_vals = (
            df.loc[eligible]
            .groupby(["周期", "产品类型"])[col]
            .rank(method="min", ascending=True)  # 越小越好
            .astype("Int64")
        )

        cnt_vals = (
            df.loc[eligible]
            .groupby(["周期", "产品类型"])[col]
            .transform("count")
            .astype("Int64")
        )

        df[f"{col}排名"] = df[f"{col}排名"].astype(object)
        df.loc[eligible, f"{col}排名"] = (
                rank_vals.astype(str) + "/" + cnt_vals.astype(str)
        )

    col = "防守能力"
    if eligible.any():
        rank_vals = (
            df.loc[eligible]
            .groupby(["周期", "产品类型"])[col]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )

        cnt_vals = (
            df.loc[eligible]
            .groupby(["周期", "产品类型"])[col]
            .transform("count")
            .astype("Int64")
        )

        df[f"{col}排名"] = df[f"{col}排名"].astype(object)
        df.loc[eligible, f"{col}排名"] = (
                rank_vals.astype(str) + "/" + cnt_vals.astype(str)
        )

    return df


# --------------------------------------------------
# 9. 格式化输出
# --------------------------------------------------
def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    格式化输出指标
    """
    df_copy = df.copy()

    # 百分比格式
    pct_cols = ["最大回撤", "贝塔", "年化波动率", "下行风险", "防守能力"]
    for col in pct_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )

    # 回撤修复时间为整数
    if "回撤修复" in df_copy.columns:
        df_copy["回撤修复"] = df_copy["回撤修复"].apply(
            lambda x: f"{int(x)}天" if pd.notna(x) else ""
        )

    rank_cols = ["最大回撤排名", "贝塔排名", "回撤修复排名", "年化波动率排名", "下行风险排名", "防守能力排名"]
    for col in rank_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x}名" if pd.notna(x) and x != "" else ""
            )

    return df_copy



# --------------------------------------------------
# 10. 主流程
# --------------------------------------------------
def main(index_secu_id="000300.IDX.CSIDX"):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]

    # 获取数据
    prd_df = fetch_fin_prd_nav()
    idx_df = fetch_index_quote(index_secu_id)

    # 计算所有产品的风险指标
    all_results = []
    for _, g in prd_df.groupby("PRD_CODE"):
        results = calc_product_risk_metrics(g, idx_df, periods)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # 计算合格性（与收益能力指标保持一致）
    nav_days_df = calc_period_nav_days(prd_df, periods)
    max_days_df = (
        nav_days_df
        .groupby(["PRD_TYP", "周期"])["nav_days"]
        .max()
        .reset_index(name="max_nav_days")
    )
    nav_days_df = nav_days_df.merge(max_days_df, on=["PRD_TYP", "周期"])
    nav_days_df["eligible"] = (
            nav_days_df["nav_days"] == nav_days_df["max_nav_days"]
    )

    eligibility = nav_days_df[["PRD_CODE", "周期", "eligible"]].copy()

    # 添加同类平均和沪深 300
    df_rank = build_benchmark_and_avg(df, idx_df, periods)

    # 排名
    df_rank = rank_risk_df(df_rank, eligibility)

    # 格式化输出
    df_formatted = format_output(df_rank)

    return df_formatted


# --------------------------------------------------
# 11. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    result = main()
    result.to_csv("风险控制指标.csv", index=False, encoding="utf-8-sig")
    print(result.head())
