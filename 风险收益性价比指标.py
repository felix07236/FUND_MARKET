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
def get_period_start(end_dt: pd.Timestamp, period: str, fund_est_dt: pd.Timestamp = None) -> pd.Timestamp:
    if period == "今年以来":
        result = pd.Timestamp(year=end_dt.year, month=1, day=1)
        # 如果基金成立日期晚于 1 月 1 日，使用成立日期
        if fund_est_dt and result < fund_est_dt:
            return fund_est_dt
        return result

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
# 5. 风险收益性价比指标计算
# --------------------------------------------------
def calc_risk_adjusted_returns(prd_df: pd.DataFrame, idx_df: pd.DataFrame, period: str,
                               risk_free_rate: float = 0.015) -> dict:
    """
    计算单个产品在指定周期内的风险收益性价比指标

    Parameters:
    -----------
    prd_df : DataFrame
        单只产品的净值数据
    idx_df : DataFrame
        指数数据
    period : str
        周期名称
    risk_free_rate : float
        无风险利率（年化），默认 3%

    Returns:
    --------
    dict : 包含所有风险收益性价比指标的字典
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
            "夏普比率": np.nan,
            "索提诺比率": np.nan,
            "卡玛比率": np.nan
        }

    # 计算日收益率
    sub = sub.copy()
    sub["日收益"] = sub["AGGR_UNIT_NVAL"].pct_change()

    # 准备指数数据
    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= start_dt) &
        (idx_df["NAV_DT"] <= base_dt)
        ].copy()

    if len(idx_sub) < 2:
        idx_merged = None
    else:
        idx_sub = idx_sub.sort_values("NAV_DT")
        idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()
        idx_merged = sub.merge(idx_sub[["NAV_DT", "指数日收益"]], on="NAV_DT", how="inner")

    # ========== 1. 夏普比率 (Sharpe Ratio) ==========
    # 公式：(组合收益率 - 无风险利率) / 组合标准差
    # 年化夏普比率 = 日夏普比率 × √252

    # 计算总收益率
    start_nav = sub["AGGR_UNIT_NVAL"].iloc[0]
    end_nav = sub["AGGR_UNIT_NVAL"].iloc[-1]
    total_return = end_nav / start_nav - 1

    # 计算年化收益率
    days = (sub["NAV_DT"].iloc[-1] - sub["NAV_DT"].iloc[0]).days
    ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1

    # 计算日收益率的标准差
    daily_std = sub["日收益"].std()

    if pd.notna(daily_std) and daily_std > 0:
        # 年化标准差
        ann_std = daily_std * np.sqrt(252)
        # 日无风险利率
        daily_rf = risk_free_rate / 252
        # 夏普比率
        sharpe_ratio = (ann_return - risk_free_rate) / ann_std
    else:
        sharpe_ratio = np.nan

    # ========== 2. 索提诺比率 (Sortino Ratio) ==========
    # 公式：(组合收益率 - 无风险利率) / 下行标准差
    # 只考虑负收益的波动

    negative_returns = sub[sub["日收益"] < 0]["日收益"]

    if len(negative_returns) >= 2:
        # 下行标准差（半标准差）
        downside_std = negative_returns.std()
        # 年化下行标准差
        ann_downside_std = downside_std * np.sqrt(252)

        if pd.notna(ann_downside_std) and ann_downside_std > 0:
            sortino_ratio = (ann_return - risk_free_rate) / ann_downside_std
        else:
            sortino_ratio = np.nan
    else:
        sortino_ratio = np.nan

    # ========== 3. 卡玛比率 (Calmar Ratio) ==========
    # 公式：年化收益率 / 最大回撤
    # 最大回撤取绝对值

    # 计算最大回撤
    sub["累计净值"] = sub["AGGR_UNIT_NVAL"]
    sub["历史最高"] = sub["累计净值"].cummax()
    sub["回撤"] = (sub["累计净值"] - sub["历史最高"]) / sub["历史最高"]
    max_drawdown = abs(sub["回撤"].min())

    if pd.notna(max_drawdown) and max_drawdown > 0:
        calmar_ratio = ann_return / max_drawdown
    else:
        calmar_ratio = np.nan

    return {
        "周期": period,
        "夏普比率": sharpe_ratio,
        "索提诺比率": sortino_ratio,
        "卡玛比率": calmar_ratio
    }


# --------------------------------------------------
# 6. 单产品多周期指标计算
# --------------------------------------------------
def calc_product_risk_adjusted_returns(prd_df: pd.DataFrame, idx_df: pd.DataFrame, periods: list,
                                       risk_free_rate: float = 0.015) -> list:

    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    est_dt = prd_df["NAV_DT"].min()

    prd_typ = prd_df["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    results = []

    for period in periods:
        metrics = calc_risk_adjusted_returns(prd_df, idx_df, period, risk_free_rate)

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

        ratio_cols = ["夏普比率", "索提诺比率", "卡玛比率"]

        row = {
            "产品代码": "同类平均",
            "产品类型": typ,
            "周期": period,
        }

        for col in ratio_cols:
            row[col] = g[col].mean()

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

        # 计算指数的风险收益性价比指标
        idx_sub = idx_sub.copy()
        idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()

        # 总收益率
        idx_total_return = idx_sub["INDEX_CLOSE"].iloc[-1] / idx_sub["INDEX_CLOSE"].iloc[0] - 1
        days = (idx_sub["NAV_DT"].iloc[-1] - idx_sub["NAV_DT"].iloc[0]).days
        idx_ann_return = (1 + idx_total_return) ** (365 / max(days, 1)) - 1

        # 夏普比率（假设无风险利率为 3%）
        risk_free_rate = 0.03
        idx_std = idx_sub["指数日收益"].std() * np.sqrt(252)
        idx_sharpe = (idx_ann_return - risk_free_rate) / idx_std if pd.notna(idx_std) and idx_std > 0 else np.nan

        # 索提诺比率
        idx_negative = idx_sub[idx_sub["指数日收益"] < 0]["指数日收益"]
        if len(idx_negative) >= 2:
            idx_downside_std = idx_negative.std() * np.sqrt(252)
            idx_sortino = (idx_ann_return - risk_free_rate) / idx_downside_std if pd.notna(
                idx_downside_std) and idx_downside_std > 0 else np.nan
        else:
            idx_sortino = np.nan

        # 卡玛比率
        idx_sub["累计净值"] = idx_sub["INDEX_CLOSE"]
        idx_sub["历史最高"] = idx_sub["累计净值"].cummax()
        idx_sub["回撤"] = (idx_sub["累计净值"] - idx_sub["历史最高"]) / idx_sub["历史最高"]
        idx_max_dd = abs(idx_sub["回撤"].min())
        idx_calmar = idx_ann_return / idx_max_dd if pd.notna(idx_max_dd) and idx_max_dd > 0 else np.nan

        rows.append({
            "产品代码": "沪深 300",
            "产品类型": "指数",
            "周期": period,
            "夏普比率": idx_sharpe,
            "索提诺比率": idx_sortino,
            "卡玛比率": idx_calmar
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------
# 8. 排名
# --------------------------------------------------
def rank_risk_adjusted_df(df: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:

    # 风险收益性价比指标：越大越好
    larger_is_better = ["夏普比率", "索提诺比率", "卡玛比率"]

    # 初始化排名列为空
    for col in larger_is_better:
        df[f"{col}排名"] = np.nan

    # 创建合格性索引
    idx = df.set_index(["产品代码", "周期"]).index
    elig_idx = eligibility.set_index(["PRD_CODE", "周期"]).index
    eligible = idx.isin(elig_idx)

    # 对每个指标进行排名
    for col in larger_is_better:
        if eligible.any():
            rank_vals = (
                df.loc[eligible]
                .groupby(["周期", "产品类型"])[col]
                .rank(method="min", ascending=False)  # 越大越好
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

    # 风险收益性价比指标保留两位小数
    ratio_cols = ["夏普比率", "索提诺比率", "卡玛比率"]
    for col in ratio_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )

    rank_cols = ["夏普比率排名", "索提诺比率排名", "卡玛比率排名"]
    for col in rank_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x}名" if pd.notna(x) and x != "" else ""
            )

    return df_copy


# --------------------------------------------------
# 10. 主流程
# --------------------------------------------------
def main(index_secu_id="000300.IDX.CSIDX", risk_free_rate: float = 0.015):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]

    # 获取数据
    prd_df = fetch_fin_prd_nav()
    idx_df = fetch_index_quote(index_secu_id)

    # 计算所有产品的风险收益性价比指标
    all_results = []
    for _, g in prd_df.groupby("PRD_CODE"):
        results = calc_product_risk_adjusted_returns(g, idx_df, periods, risk_free_rate)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # 计算合格性（与风险控制指标保持一致）
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
    df_rank = rank_risk_adjusted_df(df_rank, eligibility)

    # 格式化输出
    df_formatted = format_output(df_rank)

    return df_formatted


# --------------------------------------------------
# 11. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    result = main()
    result.to_csv("风险收益性价比指标.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(result.head())
