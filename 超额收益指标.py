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
          SELECT PRD_CODE,
                 PRD_TYP,
                 NAV_DT,
                 AGGR_UNIT_NVAL,
                 NAV_ADD_RAT,
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
def fill_special_prd_typ(df_nav: pd.DataFrame) -> pd.DataFrame:
    """填充产品类型为空的记录"""
    df = df_nav.copy()
    
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


def get_period_start(end_dt: pd.Timestamp, period: str, fund_est_dt: pd.Timestamp = None) -> pd.Timestamp:
    """获取周期的起始日期"""
    if period == "今年以来":
        result = pd.Timestamp(year=end_dt.year, month=1, day=1)
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
        
        try:
            result = pd.Timestamp(
                year=target_year,
                month=target_month,
                day=target_day
            )
        except ValueError:
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
# 5. 超额收益指标计算
# --------------------------------------------------
def calc_excess_metrics(prd_df: pd.DataFrame, idx_df: pd.DataFrame, period: str,
                        risk_free_rate: float = 0.015) -> dict:
    """
    计算单个产品在指定周期内的超额收益相关指标

    Parameters:
    -----------
    prd_df : DataFrame
        单只产品的净值数据
    idx_df : DataFrame
        指数数据
    period : str
        周期名称
    risk_free_rate : float
        无风险利率（年化），默认 1.5%

    Returns:
    --------
    dict : 包含所有超额收益指标的字典
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
            "超额收益": np.nan,
            "超额年化": np.nan,
            "超额回撤": np.nan,
            "超额回撤修复天数": np.nan,
            "超额夏普比率": np.nan,
            "超额年化波动率": np.nan,
            "超额索提诺比率": np.nan,
            "超额卡玛比率": np.nan
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
        # 合并时保留 INDEX_CLOSE 列用于计算总收益
        idx_merged = sub.merge(idx_sub[["NAV_DT", "INDEX_CLOSE", "指数日收益"]], on="NAV_DT", how="inner")

    # ========== 基础计算 ==========
    # 基金总收益率
    start_nav = sub["AGGR_UNIT_NVAL"].iloc[0]
    end_nav = sub["AGGR_UNIT_NVAL"].iloc[-1]
    fund_total_return = end_nav / start_nav - 1

    # 指数总收益率
    if idx_merged is not None and len(idx_merged) >= 2:
        idx_start = idx_merged["INDEX_CLOSE"].iloc[0]
        idx_end = idx_merged["INDEX_CLOSE"].iloc[-1]
        idx_total_return = idx_end / idx_start - 1
    else:
        idx_total_return = 0

    # 超额收益 = 基金收益 - 指数收益
    excess_return = fund_total_return - idx_total_return

    # 年化收益率
    days = (sub["NAV_DT"].iloc[-1] - sub["NAV_DT"].iloc[0]).days
    fund_ann_return = (1 + fund_total_return) ** (365 / max(days, 1)) - 1
    idx_ann_return = (1 + idx_total_return) ** (365 / max(days, 1)) - 1 if idx_total_return != 0 else 0

    # 超额年化 = 基金年化 - 指数年化
    excess_ann_return = fund_ann_return - idx_ann_return

    # ========== 1. 超额夏普比率 ==========
    # 使用超额收益的均值和标准差
    if idx_merged is not None and len(idx_merged) >= 2:
        # 计算每日超额收益
        idx_merged["超额日收益"] = idx_merged["日收益"] - idx_merged["指数日收益"]

        # 超额收益的年化均值
        excess_daily_mean = idx_merged["超额日收益"].mean()
        excess_ann_mean = excess_daily_mean * 252

        # 超额收益的标准差
        excess_std = idx_merged["超额日收益"].std()
        excess_ann_std = excess_std * np.sqrt(252)

        if pd.notna(excess_ann_std) and excess_ann_std > 0:
            excess_sharpe = (excess_ann_mean - risk_free_rate) / excess_ann_std
        else:
            excess_sharpe = np.nan

        # ========== 2. 超额年化波动率 ==========
        excess_volatility = excess_ann_std

        # ========== 3. 超额索提诺比率 ==========
        # 只考虑负的超额收益
        negative_excess = idx_merged[idx_merged["超额日收益"] < 0]["超额日收益"]

        if len(negative_excess) >= 2:
            downside_excess_std = negative_excess.std()
            ann_downside_excess_std = downside_excess_std * np.sqrt(252)

            if pd.notna(ann_downside_excess_std) and ann_downside_excess_std > 0:
                excess_sortino = (excess_ann_mean - risk_free_rate) / ann_downside_excess_std
            else:
                excess_sortino = np.nan
        else:
            excess_sortino = np.nan

        # ========== 4. 超额卡玛比率 ==========
        # 计算累计超额收益曲线
        idx_merged["累计超额"] = (1 + idx_merged["超额日收益"]).cumprod()
        idx_merged["累计超额峰值"] = idx_merged["累计超额"].cummax()
        idx_merged["超额回撤"] = (idx_merged["累计超额"] - idx_merged["累计超额峰值"]) / idx_merged["累计超额峰值"]

        max_excess_dd = abs(idx_merged["超额回撤"].min())

        if pd.notna(max_excess_dd) and max_excess_dd > 0:
            excess_calmar = excess_ann_return / max_excess_dd
        else:
            excess_calmar = np.nan

        # ========== 5. 超额回撤 ==========
        excess_drawdown = max_excess_dd

        # ========== 6. 超额回撤修复天数 ==========
        # 找到最大回撤发生的位置
        if pd.notna(max_excess_dd) and max_excess_dd > 0:
            max_dd_idx = idx_merged["超额回撤"].idxmin()
            peak_before = idx_merged.loc[:max_dd_idx, "累计超额"].idxmax()

            # 查找修复时间（回到峰值的时间）
            recovery_found = False
            for future_idx in range(max_dd_idx + 1, len(idx_merged)):
                if idx_merged.iloc[future_idx]["累计超额"] >= idx_merged.iloc[peak_before]["累计超额"]:
                    recovery_days = (idx_merged.iloc[future_idx]["NAV_DT"] - idx_merged.iloc[max_dd_idx]["NAV_DT"]).days
                    excess_recovery_days = recovery_days
                    recovery_found = True
                    break

            if not recovery_found:
                # 如果到期末还未修复，计算到期末的天数
                excess_recovery_days = (idx_merged.iloc[-1]["NAV_DT"] - idx_merged.iloc[max_dd_idx]["NAV_DT"]).days
        else:
            excess_recovery_days = 0

    else:
        # 没有指数数据时，所有超额指标都为 NaN
        excess_sharpe = np.nan
        excess_volatility = np.nan
        excess_sortino = np.nan
        excess_calmar = np.nan
        excess_drawdown = np.nan
        excess_recovery_days = np.nan

    return {
        "周期": period,
        "超额收益": excess_return,
        "超额年化": excess_ann_return,
        "超额回撤": excess_drawdown,
        "超额回撤修复天数": excess_recovery_days,
        "超额夏普比率": excess_sharpe,
        "超额年化波动率": excess_volatility,
        "超额索提诺比率": excess_sortino,
        "超额卡玛比率": excess_calmar
    }


# --------------------------------------------------
# 6. 单产品多周期指标计算
# --------------------------------------------------
def calc_product_excess_metrics(prd_df: pd.DataFrame, idx_df: pd.DataFrame, periods: list,
                                risk_free_rate: float = 0.015) -> list:
    """计算单个产品在多个周期的超额收益指标"""

    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    est_dt = prd_df["NAV_DT"].min()

    prd_typ = prd_df["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    results = []

    for period in periods:
        metrics = calc_excess_metrics(prd_df, idx_df, period, risk_free_rate)

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
    """构建同类平均和沪深 300 基准数据"""
    rows = []

    # 计算同类平均
    for (typ, period), g in df.groupby(["产品类型", "周期"]):
        if typ == "指数":
            continue

        metric_cols = [
            "超额收益", "超额年化", "超额回撤", "超额回撤修复天数",
            "超额夏普比率", "超额年化波动率", "超额索提诺比率", "超额卡玛比率"
        ]

        row = {
            "产品代码": "同类平均",
            "产品类型": typ,
            "周期": period,
        }

        for col in metric_cols:
            row[col] = g[col].mean()

        rows.append(row)

    # 计算沪深 300 基准（指数自身的超额收益为 0）
    for period in periods:
        rows.append({
            "产品代码": "沪深 300",
            "产品类型": "指数",
            "周期": period,
            "超额收益": 0,
            "超额年化": 0,
            "超额回撤": 0,
            "超额回撤修复天数": 0,
            "超额夏普比率": 0,
            "超额年化波动率": 0,
            "超额索提诺比率": 0,
            "超额卡玛比率": 0
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------
# 8. 排名
# --------------------------------------------------
def rank_excess_df(df: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    """对超额收益指标进行排名"""
    
    # 越大越好的指标
    larger_is_better = [
        "超额收益", "超额年化", "超额夏普比率", "超额索提诺比率", "超额卡玛比率"
    ]
    
    # 越小越好的指标
    smaller_is_better = [
        "超额回撤", "超额回撤修复天数", "超额年化波动率"
    ]
    
    # 初始化排名列为空字符串
    all_metrics = larger_is_better + smaller_is_better
    for col in all_metrics:
        df[f"{col}排名"] = ""
    
    # 创建合格性索引
    idx = df.set_index(["产品代码", "周期"]).index
    elig_idx = eligibility.set_index(["PRD_CODE", "周期"]).index
    eligible = idx.isin(elig_idx)
    
    # 对每个指标进行排名
    for col in all_metrics:
        if eligible.any():
            ascending = col in smaller_is_better
            rank_vals = (
                df.loc[eligible]
                .groupby(["周期", "产品类型"])[col]
                .rank(method="min", ascending=ascending)
                .astype("Int64")
            )
            
            cnt_vals = (
                df.loc[eligible]
                .groupby(["周期", "产品类型"])[col]
                .transform("count")
                .astype("Int64")
            )
            
            df.loc[eligible, f"{col}排名"] = (
                    rank_vals.astype(str) + "/" + cnt_vals.astype(str) + "名"
            )
    
    return df


# --------------------------------------------------
# 9. 主流程
# --------------------------------------------------
if __name__ == "__main__":
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # 定义周期列表
    PERIODS = [
        "成立以来",
        "今年以来",
        "近1年",
        "近2年",
        "近3年",
        "近5年"
    ]
    # 获取基金净值数据
    df_nav = fetch_fin_prd_nav()

    # 获取沪深 300 指数数据
    df_idx = fetch_index_quote("000300.IDX.CSIDX")
    
    # 填充产品类型（直接使用上面定义的函数）
    df_nav = fill_special_prd_typ(df_nav)
    
    # 计算每个产品的指标
    all_results = []

    for prd_code, g in df_nav.groupby("PRD_CODE"):
        results = calc_product_excess_metrics(g, df_idx, PERIODS)
        all_results.extend(results)

    df_metrics = pd.DataFrame(all_results)

    eligibility = calc_period_nav_days(df_nav, PERIODS)

    df_with_bench = build_benchmark_and_avg(df_metrics, df_idx, PERIODS)

    df_ranked = rank_excess_df(df_with_bench, eligibility)

    return_cols = [
        "产品代码", "产品类型", "周期",
        "超额收益", "超额收益排名",
        "超额年化", "超额年化排名",
        "超额回撤", "超额回撤排名",
        "超额回撤修复天数", "超额回撤修复天数排名",
        "超额夏普比率", "超额夏普比率排名",
        "超额年化波动率", "超额年化波动率排名",
        "超额索提诺比率", "超额索提诺比率排名",
        "超额卡玛比率", "超额卡玛比率排名"
    ]

    df_final = df_ranked[return_cols]
    
    # 百分比格式化 - 先转换列为 object 类型
    pct_cols = ["超额收益", "超额年化"]
    for col in pct_cols:
        # 先将整列转换为 object 类型
        df_final[col] = df_final[col].astype(object)
        mask = df_final[col].notna()
        # 计算格式化后的值
        numeric_values = pd.to_numeric(df_ranked.loc[mask, col], errors='coerce')
        formatted = (numeric_values * 100).round(2).astype(str) + "%"
        df_final.loc[mask, col] = formatted
    
    # 数值格式化 - 先转换列为 object 类型
    num_cols = ["超额回撤", "超额回撤修复天数", "超额夏普比率", "超额年化波动率",
                "超额索提诺比率", "超额卡玛比率"]
    for col in num_cols:
        # 先将整列转换为 object 类型
        df_final[col] = df_final[col].astype(object)
        mask = df_final[col].notna()
        if mask.any():
            numeric_values = pd.to_numeric(df_ranked.loc[mask, col], errors='coerce')
            formatted = numeric_values.round(4).astype(str)
            df_final.loc[mask, col] = formatted
    
    # 导出 CSV
    output_file = "超额收益指标分析.csv"
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至：{output_file}")
    
    # 显示示例
    print("\n示例数据：")
    print(df_final.head(10))

