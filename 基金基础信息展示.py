import oracledb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def fetch_trading_days() -> pd.DataFrame:
    """获取交易日数据"""
    sql = """
          SELECT CALD_DATE, IS_TRD_DT
          FROM DATA_MART_04.TRD_CALD_DTL
          WHERE CALD_ID = '1'
          """
    conn = get_oracle_conn()
    try:
        df = pd.read_sql(sql, conn)
        # 转换CALD_DATE为日期类型
        df["CALD_DATE"] = pd.to_datetime(df["CALD_DATE"], format="%Y%m%d", errors="coerce")
        # 转换IS_TRD_DT为布尔类型
        df["IS_TRD_DT"] = df["IS_TRD_DT"].astype(int) == 1
        return df
    finally:
        conn.close()


def fetch_fin_prd_nav(limit: int = None) -> pd.DataFrame:
    sql = """
          SELECT PRD_TYP,
                 UNIT_NVAL,
                 AGGR_UNIT_NVAL,
                 NAV_DT,
                 PRD_CODE
          FROM DATA_MART_04.FIN_PRD_NAV
          """
    if limit is not None:
        sql += f" WHERE ROWNUM <= {limit}"

    conn = get_oracle_conn()
    try:
        df = pd.read_sql(sql, conn)
        return df
    finally:
        conn.close()

def fetch_pty_prd_base_info() -> pd.DataFrame:
    """获取产品基础信息表"""
    sql = """
          SELECT PRD_CODE,
                 FOUND_DT,
                 PRD_NAME,
                 PRD_FULL_NAME,
                 PRD_TYP
          FROM DATA_MART_04.PTY_PRD_BASE_INFO
          """
    conn = get_oracle_conn()
    try:
        df = pd.read_sql(sql, conn)
        df["FOUND_DT"] = pd.to_datetime(
            df["FOUND_DT"],
            format="%Y%m%d",
            errors="coerce"
        )
        return df
    finally:
        conn.close()


# -----------------------------# 3. 年化计算函数（单利 / 复利）# -----------------------------
def calc_since_annualized_return(
        start_value: float,
        end_value: float,
        days: int,
        annual_days: int,
        method: str
) -> float:
    if days <= 0 or start_value <= 0:
        return np.nan

    total_return = (end_value - start_value) / start_value

    if method == "simple":
        # 单利：线性年化
        return total_return * (annual_days / days)
    elif method == "compound":
        # 复利：指数年化
        return (end_value / start_value) ** (annual_days / days) - 1
    else:
        raise ValueError(f"不支持的年化方式：{method}")


# -----------------------------# 计算成立以来年化 # -----------------------------
def calc_established_annualized(
        virtual_start_dt: pd.Timestamp,
        current_dt: pd.Timestamp,
        end_aggr: float,
        start_aggr: float = 1.0,
        method: str = "compound",
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None
) -> tuple:
    # 初始化实际期末日
    actual_end_dt = current_dt

    # 计算持有天数和年化天数
    if day_type == "交易日":
        if trading_days_df is not None:
            # 过滤出虚拟起始点到计算基准日之间的交易日
            trading_days = trading_days_df[
                (trading_days_df["CALD_DATE"] > virtual_start_dt) &
                (trading_days_df["CALD_DATE"] <= current_dt) &
                (trading_days_df["IS_TRD_DT"] == True)
                ]

            # 计算持有天数
            days_since = len(trading_days)

            # 如果期末日期是非交易日，找到最近的交易日作为实际期末日
            if current_dt not in trading_days["CALD_DATE"].values:
                # 找到小于等于current_dt的最大交易日
                valid_trading_days = trading_days[trading_days["CALD_DATE"] <= current_dt]
                if len(valid_trading_days) > 0:
                    actual_end_dt = valid_trading_days["CALD_DATE"].max()
        else:
            # 如果没有交易日数据，回退到自然日计算
            days_since = (current_dt - virtual_start_dt).days
        # 交易日年化天数为250
        annual_days = 250
    else:
        # 自然日计算，完全忽略trading_days_df参数
        days_since = (current_dt - virtual_start_dt).days
        # 自然日年化天数为365
        annual_days = 365

    # 防止除以0
    if days_since <= 0:
        days_since = 1

    # 成立以来收益
    since_return = (end_aggr - start_aggr) / start_aggr

    # 成立以来年化
    annualized = calc_since_annualized_return(
        start_value=start_aggr,
        end_value=end_aggr,
        days=days_since,
        annual_days=annual_days,
        method=method
    )

    return since_return, annualized, actual_end_dt


def count_interval_days(
        df_prd: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
) -> int:
    return df_prd[
        (df_prd["NAV_DT"] >= start_dt) &
        (df_prd["NAV_DT"] <= end_dt) &
        (df_prd["AGGR_UNIT_NVAL"].notna())
        ]["NAV_DT"].nunique()


def calculate_holiday_days(
        current_date: pd.Timestamp,
        trading_days_df: pd.DataFrame
) -> int:

    # 确保current_date是Timestamp类型
    if not isinstance(current_date, pd.Timestamp):
        current_date = pd.Timestamp(current_date)

    # 过滤出所有交易日
    trading_dates = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()

    # 找到大于current_date的第一个交易日
    next_trading_date = trading_dates[trading_dates > current_date]

    if len(next_trading_date) == 0:
        # 如果没有找到下一个交易日，返回1（只包括当前日期）
        return 1

    next_trading_date = next_trading_date.iloc[0]

    # 计算天数差（从当前日期到下一个交易日的天数）
    days_diff = (next_trading_date - current_date).days
    return days_diff


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


# ==================================================
# 5.今年以来收益（自然年 + 首个交易日）
# ==================================================

def calc_ytd_return(
        grp: pd.DataFrame,
        base_dt: pd.Timestamp
) -> float:
    year = base_dt.year
    year_start = pd.Timestamp(f"{year}-01-01")
    prev_year_end = pd.Timestamp(f"{year - 1}-12-31")

    last_day_prev_year = grp[
        (grp["NAV_DT"] <= prev_year_end)
    ]["NAV_DT"].max()

    if pd.isna(last_day_prev_year):
        first_day = grp[
            (grp["NAV_DT"] >= year_start) &
            (grp["NAV_DT"] <= base_dt)
            ]["NAV_DT"].min()

        if pd.isna(first_day):
            return np.nan

        start_val = grp.loc[
            grp["NAV_DT"] == first_day,
            "AGGR_UNIT_NVAL"
        ].values[0]
    else:
        start_val = grp.loc[
            grp["NAV_DT"] == last_day_prev_year,
            "AGGR_UNIT_NVAL"
        ].values[0]

    end_val = grp.loc[
        grp["NAV_DT"] == base_dt,
        "AGGR_UNIT_NVAL"
    ].values[0]

    return (end_val - start_val) / start_val


# ==================================================
# 6. 计算每个 PRD_CODE 的指标
# ==================================================

def calculate_all_fund_metrics(
        df: pd.DataFrame,
        df_base_info: pd.DataFrame,
        annual_method: str = "simple",
        day_type: str = "自然日"
) -> pd.DataFrame:
    df = df.copy()
    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"], format="%Y%m%d", errors="coerce")

    # 获取交易日数据（仅当day_type为交易日时）
    trading_days_df = None
    if day_type == "交易日":
        trading_days_df = fetch_trading_days()
    # 当day_type为自然日时，确保trading_days_df为None，完全按自然日计算

    for prd_code in df["PRD_CODE"].unique():
        prd_data = df[df["PRD_CODE"] == prd_code]
        unique_types = prd_data["PRD_TYP"].dropna().unique()

        if len(unique_types) == 1:
            df.loc[(df["PRD_CODE"] == prd_code) & (df["PRD_TYP"].isna()), "PRD_TYP"] = unique_types[0]

    results = []

    # ========== 近一年收益排名 ==========
    yly_rank_dict = {}

    # 先计算每个产品的近一年收益和相关信息
    product_data = []
    for prd_code, grp in df.groupby("PRD_CODE"):
        grp = grp.sort_values("NAV_DT")
        base_dt = grp["NAV_DT"].max()

        # 近一年的理论开始日
        start_dt = base_dt.replace(year=base_dt.year - 1) - timedelta(days=1)

        # 从基础信息表获取产品实际成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            # 如果基础信息表没有成立日，跳过该产品
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 判断是否符合条件：成立日 <= 理论开始日
        if fund_established_dt > start_dt:
            continue

        # 计算近一年收益
        sub = grp[
            (grp["NAV_DT"] >= start_dt) &
            (grp["NAV_DT"] <= base_dt)
            ]

        if len(sub) < 2:
            # 数据不足，无法计算收益
            ret = np.nan
        else:
            # 确保 AGGR_UNIT_NVAL 是数值类型
            end_val = float(sub.iloc[-1]["AGGR_UNIT_NVAL"])
            start_val = float(sub.iloc[0]["AGGR_UNIT_NVAL"])
            ret = (end_val - start_val) / start_val

        # 获取产品类型
        prd_typ = grp["PRD_TYP"].dropna().unique()
        if len(prd_typ) == 0:
            continue
        prd_typ = prd_typ[0]

        product_data.append({
            "PRD_TYP": prd_typ,
            "PRD_CODE": prd_code,
            "base_dt": base_dt,
            "RET": ret
        })

    # 按产品类型和计算基准日分组计算排名
    if product_data:
        rank_df = pd.DataFrame(product_data)

        # 按产品类型和计算基准日分组
        for (prd_typ, base_dt), group in rank_df.groupby(["PRD_TYP", "base_dt"]):
            if len(group) == 0:
                continue

            # 计算排名
            group["RANK"] = group["RET"].rank(method="min", ascending=False).astype("Int64")

            # 记录排名
            total_count = len(group)
            for _, row in group.iterrows():
                rank_num = row["RANK"]
                if not pd.isna(rank_num):
                    yly_rank_dict[(prd_typ, row["PRD_CODE"])] = (rank_num, total_count)

    # ---------- 遍历每个产品 ----------
    for prd_code, group in df.groupby("PRD_CODE"):
        # 从基础信息表获取成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            # 如果基础信息表没有成立日，跳过该产品
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 数据清洗：过滤掉成立日之前的脏数据
        group_clean = group[group["NAV_DT"] >= fund_established_dt].copy()

        if group_clean.empty:
            # 如果清洗后没有数据，跳过该产品
            continue

        group_clean = group_clean.sort_values("NAV_DT").reset_index(drop=True)
        if group_clean.empty:
            continue

        latest = group_clean.iloc[-1]
        current_dt = group_clean["NAV_DT"].max()
        prd_typ = group_clean.iloc[0]["PRD_TYP"]

        # ---- 今年来收益 ----
        ytd_return = calc_ytd_return(group_clean, current_dt)

        # ---- 成立以来收益 ----
        # 统一使用成立日的前一天作为虚拟起始点，期初净值设为1.0
        establish_date = fund_established_dt

        # 虚拟起始点：成立日的前一天
        virtual_start_dt = establish_date - pd.Timedelta(days=1)

        # 期初净值统一设为1.0
        start_aggr = 1.0

        # 计算成立以来收益和年化收益率
        since_return, annualized, actual_end_dt = calc_established_annualized(
            virtual_start_dt=virtual_start_dt,
            current_dt=current_dt,
            end_aggr=latest["AGGR_UNIT_NVAL"],
            start_aggr=start_aggr,
            method=annual_method,
            day_type=day_type,
            trading_days_df=trading_days_df
        )

        # 如果实际期末日与current_dt不同，需要获取对应的期末净值
        if actual_end_dt != current_dt:
            actual_end_data = group_clean[group_clean["NAV_DT"] == actual_end_dt]
            if len(actual_end_data) > 0:
                end_aggr = actual_end_data.iloc[0]["AGGR_UNIT_NVAL"]
                # 重新计算收益和年化，使用正确的期末净值
                since_return, annualized, _ = calc_established_annualized(
                    virtual_start_dt=virtual_start_dt,
                    current_dt=actual_end_dt,
                    end_aggr=end_aggr,
                    start_aggr=start_aggr,
                    method=annual_method,
                    day_type=day_type,
                    trading_days_df=trading_days_df
                )

        # ---- 最大回撤 ----
        group_clean["cum_max"] = group_clean["UNIT_NVAL"].cummax()
        group_clean["drawdown"] = (group_clean["UNIT_NVAL"] - group_clean["cum_max"]) / group_clean["cum_max"]
        max_drawdown = group_clean["drawdown"].min()

        # ---- 夏普比率 ----
        sharpe_annual_factor = 250 if day_type == "交易日" else 365

        # ========== 年化波动率计算：第一天使用成立日净值/虚拟起始日(1.0) ==========
        # 构建包含虚拟起始点的完整序列
        virtual_start_nav = pd.DataFrame({
            "NAV_DT": [virtual_start_dt],
            "AGGR_UNIT_NVAL": [1.0]
        })

        # 合并虚拟起始点和实际净值数据
        full_nav_series = pd.concat([virtual_start_nav, group_clean[["NAV_DT", "AGGR_UNIT_NVAL"]]], ignore_index=True)
        full_nav_series = full_nav_series.sort_values("NAV_DT").reset_index(drop=True)

        # ========== 关键：交易日模式需要剔除非交易日 ==========
        if day_type == "交易日" and trading_days_df is not None:
            # 获取所有交易日
            trading_dates = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)

            # 过滤出交易日的数据（包含虚拟起始日）
            full_nav_series = full_nav_series[
                full_nav_series["NAV_DT"].isin(trading_dates) |
                (full_nav_series["NAV_DT"] == virtual_start_dt)
                ].reset_index(drop=True)

        # 计算每日收益率（第一天 = 成立日净值 / 1.0 - 1）
        daily_ret = full_nav_series["AGGR_UNIT_NVAL"].pct_change().dropna()

        if len(daily_ret) > 0:
            # 日收益率标准差
            daily_ret_std = daily_ret.std()

            # 年化无风险利率
            ann_rf = 0.015

            if day_type == "自然日":
                # 自然日模式：日无风险利率 = 1.5%/365
                daily_rf = ann_rf / 365

                # 先计算每日超额收益率序列，再求平均
                daily_excess_ret_series = daily_ret - daily_rf
                daily_excess_ret_mean = daily_excess_ret_series.mean()

            else:  # 交易日模式
                # daily_ret的索引对应的是full_nav_series中相邻两个日期的起始日期
                daily_rf_list = []

                # 遍历daily_ret的索引，获取对应的起始日期
                for i, current_idx in enumerate(daily_ret.index):
                    # 从full_nav_series中获取当前日期
                    current_date = full_nav_series.loc[current_idx, "NAV_DT"]

                    # 计算从当前日期到下一个交易日的天数（包括当前日期）
                    holiday_days = calculate_holiday_days(current_date, trading_days_df)
                    # 日无风险利率 = 1.5%/365 * 节假日天数
                    daily_rf = (ann_rf / 365) * holiday_days
                    daily_rf_list.append(daily_rf)

                # 将每日无风险利率转换为Series，与daily_ret对齐
                daily_rf_series = pd.Series(daily_rf_list, index=daily_ret.index)

                # 先计算每日超额收益率
                daily_excess_ret_series = daily_ret - daily_rf_series

                # 再求平均
                daily_excess_ret_mean = daily_excess_ret_series.mean()

            # 年化夏普比率 = (平均日超额收益率 / 日收益率标准差) × √年化系数
            if daily_ret_std != 0:
                sharpe = (daily_excess_ret_mean / daily_ret_std) * np.sqrt(sharpe_annual_factor)
            else:
                sharpe = np.nan
        else:
            sharpe = np.nan

        # ---- 近一年收益 ----
        base_dt = current_dt
        start_dt = base_dt.replace(year=base_dt.year - 1)
        start_dt = start_dt - timedelta(days=1)
        yly_df = group_clean[
            (group_clean["NAV_DT"] >= start_dt) &
            (group_clean["NAV_DT"] <= current_dt)
            ]

        yly_return = np.nan
        if len(yly_df) >= 2:
            yly_return = (
                                 yly_df.iloc[-1]["AGGR_UNIT_NVAL"] -
                                 yly_df.iloc[0]["AGGR_UNIT_NVAL"]
                         ) / yly_df.iloc[0]["AGGR_UNIT_NVAL"]

        # ---- 近一年收益排名 ----
        rank_key = (prd_typ, prd_code)
        rank_info = yly_rank_dict.get(rank_key)

        if rank_info:
            rank_num, total_in_typ = rank_info
            yly_rank = f"{rank_num}/{total_in_typ}名"
        else:
            yly_rank = "N/A"

        results.append({
            "PRD_CODE": prd_code,
            "计算基准日": current_dt.strftime("%Y-%m-%d"),
            "单位净值": round(latest["UNIT_NVAL"], 4),
            "累计净值": round(latest["AGGR_UNIT_NVAL"], 4),
            "今年来收益": f"{ytd_return:.2%}" if not np.isnan(ytd_return) else "N/A",
            "成立以来收益": f"{since_return:.2%}" if not np.isnan(since_return) else "N/A",
            "成立以来年化": f"{annualized:.2%}" if not np.isnan(annualized) else "N/A",
            "成立以来最大回撤": f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A",
            "成立以来夏普比率": round(sharpe, 2) if not np.isnan(sharpe) else "N/A",
            "近一年收益": f"{yly_return:.2%}" if not np.isnan(yly_return) else "N/A",
            "近一年收益排名": yly_rank
        })

    return pd.DataFrame(results)


# -----------------------------
# 7. 示例调用
# -----------------------------
if __name__ == "__main__":
    df = fetch_fin_prd_nav()
    df_base_info = fetch_pty_prd_base_info()
    df = fill_special_prd_typ(df, df_base_info)

    print("\n各指标计算的结果：")
    metrics_df = calculate_all_fund_metrics(
        df,
        df_base_info,
        annual_method="compound",
        day_type="交易日"
    )

    metrics_df = metrics_df.sort_values(["PRD_CODE"]).reset_index(drop=True)
    product_count = len(metrics_df)

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("📊 所有产品指标计算结果")
    output_lines.append(f"✅ 共输出产品数量：{product_count}")
    output_lines.append("=" * 80)
    output_lines.append("")

    for _, row in metrics_df.iterrows():
        output_lines.append("-" * 80)
        for col in metrics_df.columns:
            output_lines.append(f"{col:20s}: {row[col]}")

    output_lines.append("=" * 80)
    output_content = "\n".join(output_lines)
    print(output_content)
    with open("基金基础信息展示结果.txt", "w", encoding="utf-8") as f:
        f.write(output_content)
    print("\n💾 结果已保存至：基金基础信息展示结果.txt")