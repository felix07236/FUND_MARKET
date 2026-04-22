import oracledb
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os


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
def fetch_trading_days() -> pd.DataFrame:
    """获取交易日数据"""
    sql = """
          SELECT CALD_DATE, IS_TRD_DT
          FROM DATA_MART_04.TRD_CALD_DTL
          WHERE CALD_ID = '1' \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, con=conn)

    df["CALD_DATE"] = pd.to_datetime(df["CALD_DATE"], format="%Y%m%d", errors="coerce")
    df["IS_TRD_DT"] = df["IS_TRD_DT"].astype(int) == 1
    return df


def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE, \
                 PRD_TYP, \
                 NAV_DT, \
                 AGGR_UNIT_NVAL,
                 UNIT_NVAL
          FROM DATA_MART_04.FIN_PRD_NAV \
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


def fetch_product_base_info() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE,
                 FOUND_DT,
                 PRD_NAME,
                 PRD_FULL_NAME,
                 PRD_TYP
          FROM DATA_MART_04.PTY_PRD_BASE_INFO \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["FOUND_DT"] = pd.to_datetime(
        df["FOUND_DT"],
        format="%Y%m%d",
        errors="coerce"
    )
    return df


# --------------------------------------------------
# 3. 产品类型填充
# --------------------------------------------------
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


# --------------------------------------------------
# 4. 工具函数
# --------------------------------------------------

def calc_since_annualized_return(
        start_value: float,
        end_value: float,
        days: int,
        annual_days: int
) -> float:
    if days <= 0 or start_value <= 0:
        return np.nan

    # 复利年化
    return (end_value / start_value) ** (annual_days / days) - 1


def calc_return_metric(start_value: float, end_value: float) -> float:
    if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
        return np.nan
    return (end_value - start_value) / start_value


def calc_annualized_return_metric(start_value: float, end_value: float, days: int, annual_days: int) -> float:
    return calc_since_annualized_return(
        start_value=start_value,
        end_value=end_value,
        days=days,
        annual_days=annual_days
    )


def calc_alpha_metric(fund_excess_ret: pd.Series, idx_excess_ret: pd.Series, beta: float) -> float:
    if len(fund_excess_ret) == 0 or len(idx_excess_ret) == 0:
        return np.nan
    return fund_excess_ret.mean() - beta * idx_excess_ret.mean()


def calc_monthly_win_rate_metric(merged_monthly_df: pd.DataFrame) -> float:
    if len(merged_monthly_df) < 2:
        return np.nan
    win_months = merged_monthly_df["月收益"] > merged_monthly_df["指数月收益"]
    return win_months.mean()


def calc_attack_metric(merged_monthly_df: pd.DataFrame) -> float:
    if len(merged_monthly_df) < 2:
        return np.nan
    up = merged_monthly_df[merged_monthly_df["指数月收益"] > 0]
    if not up.empty and up["指数月收益"].mean() != 0:
        return up["月收益"].mean() / up["指数月收益"].mean()
    return np.nan


def calc_period_annualized(
        virtual_start_dt: pd.Timestamp,
        current_dt: pd.Timestamp,
        end_aggr: float,
        start_aggr: float = 1.0,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None,
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

    # 区间收益、年化收益
    total_return = calc_return_metric(start_aggr, end_aggr)
    annualized = calc_annualized_return_metric(start_aggr, end_aggr, days_since, annual_days)

    return total_return, annualized, actual_end_dt, days_since


def get_period_start(end_dt: pd.Timestamp, period: str) -> pd.Timestamp:
    if period == "今年以来":
        return pd.Timestamp(year=end_dt.year - 1, month=12, day=31)
    if period.startswith("近") and period.endswith("年"):
        try:
            n_years = int(period[1:-1])
        except ValueError:
            raise ValueError(f"无法解析周期: {period}")

        target_year = end_dt.year - n_years
        target_month = end_dt.month
        target_day = end_dt.day - 1

        if target_day < 1:
            first_day_of_month = pd.Timestamp(
                year=target_year,
                month=target_month,
                day=1
            )
            result = first_day_of_month - pd.Timedelta(days=1)
        else:
            result = pd.Timestamp(
                year=target_year,
                month=target_month,
                day=target_day
            )

        return result

    elif period == "成立以来":
        return None

    else:
        raise ValueError(f"未知周期: {period}")


def calc_alpha_metric_from_daily_returns(
        fund_daily_ret: pd.Series,
        idx_daily_ret: pd.Series,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None
) -> float:
    if len(fund_daily_ret) < 2 or len(idx_daily_ret) < 2:
        return np.nan

    merged = pd.DataFrame({
        "fund_ret": fund_daily_ret,
        "idx_ret": idx_daily_ret
    }).dropna()

    if len(merged) < 2:
        return np.nan

    cov_matrix_raw = np.cov(merged["fund_ret"], merged["idx_ret"], ddof=1)
    cov_fund_idx = cov_matrix_raw[0, 1]
    var_idx = cov_matrix_raw[1, 1]

    if var_idx == 0:
        beta = 0
    else:
        beta = cov_fund_idx / var_idx

    ann_rf = 0.015

    if day_type == "自然日":
        daily_rf = ann_rf / 365
        fund_excess_ret = merged["fund_ret"] - daily_rf
        idx_excess_ret = merged["idx_ret"] - daily_rf
    else:
        fund_excess_ret_list = []
        idx_excess_ret_list = []
        daily_rf_list = []
        holiday_days_list = []

        for i in range(len(merged)):
            current_date = merged.index[i]

            if trading_days_df is not None:
                trading_dates = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
                next_trading_date = trading_dates[trading_dates > current_date]

                if len(next_trading_date) == 0:
                    holiday_days = 1
                else:
                    next_trading_date = next_trading_date.iloc[0]
                    holiday_days = (next_trading_date - current_date).days
            else:
                holiday_days = 1

            daily_rf = (ann_rf / 365) * holiday_days
            daily_rf_list.append(daily_rf)
            holiday_days_list.append(holiday_days)

            fund_excess_ret_list.append(merged.iloc[i]["fund_ret"] - daily_rf)
            idx_excess_ret_list.append(merged.iloc[i]["idx_ret"] - daily_rf)

        fund_excess_ret = pd.Series(fund_excess_ret_list, index=merged.index)
        idx_excess_ret = pd.Series(idx_excess_ret_list, index=merged.index)

    alpha = calc_alpha_metric(fund_excess_ret, idx_excess_ret, beta)
    
    return alpha


def calc_monthly_win_rate_metric_from_monthly_returns(
        fund_month: pd.DataFrame,
        idx_month: pd.DataFrame
) -> float:
    merged = fund_month.merge(idx_month, on="年月", how="inner")
    return calc_monthly_win_rate_metric(merged)


def calc_attack_metric_from_monthly_returns(
        fund_month: pd.DataFrame,
        idx_month: pd.DataFrame
) -> float:
    merged = fund_month.merge(idx_month, on="年月", how="inner")
    return calc_attack_metric(merged)


def _add_virtual_start_nav(prd_df_clean: pd.DataFrame, virtual_start_dt: pd.Timestamp) -> pd.DataFrame:
    virtual_start_nav = pd.DataFrame({
        "NAV_DT": [virtual_start_dt],
        "AGGR_UNIT_NVAL": [1.0]
    })
    full_nav_series = pd.concat(
        [virtual_start_nav, prd_df_clean[["NAV_DT", "AGGR_UNIT_NVAL"]]],
        ignore_index=True
    )
    return full_nav_series.sort_values("NAV_DT").reset_index(drop=True)


def _filter_complete_month_returns(month_df: pd.DataFrame, value_col: str, complete_months: list) -> pd.DataFrame:
    month_df = month_df.copy()
    if "上月末净值" not in month_df.columns:
        month_df["上月末净值"] = month_df["月末净值"].shift(1)
    month_df[value_col] = month_df["月末净值"] / month_df["上月末净值"] - 1

    if len(month_df) >= 2 and complete_months:
        months_to_keep = set(complete_months)
        for month in complete_months:
            months_to_keep.add(month - 1)
        month_df = month_df[month_df["年月"].isin(months_to_keep)].reset_index(drop=True)

    month_df["是否完整月"] = month_df["年月"].isin(complete_months) if complete_months else True
    result = month_df[["年月", value_col, "是否完整月"]].dropna()
    return result[result["是否完整月"]].drop(columns=["是否完整月"])


def _has_missing_data_in_period(
        nav_df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None
) -> bool:
    period_nav = nav_df[(nav_df["NAV_DT"] >= start_dt) & (nav_df["NAV_DT"] <= end_dt)].copy()
    if period_nav.empty:
        return True

    actual_dates = set(period_nav["NAV_DT"].dropna().unique())
    if day_type == "交易日" and trading_days_df is not None:
        expected_dates = set(
            trading_days_df[
                (trading_days_df["IS_TRD_DT"] == True) &
                (trading_days_df["CALD_DATE"] >= start_dt) &
                (trading_days_df["CALD_DATE"] <= end_dt)
            ]["CALD_DATE"].unique()
        )
    else:
        expected_dates = set(pd.date_range(start=start_dt, end=end_dt, freq="D"))

    return len(expected_dates - actual_dates) > 0


def _get_idx_start_date_by_anchor(idx_df: pd.DataFrame, anchor_dt: pd.Timestamp) -> pd.Timestamp:
    idx_before = idx_df[idx_df["NAV_DT"] <= anchor_dt]
    if len(idx_before) == 0:
        return idx_df["NAV_DT"].min()
    return idx_before["NAV_DT"].max()


def _normalize_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    return pd.to_datetime(ts)


def _build_metric_row(
        day_type: str,
        prd_code: str,
        prd_typ: str,
        period: str,
        calc_base_dt,
        theory_start_dt_for_rank,
        idx_start_dt,
        idx_end_dt,
        total_ret=np.nan,
        ann_ret=np.nan,
        alpha=np.nan,
        win_rate=np.nan,
        attack=np.nan
) -> dict:
    return {
        "计算模式": day_type,
        "产品代码": prd_code,
        "产品类型": prd_typ,
        "周期": period,
        "计算基准日": calc_base_dt.date() if pd.notna(calc_base_dt) else pd.NaT,
        "THEORY_START_DT": theory_start_dt_for_rank,
        "INDEX_START_DT_CALC": idx_start_dt,
        "INDEX_END_DT_CALC": idx_end_dt,
        "收益": total_ret,
        "年化收益": ann_ret,
        "Alpha": alpha,
        "月超额胜率": win_rate,
        "进攻能力": attack
    }


def _resolve_actual_theoretical_start(
        period: str,
        fund_established_dt: pd.Timestamp,
        theoretical_start_dt: pd.Timestamp,
        day_type: str,
        trading_days_df: pd.DataFrame = None
) -> pd.Timestamp:
    if period == "成立以来":
        return fund_established_dt

    if day_type == "交易日" and trading_days_df is not None:
        trading_dates_sorted = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
        prev_trading_days = trading_dates_sorted[trading_dates_sorted <= theoretical_start_dt]
        if len(prev_trading_days) > 0:
            return prev_trading_days.max()
    return theoretical_start_dt


def _calc_complete_months(
        full_nav_series: pd.DataFrame,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None
) -> list:
    complete_months = []
    if len(full_nav_series) < 1:
        return complete_months

    actual_data_start = full_nav_series["NAV_DT"].min()
    actual_data_end = full_nav_series["NAV_DT"].max()
    all_months = full_nav_series["年月"].unique()

    for month_period in all_months:
        is_complete = True

        if month_period == actual_data_start.to_period("M"):
            if day_type == "交易日" and trading_days_df is not None:
                trading_dates_in_start_month = trading_days_df[
                    (trading_days_df["IS_TRD_DT"] == True) &
                    (trading_days_df["CALD_DATE"].dt.to_period("M") == month_period)
                    ]["CALD_DATE"]
                if len(trading_dates_in_start_month) > 0:
                    first_trading_day_of_month = trading_dates_in_start_month.min()
                    if actual_data_start > first_trading_day_of_month:
                        is_complete = False
            else:
                if actual_data_start.day != 1:
                    is_complete = False

        if month_period == actual_data_end.to_period("M"):
            if day_type == "交易日" and trading_days_df is not None:
                trading_dates_in_end_month = trading_days_df[
                    (trading_days_df["IS_TRD_DT"] == True) &
                    (trading_days_df["CALD_DATE"].dt.to_period("M") == month_period)
                    ]["CALD_DATE"]
                if len(trading_dates_in_end_month) > 0:
                    last_trading_day_of_month = trading_dates_in_end_month.max()
                    if actual_data_end < last_trading_day_of_month:
                        is_complete = False
            else:
                end_month_last_day = actual_data_end + pd.offsets.MonthEnd(0)
                if actual_data_end.day != end_month_last_day.day:
                    is_complete = False

        if is_complete:
            complete_months.append(month_period)

    return complete_months


def _calc_monthly_return_series(
        nav_series: pd.DataFrame,
        value_col: str,
        output_col: str,
        complete_months: list,
        period: str,
        start_nav_value: float = np.nan
) -> pd.DataFrame:
    month_end_nav = (
        nav_series.groupby("年月")[value_col]
        .last()
        .reset_index(name="月末净值")
    )
    month_end_nav["上月末净值"] = month_end_nav["月末净值"].shift(1)
    if period == "今年以来" and len(month_end_nav) > 0:
        first_idx = month_end_nav.index[0]
        if pd.isna(month_end_nav.loc[first_idx, "上月末净值"]) and pd.notna(start_nav_value) and start_nav_value > 0:
            month_end_nav.loc[first_idx, "上月末净值"] = start_nav_value
    return _filter_complete_month_returns(month_end_nav, output_col, complete_months)


def _calc_hs300_metrics_for_row(
        idx_sub: pd.DataFrame,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None,
        trading_dates_set: set = None,
) -> dict:
    if day_type == "交易日" and trading_days_df is not None:
        trading_dates = trading_dates_set if trading_dates_set is not None else set(
            trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values
        )
        idx_sub = idx_sub[idx_sub["NAV_DT"].isin(trading_dates)].copy()

    if len(idx_sub) < 2:
        return {
            "沪深300收益": np.nan,
            "沪深300年化收益": np.nan,
            "沪深300Alpha": np.nan,
            "沪深300月超额胜率": np.nan,
            "沪深300进攻能力": np.nan
        }

    ret = idx_sub["INDEX_CLOSE"].iloc[-1] / idx_sub["INDEX_CLOSE"].iloc[0] - 1

    if day_type == "交易日" and trading_days_df is not None:
        # 与产品口径完全对齐：从交易日表统计区间内的交易日数量
        start_dt = idx_sub["NAV_DT"].iloc[0]
        end_dt = idx_sub["NAV_DT"].iloc[-1]

        trading_days_in_range = trading_days_df[
            (trading_days_df["CALD_DATE"] > start_dt) &
            (trading_days_df["CALD_DATE"] <= end_dt) &
            (trading_days_df["IS_TRD_DT"] == True)
        ]
        days = max(len(trading_days_in_range), 1)
        annual_days = 250
    else:
        start_dt = idx_sub["NAV_DT"].iloc[0]
        end_dt = idx_sub["NAV_DT"].iloc[-1]
        days = (end_dt - start_dt).days
        annual_days = 365

    ann_ret = (1 + ret) ** (annual_days / max(days, 1)) - 1

    return {
        "沪深300收益": ret,
        "沪深300年化收益": ann_ret,
        "沪深300Alpha": 0,
        "沪深300月超额胜率": np.nan,
        "沪深300进攻能力": 1
    }


def _is_product_eligible_for_row(
        comp_row: pd.Series,
        target_row: pd.Series,
        product_base_info: pd.DataFrame = None,
        established_dt_map: dict = None
) -> bool:
    comp_prd_code = comp_row["产品代码"]
    if established_dt_map is not None:
        comp_established_dt = established_dt_map.get(comp_prd_code, pd.NaT)
    else:
        comp_base = product_base_info[product_base_info["PRD_CODE"] == comp_prd_code]
        if len(comp_base) == 0:
            return False
        comp_established_dt = comp_base["FOUND_DT"].iloc[0]
    if pd.isna(comp_established_dt):
        return False

    period = target_row["周期"]
    target_start = _normalize_timestamp(target_row["THEORY_START_DT"])

    if pd.isna(target_start):
        return False

    if period == "成立以来":
        target_code = target_row["产品代码"]
        if established_dt_map is not None:
            target_established_dt = established_dt_map.get(target_code, pd.NaT)
        else:
            target_base = product_base_info[product_base_info["PRD_CODE"] == target_code]
            if len(target_base) == 0:
                return False
            target_established_dt = target_base["FOUND_DT"].iloc[0]
        if pd.isna(target_established_dt):
            return False
        return comp_established_dt <= target_established_dt

    # 非成立以来：成立日在计算区间之后（> 区间起始日）不参与
    return comp_established_dt <= target_start


def _get_comparable_product_indices(
        df: pd.DataFrame,
        target_row: pd.Series,
        product_base_info: pd.DataFrame = None,
        established_dt_map: dict = None,
        candidate_indices: list = None
) -> list:
    comparable_indices = []
    if candidate_indices is None:
        source_iter = df.iterrows()
    else:
        source_iter = ((idx, df.loc[idx]) for idx in candidate_indices)

    for comp_idx, comp_row in source_iter:
        if comp_row["产品类型"] != target_row["产品类型"]:
            continue
        if comp_row["周期"] != target_row["周期"]:
            continue
        if comp_row["计算基准日"] != target_row["计算基准日"]:
            continue
        if _is_product_eligible_for_row(
                comp_row,
                target_row,
                product_base_info=product_base_info,
                established_dt_map=established_dt_map
        ):
            comparable_indices.append(comp_idx)
    return comparable_indices


# --------------------------------------------------
# 5. 单产品指标计算
# --------------------------------------------------
def calc_product_metrics(prd_df, idx_df, periods, fund_established_dt,
                         day_type="自然日", trading_days_df=None, trading_dates_set=None):
    prd_df_clean = prd_df[prd_df["NAV_DT"] >= fund_established_dt].copy()

    if prd_df_clean.empty:
        return []

    prd_df_clean = prd_df_clean.sort_values("NAV_DT")
    base_dt = prd_df_clean["NAV_DT"].max()

    prd_typ = prd_df_clean["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    prd_code = prd_df_clean["PRD_CODE"].iloc[0]
    
    results = []

    for period in periods:
        theoretical_start_dt = get_period_start(base_dt, period) if period != "成立以来" else None

        if period == "成立以来":
            theory_start_dt_for_rank = fund_established_dt
        else:
            theory_start_dt_for_rank = theoretical_start_dt

        actual_theoretical_start = _resolve_actual_theoretical_start(
            period=period,
            fund_established_dt=fund_established_dt,
            theoretical_start_dt=theoretical_start_dt,
            day_type=day_type,
            trading_days_df=trading_days_df
        )

        if period == "成立以来":
            actual_theoretical_start = fund_established_dt - pd.Timedelta(days=1)
            nav_start = 1.0
            actual_start_dt = actual_theoretical_start
            virtual_start_dt = actual_theoretical_start
        else:
            sub_at_start = prd_df_clean[prd_df_clean["NAV_DT"] == actual_theoretical_start]
            has_start_data = len(sub_at_start) > 0
            if not has_start_data:
                results.append(_build_metric_row(
                    day_type=day_type,
                    prd_code=prd_code,
                    prd_typ=prd_typ,
                    period=period,
                    calc_base_dt=base_dt,
                    theory_start_dt_for_rank=theory_start_dt_for_rank,
                    idx_start_dt=actual_theoretical_start,
                    idx_end_dt=base_dt
                ))
                continue

            nav_start = sub_at_start.iloc[0]["AGGR_UNIT_NVAL"]
            actual_start_dt = actual_theoretical_start
            virtual_start_dt = actual_theoretical_start

        sub_end = prd_df_clean[prd_df_clean["NAV_DT"] <= base_dt]
        if len(sub_end) == 0:
            continue

        nav_end = sub_end.iloc[-1]["AGGR_UNIT_NVAL"]
        current_dt = sub_end.iloc[-1]["NAV_DT"]

        # 严格区间：周期起始日至计算基准日任意一天缺值，则该周期不纳入计算
        missing_check_start_dt = fund_established_dt if period == "成立以来" else actual_theoretical_start
        if _has_missing_data_in_period(
                nav_df=prd_df_clean,
                start_dt=missing_check_start_dt,
                end_dt=current_dt,
                day_type=day_type,
                trading_days_df=trading_days_df
        ):
            results.append(_build_metric_row(
                day_type=day_type,
                prd_code=prd_code,
                prd_typ=prd_typ,
                period=period,
                calc_base_dt=current_dt,
                theory_start_dt_for_rank=theory_start_dt_for_rank,
                idx_start_dt=actual_theoretical_start,
                idx_end_dt=current_dt
            ))
            continue

        if pd.isna(nav_start) or pd.isna(nav_end):
            continue

        total_ret, ann_ret, actual_end_dt, _ = calc_period_annualized(
            virtual_start_dt=virtual_start_dt,
            current_dt=current_dt,
            end_aggr=nav_end,
            start_aggr=nav_start,
            day_type=day_type,
            trading_days_df=trading_days_df,
        )

        if actual_end_dt != current_dt:
            actual_end_data = prd_df_clean[prd_df_clean["NAV_DT"] == actual_end_dt]
            if len(actual_end_data) > 0:
                end_aggr_adjusted = actual_end_data.iloc[0]["AGGR_UNIT_NVAL"]
                total_ret, ann_ret, actual_end_dt, _ = calc_period_annualized(
                    virtual_start_dt=virtual_start_dt,
                    current_dt=actual_end_dt,
                    end_aggr=end_aggr_adjusted,
                    start_aggr=nav_start,
                    day_type=day_type,
                    trading_days_df=trading_days_df,
                )
                actual_end_dt_for_monthly = actual_end_dt
            else:
                actual_end_dt_for_monthly = current_dt
        else:
            actual_end_dt_for_monthly = current_dt

        if period == "成立以来":
            full_nav_series = _add_virtual_start_nav(prd_df_clean, actual_theoretical_start)
        else:
            full_nav_series = prd_df_clean[prd_df_clean["NAV_DT"] >= actual_theoretical_start].copy()
            full_nav_series = full_nav_series.sort_values("NAV_DT").reset_index(drop=True)

        if day_type == "交易日" and trading_days_df is not None:
            trading_dates = trading_dates_set if trading_dates_set is not None else set(
                trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values
            )
            full_nav_series = full_nav_series[
                full_nav_series["NAV_DT"].isin(trading_dates) |
                (full_nav_series["NAV_DT"] == actual_theoretical_start)
                ].reset_index(drop=True)

        full_nav_series["年月"] = full_nav_series["NAV_DT"].dt.to_period("M")

        complete_months = _calc_complete_months(
            full_nav_series=full_nav_series,
            day_type=day_type,
            trading_days_df=trading_days_df
        )
        fund_month = _calc_monthly_return_series(
            nav_series=full_nav_series,
            value_col="AGGR_UNIT_NVAL",
            output_col="月收益",
            complete_months=complete_months,
            period=period,
            start_nav_value=nav_start
        )

        # ========== 用单位净值填充累计净值的缺失值 ==========
        nav_series = full_nav_series.set_index("NAV_DT")["AGGR_UNIT_NVAL"]
        if "UNIT_NVAL" in full_nav_series.columns:
            unit_nav_series = full_nav_series.set_index("NAV_DT")["UNIT_NVAL"]
            # 如果累计净值为NaN，则使用单位净值填充
            nav_series = nav_series.fillna(unit_nav_series)
        
        fund_daily_ret = nav_series.pct_change().dropna()

        idx_start_date = actual_theoretical_start
        idx_end_date = current_dt

        idx_sub = idx_df[
            (idx_df["NAV_DT"] >= idx_start_date) &
            (idx_df["NAV_DT"] <= idx_end_date)
            ]

        if len(idx_sub) < 2:
            results.append(_build_metric_row(
                day_type=day_type,
                prd_code=prd_code,
                prd_typ=prd_typ,
                period=period,
                calc_base_dt=current_dt,
                theory_start_dt_for_rank=theory_start_dt_for_rank,
                idx_start_dt=idx_start_date,
                idx_end_dt=idx_end_date,
                total_ret=total_ret,
                ann_ret=ann_ret
            ))
            continue

        idx_sub_copy = idx_sub.copy()
        idx_sub_copy["年月"] = idx_sub_copy["NAV_DT"].dt.to_period("M")

        # ========== 足月逻辑：只保留完整月份（与基金保持一致）==========
        if len(idx_sub_copy) > 0:
            # 使用基金的起始月和结束月作为基准
            if len(full_nav_series) > 0:
                fund_start_month = full_nav_series["NAV_DT"].min().to_period("M")
                fund_end_month = full_nav_series["NAV_DT"].max().to_period("M")

                # 只保留基金完整月份范围内的指数数据
                idx_sub_copy = idx_sub_copy[
                    (idx_sub_copy["NAV_DT"].dt.to_period("M") >= fund_start_month) &
                    (idx_sub_copy["NAV_DT"].dt.to_period("M") <= fund_end_month)
                    ].reset_index(drop=True)

        idx_start_row = idx_sub_copy[idx_sub_copy["NAV_DT"] == idx_start_date]
        if len(idx_start_row) > 0:
            idx_start_close = idx_start_row.iloc[0]["INDEX_CLOSE"]
        elif len(idx_sub_copy) > 0:
            idx_start_close = idx_sub_copy.iloc[0]["INDEX_CLOSE"]
        else:
            idx_start_close = np.nan
        idx_month = _calc_monthly_return_series(
            nav_series=idx_sub_copy,
            value_col="INDEX_CLOSE",
            output_col="指数月收益",
            complete_months=complete_months,
            period=period,
            start_nav_value=idx_start_close
        )

        idx_daily_ret = idx_sub_copy.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()

        # ========== 自然日模式：对齐基金和指数的日期，确保两者都有数据 ==========
        if day_type == "自然日":
            # 取基金和指数都有数据的日期交集
            common_dates = fund_daily_ret.index.intersection(idx_daily_ret.index)
            fund_daily_ret = fund_daily_ret.loc[common_dates]
            idx_daily_ret = idx_daily_ret.loc[common_dates]

        if day_type == "交易日" and trading_days_df is not None:
            trading_dates = trading_dates_set if trading_dates_set is not None else set(
                trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values
            )
            idx_daily_ret = idx_daily_ret[idx_daily_ret.index.isin(trading_dates)]

        alpha_value = calc_alpha_metric_from_daily_returns(
            fund_daily_ret=fund_daily_ret,
            idx_daily_ret=idx_daily_ret,
            day_type=day_type,
            trading_days_df=trading_days_df
        )

        monthly_win_rate = calc_monthly_win_rate_metric_from_monthly_returns(fund_month, idx_month)
        attack_value = calc_attack_metric_from_monthly_returns(fund_month, idx_month)

        results.append(_build_metric_row(
            day_type=day_type,
            prd_code=prd_code,
            prd_typ=prd_typ,
            period=period,
            calc_base_dt=current_dt,
            theory_start_dt_for_rank=theory_start_dt_for_rank,
            idx_start_dt=idx_start_date,
            idx_end_dt=idx_end_date,
            total_ret=total_ret,
            ann_ret=ann_ret,
            alpha=alpha_value,
            win_rate=monthly_win_rate,
            attack=attack_value
        ))

    return results


# --------------------------------------------------
# 6. 同类平均 & 沪深300（基于全部产品）
# --------------------------------------------------
def build_benchmark_and_avg(
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        product_base_info: pd.DataFrame,
        day_type="自然日",
        trading_days_df=None
) -> pd.DataFrame:
    result = df.copy()
    metrics = ["收益", "年化收益", "Alpha", "月超额胜率", "进攻能力"]
    established_dt_map = (
        product_base_info.dropna(subset=["PRD_CODE"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")["FOUND_DT"]
        .to_dict()
    )
    trading_dates_set = None
    if day_type == "交易日" and trading_days_df is not None:
        trading_dates_set = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)
    group_to_indices = result.groupby(["产品类型", "周期", "计算基准日"]).groups

    for metric in metrics:
        result[f"同类平均{metric}"] = np.nan
        result[f"沪深300{metric}"] = np.nan

    for idx, row in result.iterrows():
        group_key = (row["产品类型"], row["周期"], row["计算基准日"])
        candidate_indices = list(group_to_indices.get(group_key, []))
        comparable_indices = _get_comparable_product_indices(
            result,
            row,
            product_base_info=product_base_info,
            established_dt_map=established_dt_map,
            candidate_indices=candidate_indices
        )
        
        if len(comparable_indices) > 0:
            comparable_df = result.loc[comparable_indices]

            for metric in metrics:
                result.loc[idx, f"同类平均{metric}"] = comparable_df[metric].mean()

        idx_start_dt = _normalize_timestamp(row.get("INDEX_START_DT_CALC", pd.NaT))
        idx_end_dt = _normalize_timestamp(row.get("INDEX_END_DT_CALC", pd.NaT))
        if pd.isna(idx_start_dt) or pd.isna(idx_end_dt):
            continue

        idx_sub = idx_df[
            (idx_df["NAV_DT"] >= idx_start_dt) &
            (idx_df["NAV_DT"] <= idx_end_dt)
            ]
        
        hs300_metrics = _calc_hs300_metrics_for_row(
            idx_sub=idx_sub,
            day_type=day_type,
            trading_days_df=trading_days_df,
            trading_dates_set=trading_dates_set,
        )
        for metric_name, metric_value in hs300_metrics.items():
            result.loc[idx, metric_name] = metric_value

    return result


# --------------------------------------------------
# 7. 排名（使用成立日筛选逻辑）
# --------------------------------------------------
def rank_by_established_date(df: pd.DataFrame, product_base_info: pd.DataFrame) -> pd.DataFrame:
    metrics = ["收益", "年化收益", "Alpha", "月超额胜率", "进攻能力"]
    established_dt_map = (
        product_base_info.dropna(subset=["PRD_CODE"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")["FOUND_DT"]
        .to_dict()
    )
    group_to_indices = df.groupby(["产品类型", "周期", "计算基准日"]).groups

    # 初始化排名列为空字符串
    for col in metrics:
        df[f"{col}排名"] = ""

    # 对每个产品单独计算排名
    for idx, row in df.iterrows():
        prd_code = row["产品代码"]
        # 跳过基准产品（兼容历史口径）
        if prd_code in ["同类平均", "沪深300", "沪深 300"]:
            continue

        if pd.isna(established_dt_map.get(prd_code, pd.NaT)):
            continue

        if not _is_product_eligible_for_row(row, row, established_dt_map=established_dt_map):
            continue

        group_key = (row["产品类型"], row["周期"], row["计算基准日"])
        candidate_indices = list(group_to_indices.get(group_key, []))
        comparable_products = _get_comparable_product_indices(
            df,
            row,
            product_base_info=product_base_info,
            established_dt_map=established_dt_map,
            candidate_indices=candidate_indices
        )

        if len(comparable_products) == 0:
            continue

        # 在这些可比产品中计算排名
        comparable_df = df.loc[comparable_products]

        # 对每个指标分别排名
        for metric in metrics:
            # 只对有有效值的产品进行排名
            valid_mask = comparable_df[metric].notna()
            valid_df = comparable_df[valid_mask]
            
            if len(valid_df) == 0 or idx not in valid_df.index:
                continue
            
            rank_series = valid_df[metric].rank(method="min", ascending=False).astype(int)
            
            my_rank = rank_series[idx]
            total_count = len(valid_df)  # 只统计有效产品的数量

            # 赋值排名
            df.loc[idx, f"{metric}排名"] = f"{my_rank}/{total_count}名"

    return df


# --------------------------------------------------
# 9. 主流程
# --------------------------------------------------
def main(index_secu_id="000300.IDX.CSIDX", day_type="交易日"):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]

    prd_df = fetch_fin_prd_nav()
    idx_df = fetch_index_quote(index_secu_id)
    product_base_info = fetch_product_base_info()

    # 获取交易日数据（仅当day_type为交易日时）
    trading_days_df = None
    trading_dates_set = None
    if day_type == "交易日":
        trading_days_df = fetch_trading_days()
        trading_dates_set = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)

    # ========== 产品类型填充 ==========
    prd_df = fill_special_prd_typ(prd_df, product_base_info)

    established_dt_map = (
        product_base_info.dropna(subset=["PRD_CODE"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")["FOUND_DT"]
        .to_dict()
    )

    grouped_products = list(prd_df.groupby("PRD_CODE"))
    max_workers = min(8, max(1, (os.cpu_count() or 4)))
    future_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for prd_code, g in grouped_products:
            fund_established_dt = established_dt_map.get(prd_code, pd.NaT)
            if pd.isna(fund_established_dt):
                future_list.append(None)
                continue
            future = executor.submit(
                calc_product_metrics,
                g,
                idx_df,
                periods,
                fund_established_dt,
                day_type,
                trading_days_df,
                trading_dates_set
            )
            future_list.append(future)

        all_results = []
        for future in future_list:
            if future is None:
                continue
            all_results.extend(future.result())

    df = pd.DataFrame(all_results)
    df_rank = build_benchmark_and_avg(df, idx_df, product_base_info, day_type, trading_days_df)
    df_rank = rank_by_established_date(df_rank, product_base_info)

    # ========== 产品指标 → 指数指标 → 同类平均指标 → 排名 ==========
    base_cols = ["产品代码", "产品类型", "周期", "计算基准日", "THEORY_START_DT", "计算模式"]

    # 产品自身指标
    product_metrics = ["收益", "年化收益", "Alpha", "月超额胜率", "进攻能力"]

    # 沪深300指数指标
    index_metrics = ["沪深300收益", "沪深300年化收益", "沪深300Alpha", "沪深300月超额胜率", "沪深300进攻能力"]

    # 同类平均指标
    category_avg_metrics = ["同类平均收益", "同类平均年化收益", "同类平均Alpha", "同类平均月超额胜率",
                            "同类平均进攻能力"]

    # 排名指标
    rank_metrics = ["收益排名", "年化收益排名", "Alpha排名", "月超额胜率排名", "进攻能力排名"]

    # 按顺序组合所有列
    ordered_cols = base_cols + product_metrics + index_metrics + category_avg_metrics + rank_metrics

    # 只保留存在的列（避免 KeyError）
    final_cols = [col for col in ordered_cols if col in df_rank.columns]
    df_rank = df_rank[final_cols]

    pct_cols = [
        "收益", "年化收益", "Alpha", "月超额胜率", "进攻能力",
        "同类平均收益", "同类平均年化收益", "同类平均Alpha", "同类平均月超额胜率", "同类平均进攻能力",
        "沪深300收益", "沪深300年化收益", "沪深300Alpha", "沪深300月超额胜率", "沪深300进攻能力"
    ]
    for col in pct_cols:
        if col in df_rank.columns:
            df_rank[col] = df_rank[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else x
            )

    df_rank = df_rank.where(pd.notna(df_rank), "--")

    return df_rank


# --------------------------------------------------
# 10. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    print("开始计算收益能力指标...")
    result = main(day_type="自然日")

    if result is not None and len(result) > 0:
        print(f"\n计算完成！共 {len(result)} 条记录")
        print("\n前10条数据预览：")
        print(result.head(10).to_string())

        result.to_csv("收益能力指标.csv", index=False, encoding="utf-8-sig", quoting=1)
        print("\n💾 结果已保存至：收益能力指标.csv")
    else:
        print("\n⚠️ 警告：没有计算出任何数据！")
