import os
import oracledb
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from dateutil.relativedelta import relativedelta

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


def fetch_pty_prd_base_info() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE, \
                 FOUND_DT, \
                 PRD_NAME, \
                 PRD_FULL_NAME, \
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


def fetch_trading_days() -> pd.DataFrame:
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


def fill_special_prd_typ(df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    df = df_nav.copy()
    if "PRD_TYP" in df_base_info.columns:
        base_prd_typ_map = (
            df_base_info[df_base_info["PRD_TYP"].notna()]
            .groupby("PRD_CODE")["PRD_TYP"]
            .first()
            .to_dict()
        )
        mask_base = df["PRD_TYP"].isna() & df["PRD_CODE"].isin(base_prd_typ_map)
        df.loc[mask_base, "PRD_TYP"] = df.loc[mask_base, "PRD_CODE"].map(base_prd_typ_map)
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
# 3. 工具函数（理论起算日、缺失检查与「收益能力指标」/「风险控制指标」对齐）
# --------------------------------------------------
def _normalize_period_key(period: str) -> str:
    return period.replace(" ", "").strip()


def get_period_dates_for_drawdown(
        end_dt: pd.Timestamp,
        period: str,
        market_df: pd.DataFrame,
) -> tuple:
    """
    成立以来：起点为行情（指数）最早日期；今年以来：上年末；近 n 年/月：先偏移再「日减 1」规则。
    与「风险控制指标」get_period_dates_for_drawdown 一致。
    """
    period = _normalize_period_key(period)
    if period == "成立以来":
        start_dt = market_df["NAV_DT"].min()
        return start_dt, end_dt
    if period == "今年以来":
        return pd.Timestamp(year=end_dt.year - 1, month=12, day=31), end_dt
    if period.startswith("近") and period.endswith("年"):
        n_years_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
        chinese_num = period[1:-1]
        n_years = n_years_map.get(chinese_num)
        if n_years is None:
            try:
                n_years = int(chinese_num)
            except ValueError:
                n_years = 1
        target_year = end_dt.year - n_years
        target_month = end_dt.month
        target_day = end_dt.day - 1
        if target_day < 1:
            first_day_of_month = pd.Timestamp(target_year, target_month, 1)
            result = first_day_of_month - timedelta(days=1)
        else:
            result = pd.Timestamp(target_year, target_month, target_day)
        return result, end_dt
    if period.startswith("近") and period.endswith("月"):
        n_months_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        chinese_num = period[1:-1]
        n_months = n_months_map.get(chinese_num)
        if n_months is None:
            try:
                n_months = int(chinese_num)
            except ValueError:
                n_months = 1
        target_date = end_dt - relativedelta(months=n_months)
        target_day = target_date.day - 1
        if target_day < 1:
            first_day_of_month = pd.Timestamp(target_date.year, target_date.month, 1)
            result = first_day_of_month - timedelta(days=1)
        else:
            result = pd.Timestamp(target_date.year, target_date.month, target_day)
        return result, end_dt
    raise ValueError(f"未知周期：{period}")


def _resolve_actual_theoretical_start(
        period: str,
        fund_established_dt: pd.Timestamp,
        theoretical_start_dt: pd.Timestamp,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> pd.Timestamp:
    """与「收益能力指标」一致：交易日起点为理论起点当日或之前最近交易日。成立以来为成立日。"""
    if _normalize_period_key(period) == "成立以来":
        return fund_established_dt

    if day_type == "交易日" and trading_days_df is not None:
        trading_dates_sorted = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
        prev_trading_days = trading_dates_sorted[trading_dates_sorted <= theoretical_start_dt]
        if len(prev_trading_days) > 0:
            return prev_trading_days.max()
    return theoretical_start_dt


def _has_missing_data_in_period(
        nav_df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        day_type: str = "自然日",
        trading_days_df: pd.DataFrame = None,
) -> bool:
    """
    与「收益能力指标」一致：从 start_dt 到 end_dt 内，自然日需逐日有净值，交易日需每个交易日有净值，否则视为缺失。
    """
    period_nav = nav_df[(nav_df["NAV_DT"] >= start_dt) & (nav_df["NAV_DT"] <= end_dt)]
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


def get_fund_established_dt(prd_df: pd.DataFrame, df_base_info: pd.DataFrame = None) -> pd.Timestamp:
    if df_base_info is not None and len(df_base_info) > 0:
        if pd.notna(df_base_info.iloc[0]["FOUND_DT"]):
            return pd.Timestamp(df_base_info.iloc[0]["FOUND_DT"]).normalize()
    return pd.Timestamp(prd_df["NAV_DT"].min()).normalize()


def _filter_daily_ret_by_trading_days(ret: pd.Series, trading_days_df: pd.DataFrame) -> pd.Series:
    td_idx = pd.to_datetime(
        trading_days_df.loc[trading_days_df["IS_TRD_DT"], "CALD_DATE"]
    ).dt.normalize().unique()
    ix = pd.DatetimeIndex(ret.index).normalize()
    return ret.loc[ix.isin(td_idx)]


def _vol_annual_factor(day_type: str) -> float:
    return np.sqrt(250) if day_type == "交易日" else np.sqrt(365)


def _calculate_holiday_days(
        current_date: pd.Timestamp,
        trading_days_df: pd.DataFrame,
) -> int:
    """与「基金基础信息展示」calculate_holiday_days 一致：当前日到下一交易日的日历间隔天数。"""
    current_date = pd.Timestamp(current_date).normalize()
    trading_dates = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
    next_trading_date = trading_dates[trading_dates > current_date]
    if len(next_trading_date) == 0:
        return 1
    next_trading_date = pd.Timestamp(next_trading_date.iloc[0]).normalize()
    return max(int((next_trading_date - current_date).days), 1)


def _compute_sharpe_stats_basic_info_style(
        nav_block: pd.DataFrame,
        day_type: str,
        trading_days_df: pd.DataFrame,
        ann_rf: float,
        virtual_start_dt: pd.Timestamp = None,
        virtual_start_nav: float = None,
) -> tuple:
    """
    与「基金基础信息展示」夏普口径一致；返回
    (年化夏普, 平均日超额收益率, 日收益率标准差, 年化系数 250/365)。
    年化夏普 = (平均日超额收益率 / 日收益率标准差) × √年化系数。
    """
    sharpe_annual_factor = 250 if day_type == "交易日" else 365
    nan_pack = (np.nan, np.nan, np.nan, float(sharpe_annual_factor))

    full = nav_block[["NAV_DT", "AGGR_UNIT_NVAL"]].copy().sort_values("NAV_DT").reset_index(drop=True)
    if virtual_start_dt is not None and virtual_start_nav is not None:
        vs = pd.Timestamp(virtual_start_dt).normalize()
        vrow = pd.DataFrame({"NAV_DT": [vs], "AGGR_UNIT_NVAL": [float(virtual_start_nav)]})
        full = pd.concat([vrow, full], ignore_index=True).sort_values("NAV_DT").reset_index(drop=True)

    if day_type == "交易日" and trading_days_df is not None:
        trading_dates = set(
            pd.Timestamp(x).normalize()
            for x in trading_days_df.loc[trading_days_df["IS_TRD_DT"], "CALD_DATE"]
        )
        m = full["NAV_DT"].dt.normalize().isin(trading_dates)
        if virtual_start_dt is not None:
            vs = pd.Timestamp(virtual_start_dt).normalize()
            m = m | (full["NAV_DT"].dt.normalize() == vs)
        full = full.loc[m].sort_values("NAV_DT").reset_index(drop=True)

    if len(full) < 2:
        return nan_pack

    daily_ret = full["AGGR_UNIT_NVAL"].pct_change().dropna()
    if len(daily_ret) == 0:
        return nan_pack

    daily_ret_std = daily_ret.std()
    if day_type == "自然日":
        daily_rf = ann_rf / 365.0
        daily_excess_ret_mean = (daily_ret - daily_rf).mean()
    else:
        if trading_days_df is None:
            daily_rf = ann_rf / 250.0
            daily_excess_ret_mean = (daily_ret - daily_rf).mean()
        else:
            daily_rf_list = []
            for current_idx in daily_ret.index:
                current_date = full.loc[current_idx, "NAV_DT"]
                holiday_days = _calculate_holiday_days(current_date, trading_days_df)
                daily_rf_list.append((ann_rf / 365.0) * holiday_days)
            daily_rf_series = pd.Series(daily_rf_list, index=daily_ret.index)
            daily_excess_ret_mean = (daily_ret - daily_rf_series).mean()

    if daily_ret_std == 0 or pd.isna(daily_ret_std):
        sharpe = np.nan
    else:
        sharpe = float((daily_excess_ret_mean / daily_ret_std) * np.sqrt(sharpe_annual_factor))

    return (
        sharpe,
        float(daily_excess_ret_mean) if pd.notna(daily_excess_ret_mean) else np.nan,
        float(daily_ret_std) if pd.notna(daily_ret_std) else np.nan,
        float(sharpe_annual_factor),
    )


def _apply_day_type_to_fund_idx_returns(
        fund_daily_ret: pd.Series,
        idx_daily_ret: pd.Series,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> tuple:
    """与「风险控制指标」_apply_day_type_to_fund_idx_returns 一致。"""
    if day_type != "交易日" or trading_days_df is None:
        return fund_daily_ret, idx_daily_ret
    f = _filter_daily_ret_by_trading_days(fund_daily_ret, trading_days_df)
    i = _filter_daily_ret_by_trading_days(idx_daily_ret, trading_days_df)
    return f, i


def _fund_daily_ret_risk_control_style(
        sub: pd.DataFrame,
        theory_start_dt: pd.Timestamp,
        fund_established_dt: pd.Timestamp,
) -> pd.Series:
    """与「风险控制指标」基金日收益构造一致（此处用 AGGR_UNIT_NVAL）。"""
    fund_nav_series = sub.set_index("NAV_DT")["AGGR_UNIT_NVAL"].sort_index()
    te = pd.Timestamp(theory_start_dt).normalize()
    fe = pd.Timestamp(fund_established_dt).normalize()
    has_virtual_start = te < fe
    if has_virtual_start and len(fund_nav_series) > 0:
        first_date = fund_nav_series.index[0]
        first_nav = fund_nav_series.iloc[0]
        first_return = (first_nav - 1.0) / 1.0
        if len(fund_nav_series) > 1:
            subsequent_returns = fund_nav_series.pct_change().iloc[1:]
            fund_daily_ret = pd.concat([
                pd.Series([first_return], index=[first_date]),
                subsequent_returns,
            ]).dropna()
        else:
            fund_daily_ret = pd.Series([first_return], index=[first_date])
    else:
        fund_daily_ret = fund_nav_series.pct_change().dropna()
    return fund_daily_ret


def _downside_risk_risk_control_fund(
        sub: pd.DataFrame,
        idx_df: pd.DataFrame,
        theory_start_dt: pd.Timestamp,
        base_dt: pd.Timestamp,
        day_type: str,
        trading_days_df: pd.DataFrame,
        fund_established_dt: pd.Timestamp,
) -> float:
    """
    与「风险控制指标」calc_risk_metrics 中下行风险一致：
    vol_src 为「基金与指数对齐后的日收益」列（若合并不足 2 点则用基金日收益）；
    下行风险 = 日收益序列中负收益的标准差 × √250 或 √365（至少 2 个负收益点）。
    """
    fund_daily_ret = _fund_daily_ret_risk_control_style(sub, theory_start_dt, fund_established_dt)
    if len(fund_daily_ret) == 0:
        return np.nan

    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= theory_start_dt) &
        (idx_df["NAV_DT"] <= base_dt)
    ].copy()
    ann_scale = _vol_annual_factor(day_type)

    if len(idx_sub) < 2:
        vol_src = fund_daily_ret
        if day_type == "交易日" and trading_days_df is not None:
            vol_src = _filter_daily_ret_by_trading_days(vol_src, trading_days_df)
    else:
        idx_sub = idx_sub.sort_values("NAV_DT")
        idx_daily_ret = idx_sub.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()
        fund_m, idx_m = _apply_day_type_to_fund_idx_returns(
            fund_daily_ret, idx_daily_ret, day_type, trading_days_df
        )
        merged = pd.DataFrame({
            "日收益": fund_m,
            "指数日收益": idx_m,
        }).dropna()
        vol_src = merged["日收益"] if len(merged) >= 2 else fund_daily_ret

    if len(vol_src) < 2:
        return np.nan
    neg = vol_src[vol_src < 0]
    if len(neg) < 2:
        return np.nan
    ds = neg.std()
    return float(ds * ann_scale) if pd.notna(ds) else np.nan


def _downside_risk_risk_control_index_only(
        idx_sub: pd.DataFrame,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> float:
    """与「风险控制指标」_calc_hs300_risk_for_row_slice 中指数下行风险一致。"""
    idx_daily_ret = idx_sub.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()
    if day_type == "交易日" and trading_days_df is not None and len(idx_daily_ret) > 0:
        idx_daily_ret = _filter_daily_ret_by_trading_days(idx_daily_ret, trading_days_df)
    ann_scale = _vol_annual_factor(day_type)
    neg = idx_daily_ret[idx_daily_ret < 0]
    if len(neg) < 2:
        return np.nan
    ds = neg.std()
    return float(ds * ann_scale) if pd.notna(ds) else np.nan


def _empty_ratio_result(period: str) -> dict:
    return {
        "周期": period,
        "夏普比率": np.nan,
        "索提诺比率": np.nan,
        "卡玛比率": np.nan,
    }


RATIO_METRICS = ["夏普比率", "索提诺比率", "卡玛比率"]


def _normalize_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    return pd.to_datetime(ts)


def _is_product_eligible_for_row(
        comp_row: pd.Series,
        target_row: pd.Series,
        established_dt_map: dict,
) -> bool:
    comp_prd_code = comp_row["产品代码"]
    comp_established_dt = _normalize_timestamp(established_dt_map.get(comp_prd_code, pd.NaT))
    if pd.isna(comp_established_dt):
        return False

    period = target_row["周期"]
    target_start = _normalize_timestamp(target_row.get("THEORY_START_DT", pd.NaT))
    if pd.isna(target_start):
        return False

    if period == "成立以来":
        target_code = target_row["产品代码"]
        target_established_dt = _normalize_timestamp(established_dt_map.get(target_code, pd.NaT))
        if pd.isna(target_established_dt):
            return False
        return comp_established_dt <= target_established_dt

    return comp_established_dt <= target_start


def _get_comparable_product_indices(
        df: pd.DataFrame,
        target_row: pd.Series,
        established_dt_map: dict,
        candidate_indices: list = None,
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
        if comp_row.get("计算模式") != target_row.get("计算模式"):
            continue
        if _is_product_eligible_for_row(comp_row, target_row, established_dt_map):
            comparable_indices.append(comp_idx)
    return comparable_indices


# --------------------------------------------------
# 4. 周期 NAV 天数（仅用于排名资格，与风险控制指标窗口对齐）
# --------------------------------------------------
def calc_period_nav_days(
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        periods: list,
        established_dt_map: dict,
) -> pd.DataFrame:
    records = []

    for prd_code, g in df.groupby("PRD_CODE"):
        g = g.sort_values("NAV_DT")
        base_dt = g["NAV_DT"].max()
        fund_est = established_dt_map.get(prd_code, g["NAV_DT"].min())
        if pd.notna(fund_est):
            fund_est = pd.Timestamp(fund_est).normalize()

        for period in periods:
            pnorm = _normalize_period_key(period)
            theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)
            theory_start_dt = pd.Timestamp(theory_start_dt).normalize()
            period_end_dt = pd.Timestamp(period_end_dt).normalize()

            if pnorm == "成立以来":
                start_dt = fund_est if pd.notna(fund_est) else g["NAV_DT"].min()
            else:
                start_dt = max(theory_start_dt, fund_est) if pd.notna(fund_est) else theory_start_dt

            nav_days = g[
                (g["NAV_DT"] >= start_dt) &
                (g["NAV_DT"] <= base_dt)
            ]["NAV_DT"].nunique()

            records.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": g["PRD_TYP"].iloc[0] if pd.notna(g["PRD_TYP"].iloc[0]) else "未分类",
                "周期": period,
                "nav_days": nav_days,
            })

    return pd.DataFrame(records)


# --------------------------------------------------
# 5. 风险收益性价比指标计算
# --------------------------------------------------
def calc_risk_adjusted_returns(
        prd_df: pd.DataFrame,
        idx_df: pd.DataFrame,
        period: str,
        df_base_info: pd.DataFrame = None,
        risk_free_rate: float = 0.015,
        day_type: str = "交易日",
        trading_days_df: pd.DataFrame = None,
) -> dict:
    """
    计算单个产品在指定周期内的风险收益性价比指标。
    区间口径与「收益能力指标」/「风险控制指标」一致：理论起点当日（交易日为此前最近交易日）无净值则不计算；
    区间内若存在缺失净值日，则不参与计算。
    夏普比率计算与「基金基础信息展示」一致（日超额收益均值 / 日收益标准差 × √250 或 √365；交易日无风险按间隔天数折算）。
    索提诺比率分子与夏普一致（平均日超额收益率 × √250/√365），分母为「风险控制指标」同款下行风险（负日收益标准差 × √250/√365）。
    """
    pnorm = _normalize_period_key(period)
    prd_df = prd_df.sort_values("NAV_DT")
    base_dt = prd_df["NAV_DT"].max()
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)
    theory_start_dt = pd.Timestamp(theory_start_dt).normalize()
    period_end_dt = pd.Timestamp(period_end_dt).normalize()

    if pnorm == "成立以来":
        if pd.isna(fund_established_dt):
            return _empty_ratio_result(period)
        theory_start_dt = pd.Timestamp(fund_established_dt).normalize() - pd.Timedelta(days=1)
        period_end_dt = base_dt

    prd_df_clean = prd_df
    if pd.notna(fund_established_dt):
        prd_df_clean = prd_df[prd_df["NAV_DT"] >= fund_established_dt].copy()
    if prd_df_clean.empty:
        return _empty_ratio_result(period)

    actual_theoretical_start = _resolve_actual_theoretical_start(
        period, fund_established_dt, theory_start_dt, day_type, trading_days_df
    )
    if pnorm == "成立以来":
        actual_theoretical_start = theory_start_dt

    if pnorm != "成立以来":
        ats = pd.Timestamp(actual_theoretical_start).normalize()
        if prd_df_clean[prd_df_clean["NAV_DT"].dt.normalize() == ats].empty:
            return _empty_ratio_result(period)

    missing_check_start_dt = fund_established_dt if pnorm == "成立以来" else actual_theoretical_start
    if _has_missing_data_in_period(
            nav_df=prd_df_clean,
            start_dt=missing_check_start_dt,
            end_dt=period_end_dt,
            day_type=day_type,
            trading_days_df=trading_days_df,
    ):
        return _empty_ratio_result(period)

    if pd.notna(fund_established_dt):
        actual_start_dt = max(theory_start_dt, pd.Timestamp(fund_established_dt).normalize())
    else:
        actual_start_dt = theory_start_dt
    if pnorm == "成立以来":
        actual_start_dt = pd.Timestamp(fund_established_dt).normalize()

    sub = prd_df[
        (prd_df["NAV_DT"] >= actual_start_dt) &
        (prd_df["NAV_DT"] <= period_end_dt)
    ].copy()

    if len(sub) < 2:
        return _empty_ratio_result(period)

    sub["日收益"] = sub["AGGR_UNIT_NVAL"].pct_change()

    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= theory_start_dt) &
        (idx_df["NAV_DT"] <= base_dt)
    ].copy()

    if len(idx_sub) >= 2:
        idx_sub = idx_sub.sort_values("NAV_DT")
        idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()

    start_nav = sub["AGGR_UNIT_NVAL"].iloc[0]
    end_nav = sub["AGGR_UNIT_NVAL"].iloc[-1]
    if pd.isna(start_nav) or pd.isna(end_nav) or start_nav <= 0:
        return _empty_ratio_result(period)

    total_return = end_nav / start_nav - 1
    days = (sub["NAV_DT"].iloc[-1] - sub["NAV_DT"].iloc[0]).days
    ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1

    nav_for_sharpe = sub[["NAV_DT", "AGGR_UNIT_NVAL"]].copy()
    v_dt = None
    v_nav = None
    if pnorm == "成立以来" and pd.notna(fund_established_dt):
        v_dt = pd.Timestamp(fund_established_dt).normalize() - pd.Timedelta(days=1)
        v_nav = 1.0

    sharpe_ratio, excess_mean, _, sh_fact = _compute_sharpe_stats_basic_info_style(
        nav_for_sharpe,
        day_type=day_type,
        trading_days_df=trading_days_df,
        ann_rf=risk_free_rate,
        virtual_start_dt=v_dt,
        virtual_start_nav=v_nav,
    )

    sortino_numer = (
        excess_mean * np.sqrt(sh_fact)
        if pd.notna(excess_mean) else np.nan
    )
    downside_risk = _downside_risk_risk_control_fund(
        sub,
        idx_df,
        theory_start_dt,
        base_dt,
        day_type,
        trading_days_df,
        fund_established_dt,
    )
    if pd.notna(downside_risk) and downside_risk > 0 and pd.notna(sortino_numer):
        sortino_ratio = float(sortino_numer / downside_risk)
    else:
        sortino_ratio = np.nan

    sub_dd = sub.copy()
    sub_dd["累计净值"] = sub_dd["AGGR_UNIT_NVAL"]
    sub_dd["历史最高"] = sub_dd["累计净值"].cummax()
    sub_dd["回撤"] = (sub_dd["累计净值"] - sub_dd["历史最高"]) / sub_dd["历史最高"]
    max_drawdown = abs(sub_dd["回撤"].min())

    if pd.notna(max_drawdown) and max_drawdown > 0:
        calmar_ratio = ann_return / max_drawdown
    else:
        calmar_ratio = np.nan

    return {
        "周期": period,
        "夏普比率": sharpe_ratio,
        "索提诺比率": sortino_ratio,
        "卡玛比率": calmar_ratio,
    }


# --------------------------------------------------
# 6. 单产品多周期指标计算
# --------------------------------------------------
def calc_product_risk_adjusted_returns(
        prd_df: pd.DataFrame,
        idx_df: pd.DataFrame,
        periods: list,
        df_base_info: pd.DataFrame = None,
        risk_free_rate: float = 0.015,
        day_type: str = "交易日",
        trading_days_df: pd.DataFrame = None,
) -> list:

    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    prd_typ = prd_df["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    results = []

    for period in periods:
        metrics = calc_risk_adjusted_returns(
            prd_df,
            idx_df,
            period,
            df_base_info=df_base_info,
            risk_free_rate=risk_free_rate,
            day_type=day_type,
            trading_days_df=trading_days_df,
        )

        theory_start_dt, _ = get_period_dates_for_drawdown(base_dt, period, idx_df)
        theory_start_dt = pd.Timestamp(theory_start_dt).normalize()
        if _normalize_period_key(period) == "成立以来":
            theory_start_for_rank = pd.Timestamp(fund_established_dt).normalize()
        else:
            theory_start_for_rank = theory_start_dt

        results.append({
            "产品代码": prd_df["PRD_CODE"].iloc[0],
            "产品类型": prd_typ,
            "计算模式": day_type,
            "THEORY_START_DT": theory_start_for_rank,
            **metrics,
        })

    return results


def _calc_product_worker_pack(payload):
    prd_group, idx_df, periods, df_base_info, risk_free_rate, day_type, trading_days_df = payload
    return calc_product_risk_adjusted_returns(
        prd_group,
        idx_df,
        periods,
        df_base_info=df_base_info,
        risk_free_rate=risk_free_rate,
        day_type=day_type,
        trading_days_df=trading_days_df,
    )


# --------------------------------------------------
# 7. 沪深 300 各周期指标（与原先指数行计算口径一致）
# --------------------------------------------------
def _compute_hs300_ratio_row_for_period(
        idx_df: pd.DataFrame,
        period: str,
        risk_free_rate: float,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> dict:
    base_dt = pd.to_datetime(idx_df["NAV_DT"].max())
    idx_min = pd.Timestamp(idx_df["NAV_DT"].min()).normalize()
    pnorm = _normalize_period_key(period)
    theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)
    theory_start_dt = pd.Timestamp(theory_start_dt).normalize()
    period_end_dt = pd.Timestamp(period_end_dt).normalize()

    if pnorm == "成立以来":
        missing_start = idx_min
    else:
        missing_start = _resolve_actual_theoretical_start(
            period, theory_start_dt, theory_start_dt, day_type, trading_days_df
        )

    if pnorm != "成立以来":
        ats = pd.Timestamp(missing_start).normalize()
        if idx_df[idx_df["NAV_DT"].dt.normalize() == ats].empty:
            return {m: np.nan for m in RATIO_METRICS}

    idx_nav_only = idx_df[["NAV_DT"]].drop_duplicates()
    if _has_missing_data_in_period(
            idx_nav_only,
            missing_start,
            period_end_dt,
            day_type=day_type,
            trading_days_df=trading_days_df,
    ):
        return {m: np.nan for m in RATIO_METRICS}

    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= theory_start_dt) &
        (idx_df["NAV_DT"] <= base_dt)
    ].sort_values("NAV_DT")

    if len(idx_sub) < 2:
        return {m: np.nan for m in RATIO_METRICS}

    idx_sub = idx_sub.copy()
    idx_sub["指数日收益"] = idx_sub["INDEX_CLOSE"].pct_change()

    idx_total_return = idx_sub["INDEX_CLOSE"].iloc[-1] / idx_sub["INDEX_CLOSE"].iloc[0] - 1
    days = (idx_sub["NAV_DT"].iloc[-1] - idx_sub["NAV_DT"].iloc[0]).days
    idx_ann_return = (1 + idx_total_return) ** (365 / max(days, 1)) - 1

    idx_nav_for_sharpe = idx_sub[["NAV_DT", "INDEX_CLOSE"]].rename(columns={"INDEX_CLOSE": "AGGR_UNIT_NVAL"})
    idx_sharpe, idx_excess_mean, _, idx_sh_fact = _compute_sharpe_stats_basic_info_style(
        idx_nav_for_sharpe,
        day_type=day_type,
        trading_days_df=trading_days_df,
        ann_rf=risk_free_rate,
    )
    idx_sortino_numer = (
        idx_excess_mean * np.sqrt(idx_sh_fact)
        if pd.notna(idx_excess_mean) else np.nan
    )
    idx_downside_risk = _downside_risk_risk_control_index_only(
        idx_sub, day_type, trading_days_df
    )
    if pd.notna(idx_downside_risk) and idx_downside_risk > 0 and pd.notna(idx_sortino_numer):
        idx_sortino = float(idx_sortino_numer / idx_downside_risk)
    else:
        idx_sortino = np.nan

    idx_sub["累计净值"] = idx_sub["INDEX_CLOSE"]
    idx_sub["历史最高"] = idx_sub["累计净值"].cummax()
    idx_sub["回撤"] = (idx_sub["累计净值"] - idx_sub["历史最高"]) / idx_sub["历史最高"]
    idx_max_dd = abs(idx_sub["回撤"].min())
    idx_calmar = idx_ann_return / idx_max_dd if pd.notna(idx_max_dd) and idx_max_dd > 0 else np.nan

    return {
        "夏普比率": idx_sharpe,
        "索提诺比率": idx_sortino,
        "卡玛比率": idx_calmar,
    }


def enrich_long_with_benchmark_columns(
        df_long: pd.DataFrame,
        idx_df: pd.DataFrame,
        periods: list,
        risk_free_rate: float,
        day_type: str,
        established_dt_map: dict,
        trading_days_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    与「风险控制指标」build_benchmark_and_avg 一致：逐行补充同类平均、沪深300列（长表，不另增汇总行）。
    """
    out = df_long.copy()
    group_to_indices = out.groupby(["产品类型", "周期", "计算模式", "计算基准日"]).groups

    for m in RATIO_METRICS:
        out[f"同类平均{m}"] = np.nan

    for idx, row in out.iterrows():
        group_key = (row["产品类型"], row["周期"], row["计算模式"], row["计算基准日"])
        candidate_indices = list(group_to_indices.get(group_key, []))
        comparable_indices = _get_comparable_product_indices(
            out,
            row,
            established_dt_map=established_dt_map,
            candidate_indices=candidate_indices,
        )
        if len(comparable_indices) == 0:
            continue
        comparable_df = out.loc[comparable_indices]
        for metric in RATIO_METRICS:
            out.loc[idx, f"同类平均{metric}"] = comparable_df[metric].mean()

    hs300_map = {
        p: _compute_hs300_ratio_row_for_period(
            idx_df, p, risk_free_rate, day_type, trading_days_df
        )
        for p in periods
    }
    for m in RATIO_METRICS:
        out[f"沪深300{m}"] = out["周期"].map(
            lambda x, mm=m: hs300_map.get(x, {}).get(mm, np.nan)
        )

    return out


# --------------------------------------------------
# 8. 排名（长表：与原先按周期、产品类型分组逻辑一致）
# --------------------------------------------------
def rank_risk_adjusted_long(df: pd.DataFrame, established_dt_map: dict) -> pd.DataFrame:
    df = df.copy()
    for col in RATIO_METRICS:
        df[f"{col}排名"] = ""

    group_to_indices = df.groupby(["产品类型", "周期", "计算模式", "计算基准日"]).groups

    for idx, row in df.iterrows():
        prd_code = row["产品代码"]
        if pd.isna(_normalize_timestamp(established_dt_map.get(prd_code, pd.NaT))):
            continue
        if not _is_product_eligible_for_row(row, row, established_dt_map):
            continue

        group_key = (row["产品类型"], row["周期"], row["计算模式"], row["计算基准日"])
        candidate_indices = list(group_to_indices.get(group_key, []))
        comparable_products = _get_comparable_product_indices(
            df,
            row,
            established_dt_map=established_dt_map,
            candidate_indices=candidate_indices,
        )

        if len(comparable_products) == 0:
            continue

        comparable_df = df.loc[comparable_products]
        for metric in RATIO_METRICS:
            valid_mask = comparable_df[metric].notna()
            valid_df = comparable_df[valid_mask]
            if len(valid_df) == 0 or idx not in valid_df.index:
                continue
            rank_series = valid_df[metric].rank(method="min", ascending=False).astype(int)
            my_rank = rank_series[idx]
            total_count = len(valid_df)
            df.loc[idx, f"{metric}排名"] = f"{my_rank}/{total_count}"

    return df


# --------------------------------------------------
# 10. 列顺序（与「风险控制指标」长表一致：标识列 → 产品指标 → 同类平均 → 沪深300 → 排名）
# --------------------------------------------------
def order_long_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    _col_order = (
        ["产品代码", "产品类型", "计算模式", "周期", "计算基准日"]
        + RATIO_METRICS
        + [f"同类平均{m}" for m in RATIO_METRICS]
        + [f"沪深300{m}" for m in RATIO_METRICS]
        + [f"{m}排名" for m in RATIO_METRICS]
    )
    ordered = [c for c in _col_order if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


# --------------------------------------------------
# 11. 格式化输出（长表）
# --------------------------------------------------
def format_output(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    skip_num = {"产品代码", "产品类型", "计算模式", "周期"}

    for col in df_copy.columns:
        if col == "计算基准日":
            df_copy[col] = df_copy[col].apply(
                lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
            )
            continue
        if col.endswith("排名"):
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x}名" if pd.notna(x) and x != "" else ""
            )
            continue
        if col in skip_num:
            continue
        df_copy[col] = df_copy[col].apply(
            lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else ""
        )

    return df_copy


# --------------------------------------------------
# 12. 主流程
# --------------------------------------------------
def main(
        index_secu_id="000300.IDX.CSIDX",
        risk_free_rate: float = 0.015,
        day_type: str = "交易日",
):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]

    t0 = time.perf_counter()
    prd_df = fetch_fin_prd_nav()
    idx_df = fetch_index_quote(index_secu_id)
    idx_df = idx_df.sort_values("NAV_DT", ignore_index=True)
    product_base_info = fetch_pty_prd_base_info()

    prd_df = fill_special_prd_typ(prd_df, product_base_info)

    trading_days_df = fetch_trading_days() if day_type == "交易日" else None

    established_dt_map = (
        product_base_info.dropna(subset=["PRD_CODE"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")["FOUND_DT"]
        .apply(lambda x: pd.Timestamp(x).normalize() if pd.notna(x) else pd.NaT)
        .to_dict()
    )

    calc_base_by_code = prd_df.groupby("PRD_CODE")["NAV_DT"].max()

    grouped_products = list(prd_df.groupby("PRD_CODE"))
    max_workers = min(8, max(1, (os.cpu_count() or 4)))
    nav_days_df = calc_period_nav_days(prd_df, idx_df, periods, established_dt_map)
    max_days_df = (
        nav_days_df
        .groupby(["PRD_TYP", "周期"])["nav_days"]
        .max()
        .reset_index(name="max_nav_days")
    )
    nav_days_df = nav_days_df.merge(max_days_df, on=["PRD_TYP", "周期"])
    nav_days_df["eligible"] = nav_days_df["nav_days"] == nav_days_df["max_nav_days"]
    payloads = []
    for prd_code, g in grouped_products:
        one_base = product_base_info[product_base_info["PRD_CODE"] == prd_code]
        if len(one_base) == 0:
            one_base = None
        payloads.append(
            (
                g,
                idx_df,
                periods,
                one_base,
                risk_free_rate,
                day_type,
                trading_days_df,
            )
        )

    all_results = []
    if payloads:
        chunksize = max(1, len(payloads) // (max_workers * 4) or 1)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for part in executor.map(_calc_product_worker_pack, payloads, chunksize=chunksize):
                all_results.extend(part)

    df_long = pd.DataFrame(all_results)
    if df_long.empty:
        return df_long
    df_long["计算基准日"] = df_long["产品代码"].map(calc_base_by_code)
    df_long = enrich_long_with_benchmark_columns(
        df_long,
        idx_df,
        periods,
        risk_free_rate=risk_free_rate,
        day_type=day_type,
        established_dt_map=established_dt_map,
        trading_days_df=trading_days_df,
    )

    df_long = rank_risk_adjusted_long(df_long, established_dt_map)

    df_long = order_long_output_columns(df_long)

    df_formatted = format_output(df_long)

    print(f"[计时] 风险收益性价比 main() 耗时: {time.perf_counter() - t0:.2f} 秒", flush=True)
    return df_formatted


# --------------------------------------------------
# 13. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    mp.freeze_support()
    result = main(day_type="交易日")
    result.to_csv("风险收益性价比指标.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(result.head())
