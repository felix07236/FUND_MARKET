import os
import oracledb
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time
from concurrent.futures import ProcessPoolExecutor

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
def fetch_fin_prd_nav(
        nav_dt_from: str | None = None,
        nav_dt_to: str | None = None,
) -> pd.DataFrame:
    """全量拉取净值表；可选 NAV_DT 闭区间（YYYYMMDD 字符串），与交易日历对齐以缩小扫描。"""
    sql = """
          SELECT PRD_CODE, \
                 PRD_TYP, \
                 NAV_DT, \
                 UNIT_NVAL, \
                 AGGR_UNIT_NVAL, \
                 NAV_ADD_RAT, \
                 AGGR_NAV_ADD_RAT
          FROM DATA_MART_04.FIN_PRD_NAV
          WHERE AGGR_UNIT_NVAL IS NOT NULL \
          """
    extra = []
    if nav_dt_from:
        extra.append(f"NAV_DT >= '{nav_dt_from}'")
    if nav_dt_to:
        extra.append(f"NAV_DT <= '{nav_dt_to}'")
    if extra:
        sql += " AND " + " AND ".join(extra)
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, con=conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


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


def fetch_index_quote(
        index_secu_id: str,
        trd_dt_from: str | None = None,
        trd_dt_to: str | None = None,
) -> pd.DataFrame:
    conds = [f"SECU_ID = '{index_secu_id}'"]
    if trd_dt_from:
        conds.append(f"TRD_DT >= '{trd_dt_from}'")
    if trd_dt_to:
        conds.append(f"TRD_DT <= '{trd_dt_to}'")
    sql = f"""
    SELECT TRD_DT, CLS_PRC
    FROM VAR_SECU_DQUOT
    WHERE {" AND ".join(conds)}
    """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, con=conn)

    df["NAV_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
    return df.rename(columns={"CLS_PRC": "INDEX_CLOSE"})


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


# --------------------------------------------------
# 3. 工具函数
# --------------------------------------------------
def _normalize_period_key(period: str) -> str:
    """统一「近 1 年」「近1年」等写法。"""
    return period.replace(" ", "").strip()


def _vol_annual_factor(day_type: str) -> float:
    return np.sqrt(250) if day_type == "交易日" else np.sqrt(365)


def _filter_daily_ret_by_trading_days(ret: pd.Series, trading_days_df: pd.DataFrame) -> pd.Series:
    """仅保留落在交易日历上的收益观测"""
    td_idx = pd.to_datetime(
        trading_days_df.loc[trading_days_df["IS_TRD_DT"], "CALD_DATE"]
    ).dt.normalize().unique()
    ix = pd.DatetimeIndex(ret.index).normalize()
    return ret.loc[ix.isin(td_idx)]


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


def _empty_risk_metrics_result(period: str) -> dict:
    return {
        "周期": period,
        "最大回撤": np.nan,
        "贝塔": np.nan,
        "回撤修复": np.nan,
        "年化波动率": np.nan,
        "下行风险": np.nan,
        "防守能力": np.nan,
    }


def _max_drawdown_recovery_days(
        nav_dt: np.ndarray,
        navs: np.ndarray,
) -> float:
    """
    回撤修复口径与 Excel 公式一致：
    1) 先按 runningMax 计算全区间 drawdown；
    2) 仅取「最大回撤」所在谷点 trough；
    3) 从 trough 之后找首次恢复到 trough 对应峰值(peak)的日期；
    4) 返回 recovery_date - trough_date 的自然日天数；无恢复则 NaN。
    """
    if nav_dt is None or navs is None:
        return np.nan
    if len(nav_dt) < 2 or len(navs) < 2:
        return np.nan

    navs = np.asarray(navs, dtype=float)
    running_max = np.maximum.accumulate(navs)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = (running_max - navs) / running_max

    if not np.isfinite(drawdown).any():
        return np.nan
    max_dd = np.nanmax(drawdown)
    if (not np.isfinite(max_dd)) or max_dd <= 0:
        return np.nan

    trough_candidates = np.flatnonzero(np.isclose(drawdown, max_dd, rtol=1e-12, atol=1e-12))
    if len(trough_candidates) == 0:
        return np.nan
    trough_pos = int(trough_candidates[0])

    if trough_pos >= len(navs) - 1:
        return np.nan

    target_nav = running_max[trough_pos]
    tail = navs[trough_pos + 1:]
    recovery_candidates = np.flatnonzero(tail >= target_nav)
    if len(recovery_candidates) == 0:
        return np.nan

    recovery_pos = trough_pos + 1 + int(recovery_candidates[0])
    return float((pd.Timestamp(nav_dt[recovery_pos]) - pd.Timestamp(nav_dt[trough_pos])).days)


def _apply_day_type_to_fund_idx_returns(
        fund_daily_ret: pd.Series,
        idx_daily_ret: pd.Series,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> tuple:
    """
    交易日模式下仅保留 IS_TRD_DT 的日收益；自然日不做过滤。
    """
    if day_type != "交易日" or trading_days_df is None:
        return fund_daily_ret, idx_daily_ret
    f = _filter_daily_ret_by_trading_days(fund_daily_ret, trading_days_df)
    i = _filter_daily_ret_by_trading_days(idx_daily_ret, trading_days_df)
    return f, i


def max_drawdown(nav_series: pd.Series) -> float:
    if len(nav_series) <= 1:
        return 0.0
    nav_series = (
        nav_series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if nav_series.empty:
        return 0.0
    roll_max = nav_series.cummax()
    drawdown = 1 - nav_series / roll_max
    return -min(drawdown.max(), 1.0)


def get_fund_established_dt(prd_df: pd.DataFrame, df_base_info: pd.DataFrame = None) -> pd.Timestamp:
    if df_base_info is not None and len(df_base_info) > 0:
        if pd.notna(df_base_info.iloc[0]["FOUND_DT"]):
            return df_base_info.iloc[0]["FOUND_DT"]
    return prd_df["NAV_DT"].min()


def get_nav_start(
        prd_df: pd.DataFrame,
        theory_start_dt: pd.Timestamp,
        fund_established_dt: pd.Timestamp
) -> float:
    if theory_start_dt < fund_established_dt:
        return 1.0
    sub_start = prd_df[prd_df["NAV_DT"] == theory_start_dt]
    if len(sub_start) > 0:
        return sub_start.iloc[0]["UNIT_NVAL"]
    sub_after = prd_df[prd_df["NAV_DT"] >= theory_start_dt].sort_values("NAV_DT")
    if len(sub_after) > 0:
        return sub_after.iloc[0]["UNIT_NVAL"]
    return 1.0


def get_nav_end(prd_df: pd.DataFrame, end_dt: pd.Timestamp) -> float:
    sub_end = prd_df[prd_df["NAV_DT"] == end_dt]
    if len(sub_end) > 0:
        return sub_end.iloc[0]["UNIT_NVAL"]
    return prd_df.iloc[-1]["UNIT_NVAL"]


def get_period_dates_for_drawdown(
        end_dt: pd.Timestamp,
        period: str,
        market_df: pd.DataFrame
) -> tuple:
    """
    成立以来：起点为行情（指数）最早日期；今年以来：上年末；近 n 年/月：先偏移再「日减 1」规则。
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


def get_index_start_price(idx_df: pd.DataFrame, theory_start_dt: pd.Timestamp, price_col: str = "INDEX_CLOSE") -> float:
    """指数在理论起点当日或之后的首个收盘价。"""
    sub = idx_df[idx_df["NAV_DT"] == theory_start_dt]
    if len(sub) > 0:
        return float(sub.iloc[0][price_col])
    sub_after = idx_df[idx_df["NAV_DT"] >= theory_start_dt].sort_values("NAV_DT")
    if len(sub_after) > 0:
        return float(sub_after.iloc[0][price_col])
    return np.nan


# --------------------------------------------------
# 4. 风险控制指标计算
# --------------------------------------------------
def calc_risk_metrics(
        prd_df: pd.DataFrame,
        idx_df: pd.DataFrame,
        period: str,
        df_base_info: pd.DataFrame = None,
        day_type: str = "交易日",
        trading_days_df: pd.DataFrame = None,
) -> dict:
    prd_df = prd_df.sort_values("NAV_DT")
    base_dt = prd_df["NAV_DT"].max()
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)
    pnorm = _normalize_period_key(period)
    theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)
    if pnorm == "成立以来":
        if pd.isna(fund_established_dt):
            return _empty_risk_metrics_result(period)
        theory_start_dt = pd.Timestamp(fund_established_dt).normalize() - pd.Timedelta(days=1)
        period_end_dt = base_dt

    prd_df_clean = prd_df
    if pd.notna(fund_established_dt):
        prd_df_clean = prd_df[prd_df["NAV_DT"] >= fund_established_dt]
    if prd_df_clean.empty:
        return _empty_risk_metrics_result(period)

    actual_theoretical_start = _resolve_actual_theoretical_start(
        period, fund_established_dt, theory_start_dt, day_type, trading_days_df
    )
    if pnorm == "成立以来":
        # 成立以来固定使用「成立日前一天」作为虚拟起点，产品起始净值按 1 处理
        actual_theoretical_start = theory_start_dt
    
    if pnorm != "成立以来":
        ats = pd.Timestamp(actual_theoretical_start).normalize()
        if prd_df_clean[prd_df_clean["NAV_DT"].dt.normalize() == ats].empty:
            return _empty_risk_metrics_result(period)

    missing_check_start_dt = fund_established_dt if pnorm == "成立以来" else actual_theoretical_start
    has_missing = _has_missing_data_in_period(
            nav_df=prd_df_clean,
            start_dt=missing_check_start_dt,
            end_dt=period_end_dt,
            day_type=day_type,
            trading_days_df=trading_days_df,
    )
    if has_missing:
        return _empty_risk_metrics_result(period)

    nav_start = get_nav_start(prd_df, theory_start_dt, fund_established_dt)
    if pnorm == "成立以来":
        nav_start = 1.0
    nav_end = get_nav_end(prd_df, period_end_dt)

    actual_start_dt = max(theory_start_dt, fund_established_dt)
    sub = prd_df[
        (prd_df["NAV_DT"] >= actual_start_dt) &
        (prd_df["NAV_DT"] <= period_end_dt)
        ].copy()

    if len(sub) < 2:
        return _empty_risk_metrics_result(period)

    if pd.isna(nav_start) or pd.isna(nav_end) or nav_start <= 0:
        return _empty_risk_metrics_result(period)

    # 基金日收益只算一次（虚拟起始点与贝塔/波动率共用）
    fund_nav_series = sub.set_index("NAV_DT")["UNIT_NVAL"]
    has_virtual_start = (theory_start_dt < fund_established_dt) and (nav_start == 1.0)

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

    # ========== 最大回撤：区间累计净值（起点归一）上计算 ==========
    sub["累计收益"] = sub["UNIT_NVAL"] / nav_start
    max_dd_val = max_drawdown(sub["累计收益"])

    sub["历史最高"] = sub["累计收益"].cummax()
    sub["回撤"] = (sub["累计收益"] - sub["历史最高"]) / sub["历史最高"]

    # ==========  贝塔  ==========
    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= theory_start_dt) &
        (idx_df["NAV_DT"] <= base_dt)
        ].copy()

    beta = np.nan
    merged_valid = None
    ann_vol = np.nan
    downside_risk = np.nan
    ann_scale = _vol_annual_factor(day_type)

    if len(idx_sub) >= 2:
        idx_sub = idx_sub.sort_values("NAV_DT")
        idx_daily_ret = idx_sub.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()
        fund_daily_ret_m, idx_daily_ret_m = _apply_day_type_to_fund_idx_returns(
            fund_daily_ret, idx_daily_ret, day_type, trading_days_df
        )
        merged = pd.DataFrame({
            "日收益": fund_daily_ret_m,
            "指数日收益": idx_daily_ret_m,
        }).dropna()

        if len(merged) >= 10:
            cov_m = np.cov(merged["日收益"], merged["指数日收益"], ddof=1)
            var_idx = cov_m[1, 1]
            if var_idx == 0:
                beta = 0.0
            else:
                beta = cov_m[0, 1] / var_idx
            merged_valid = merged

        vol_src = merged["日收益"] if len(merged) >= 2 else fund_daily_ret
        if len(vol_src) >= 2:
            dv = vol_src.std()
            ann_vol = dv * ann_scale if pd.notna(dv) else np.nan
            neg = vol_src[vol_src < 0]
            if len(neg) >= 2:
                ds = neg.std()
                downside_risk = ds * ann_scale if pd.notna(ds) else np.nan

    # ==========  回撤修复时间  ==========
    nav_a = sub["NAV_DT"].to_numpy()
    cum_a = sub["累计收益"].to_numpy(dtype=float)
    recovery_time = _max_drawdown_recovery_days(nav_a, cum_a)

    # ========== 防守能力 ==========
    # 公式：|基金在所有市场下跌日的平均收益| / |市场平均跌幅|
    # 值越小表示防守能力越强（基金跌得比市场少）
    if merged_valid is None or len(merged_valid) < 10:
        defensive_ability = np.nan
    else:
        down_mask = merged_valid["指数日收益"].to_numpy(dtype=float) < 0
        if down_mask.sum() < 5:
            defensive_ability = np.nan
        else:
            avg_fund_return = np.nanmean(merged_valid["日收益"].to_numpy(dtype=float)[down_mask])
            avg_market_return = np.nanmean(merged_valid["指数日收益"].to_numpy(dtype=float)[down_mask])
    
            avg_fund_drop = abs(avg_fund_return)
            avg_market_drop = abs(avg_market_return)
    
            if avg_market_drop != 0:
                defensive_ability = avg_fund_drop / avg_market_drop
            else:
                defensive_ability = np.nan
    return {
        "周期": period,
        "最大回撤": max_dd_val,
        "贝塔": beta,
        "回撤修复": recovery_time,
        "年化波动率": ann_vol,
        "下行风险": downside_risk,
        "防守能力": defensive_ability
    }


# --------------------------------------------------
# 6. 单产品多周期指标计算
# --------------------------------------------------
def calc_product_risk_metrics(
        prd_df: pd.DataFrame,
        idx_df: pd.DataFrame,
        periods: list,
        df_base_info: pd.DataFrame = None,
        day_type: str = "交易日",
        trading_days_df: pd.DataFrame = None,
) -> list:
    prd_df = prd_df.sort_values("NAV_DT")
    base_dt = prd_df["NAV_DT"].max()
    fund_est = pd.NaT
    if df_base_info is not None and len(df_base_info) > 0:
        fd = df_base_info.iloc[0]["FOUND_DT"]
        if pd.notna(fd):
            fund_est = pd.Timestamp(fd).normalize()

    prd_typ = prd_df["PRD_TYP"].iloc[0]
    if pd.isna(prd_typ):
        prd_typ = "未分类"

    results = []

    for period in periods:
        cal_theory, _ = get_period_dates_for_drawdown(base_dt, period, idx_df)
        cal_theory = pd.Timestamp(cal_theory).normalize()
        if _normalize_period_key(period) == "成立以来" and pd.notna(fund_est):
            cal_theory = pd.Timestamp(fund_est).normalize() - pd.Timedelta(days=1)
        if _normalize_period_key(period) == "成立以来":
            admission_start = fund_est
        else:
            admission_start = cal_theory
        meets = pd.notna(fund_est) and pd.notna(admission_start) and fund_est <= admission_start

        metrics = calc_risk_metrics(
            prd_df, idx_df, period, df_base_info,
            day_type=day_type, trading_days_df=trading_days_df,
        )

        results.append({
            "产品代码": prd_df["PRD_CODE"].iloc[0],
            "产品类型": prd_typ,
            "计算模式": day_type,
            "计算基准日": pd.Timestamp(base_dt).normalize(),
            "成立日": fund_est,
            "理论起始日": cal_theory,
            "符合条件": meets,
            **metrics
        })

    return results


def _normalize_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    return pd.to_datetime(ts)


RISK_BENCHMARK_METRICS = [
    "最大回撤", "贝塔", "回撤修复", "年化波动率", "下行风险", "防守能力",
]


def _is_risk_comparable_for_benchmark(
        comp_row: pd.Series,
        target_row: pd.Series,
        established_dt_map: dict,
) -> bool:
    """
    与「收益能力指标」_is_product_eligible_for_row 一致：可比的成立日规则；
    本模块使用「理论起始日」、周期归一与「产品类型+周期+计算模式+计算基准日」已分组后的候选行。
    """
    comp_prd = comp_row["产品代码"]
    if established_dt_map is None or not established_dt_map:
        return False
    comp_est = established_dt_map.get(comp_prd, pd.NaT)
    if pd.isna(comp_est):
        return False
    comp_est = _normalize_timestamp(comp_est)
    period = target_row["周期"]
    pnorm = _normalize_period_key(period)
    if pnorm == "成立以来":
        target_prd = target_row["产品代码"]
        t_est = established_dt_map.get(target_prd, pd.NaT)
        t_est = _normalize_timestamp(t_est)
        if pd.isna(t_est):
            return False
        return comp_est <= t_est
    target_theory = _normalize_timestamp(target_row.get("理论起始日", pd.NaT))
    if pd.isna(target_theory):
        return False
    return comp_est <= target_theory


def _get_risk_comparable_for_benchmark(
        df: pd.DataFrame,
        target_row: pd.Series,
        established_dt_map: dict,
        candidate_indices: list,
) -> list:
    out = []
    for comp_idx in candidate_indices:
        comp_row = df.loc[comp_idx]
        if not comp_row.get("符合条件", True):
            continue
        if comp_row.get("产品类型") == "指数":
            continue
        if _is_risk_comparable_for_benchmark(comp_row, target_row, established_dt_map):
            out.append(comp_idx)
    return out


def _calc_hs300_risk_for_row_slice(
        idx_sub: pd.DataFrame,
        idx_start: float,
        day_type: str,
        trading_days_df: pd.DataFrame = None,
) -> dict:
    """
    与原先「沪深 300」基准行计算口径一致：区间自理论起点起归一的回撤、
    日收益按计算模式取年化波动率/下行风险；贝塔=1、防守=1、回撤修复需要计算。
    """
    if len(idx_sub) < 2:
        return {m: np.nan for m in RISK_BENCHMARK_METRICS}
    idx_sub = idx_sub.sort_values("NAV_DT").copy()
    idx_cum = (idx_sub["INDEX_CLOSE"] / idx_start).to_numpy(dtype=float)
    idx_max_dd = max_drawdown(pd.Series(idx_cum))

    nav_a = idx_sub["NAV_DT"].to_numpy()
    recovery_time = _max_drawdown_recovery_days(nav_a, idx_cum)

    idx_daily_ret = idx_sub.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()
    if day_type == "交易日" and trading_days_df is not None:
        idx_daily_ret = _filter_daily_ret_by_trading_days(idx_daily_ret, trading_days_df)
    ann_scale = _vol_annual_factor(day_type)
    idx_vol = idx_daily_ret.std() * ann_scale if len(idx_daily_ret) >= 2 else np.nan
    idx_neg = idx_daily_ret[idx_daily_ret < 0]
    idx_down_risk = idx_neg.std() * ann_scale if len(idx_neg) >= 2 else np.nan
    return {
        "最大回撤": idx_max_dd,
        "贝塔": 1.0,
        "回撤修复": recovery_time,
        "年化波动率": idx_vol,
        "下行风险": idx_down_risk,
        "防守能力": 1.0,
    }


# --------------------------------------------------
# 7. 同类平均 & 沪深 300 列（与「收益能力指标」build_benchmark_and_avg 一致：逐行补列，不另增汇总行）
# --------------------------------------------------
def build_benchmark_and_avg(
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        product_base_info: pd.DataFrame,
        trading_days_df: pd.DataFrame = None,
) -> pd.DataFrame:
    result = df.reset_index(drop=True)
    for c in RISK_BENCHMARK_METRICS:
        for prefix in ("同类平均", "沪深300"):
            result[f"{prefix}{c}"] = np.nan

    if product_base_info is not None and len(product_base_info) > 0:
        established_dt_map = (
            product_base_info.dropna(subset=["PRD_CODE"])
            .drop_duplicates(subset=["PRD_CODE"], keep="first")
            .set_index("PRD_CODE")["FOUND_DT"]
            .to_dict()
        )
    else:
        established_dt_map = {}

    group_to_indices = result.groupby(
        ["产品类型", "周期", "计算模式", "计算基准日"]
    ).groups

    row_cols = ["产品类型", "周期", "计算模式", "计算基准日"]
    for i in range(len(result)):
        row_idx = result.index[i]
        row = result.iloc[i]
        ptyp, period, day_type, base_dt_row = (row[c] for c in row_cols)
        gkey = (ptyp, period, day_type, base_dt_row)
        if ptyp is None or period is None or day_type is None or pd.isna(base_dt_row):
            continue
        candidate_indices = list(group_to_indices.get(gkey, []))
        comp_ix = _get_risk_comparable_for_benchmark(
            result, row, established_dt_map, candidate_indices
        )
        if len(comp_ix) > 0:
            comp_df = result.loc[comp_ix]
            for m in RISK_BENCHMARK_METRICS:
                coln = f"同类平均{m}"
                if m == "回撤修复":
                    result.at[row_idx, coln] = comp_df[m].median()
                else:
                    result.at[row_idx, coln] = comp_df[m].mean()

        if pd.isna(base_dt_row):
            continue
        base_dt_ts = pd.Timestamp(base_dt_row)
        theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt_ts, period, idx_df)
        if _normalize_period_key(str(period)) == "成立以来":
            fund_est_row = row.get("成立日", pd.NaT)
            if pd.notna(fund_est_row):
                theory_start_dt = pd.Timestamp(fund_est_row).normalize() - pd.Timedelta(days=1)
        idx_start = get_index_start_price(idx_df, theory_start_dt)
        if pd.isna(idx_start) or idx_start <= 0:
            continue
        idx_sub = idx_df[
            (idx_df["NAV_DT"] >= theory_start_dt) &
            (idx_df["NAV_DT"] <= period_end_dt)
        ].sort_values("NAV_DT")
        if len(idx_sub) < 2:
            continue
        td = trading_days_df if str(day_type) == "交易日" else None
        slice_metrics = _calc_hs300_risk_for_row_slice(
            idx_sub, float(idx_start), str(day_type), td
        )
        for m, val in slice_metrics.items():
            result.at[row_idx, f"沪深300{m}"] = val

    return result


# --------------------------------------------------
# 8. 排名
# --------------------------------------------------
def rank_risk_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "计算模式" not in df.columns:
        df["计算模式"] = "交易日"

    smaller_is_better = ["最大回撤", "贝塔", "回撤修复", "年化波动率", "下行风险", "防守能力"]
    larger_is_better = []

    for col in smaller_is_better + larger_is_better:
        df[f"{col}排名"] = np.nan
        df[f"{col}排名"] = df[f"{col}排名"].astype(object)

    need_cols = ["符合条件", "计算基准日", "成立日", "理论起始日"]
    if not all(c in df.columns for c in need_cols):
        return df

    benchmark_codes = {"同类平均", "沪深 300"}
    mask_special = df["产品代码"].isin(benchmark_codes)
    eligible_products = df[(~mask_special) & (df["符合条件"] == True)]

    if len(eligible_products) == 0:
        return df

    ep = eligible_products
    orig_ix = ep.index.to_numpy()
    typ = ep["产品类型"].to_numpy()
    per = ep["周期"].to_numpy()
    mode = ep["计算模式"].to_numpy()
    base_dt = pd.to_datetime(ep["计算基准日"], errors="coerce").dt.normalize().to_numpy()
    fund_est = pd.to_datetime(ep["成立日"], errors="coerce").to_numpy()
    theory = pd.to_datetime(ep["理论起始日"], errors="coerce").to_numpy()
    period_norm = np.array([_normalize_period_key(str(p)) for p in per])

    for ii in range(len(ep)):
        idx = orig_ix[ii]
        fe = fund_est[ii]
        ps = theory[ii]
        comparable_indices = []
        for jj in range(len(ep)):
            if typ[jj] != typ[ii]:
                continue
            if per[jj] != per[ii]:
                continue
            if mode[jj] != mode[ii]:
                continue
            if base_dt[jj] != base_dt[ii]:
                continue
            comp_est = fund_est[jj]
            if pd.isna(comp_est):
                continue
            if period_norm[ii] == "成立以来":
                if pd.notna(fe) and pd.Timestamp(comp_est).normalize() <= pd.Timestamp(fe).normalize():
                    comparable_indices.append(orig_ix[jj])
            else:
                if pd.isna(ps):
                    continue
                if pd.Timestamp(comp_est).normalize() <= pd.Timestamp(ps).normalize():
                    comparable_indices.append(orig_ix[jj])

        if not comparable_indices:
            continue

        comparable_df = ep.loc[comparable_indices]
        total_count = len(comparable_df)

        for col in smaller_is_better:
            rk = comparable_df[col].rank(method="min", ascending=True, na_option="keep")
            my = rk.loc[idx] if idx in rk.index else np.nan
            if pd.notna(my):
                df.at[idx, f"{col}排名"] = f"{int(my)}/{total_count}"

    return df


# --------------------------------------------------
# 9. 格式化输出
# --------------------------------------------------
def format_output(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for _c in ["计算基准日", "成立日", "理论起始日", "符合条件"]:
        if _c in df_copy.columns:
            df_copy = df_copy.drop(columns=[_c])

    # 百分比格式（与「收益能力指标」一致：无值处保持 NaN，最后统一为 --）
    pct_cols = ["最大回撤", "贝塔", "年化波动率", "下行风险", "防守能力"]
    for col in pct_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else x
            )

    for pfx in ("同类平均", "沪深300"):
        for col in RISK_BENCHMARK_METRICS:
            name = f"{pfx}{col}"
            if name not in df_copy.columns:
                continue
            if col in ("最大回撤", "贝塔", "年化波动率", "下行风险", "防守能力"):
                df_copy[name] = df_copy[name].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else x
                )
            elif col == "回撤修复":
                df_copy[name] = df_copy[name].apply(
                    lambda x: f"{int(x)}天" if pd.notna(x) else x
                )

    # 回撤修复时间为整数
    if "回撤修复" in df_copy.columns:
        df_copy["回撤修复"] = df_copy["回撤修复"].apply(
            lambda x: f"{int(x)}天" if pd.notna(x) else x
        )

    rank_cols = ["最大回撤排名", "贝塔排名", "回撤修复排名", "年化波动率排名", "下行风险排名", "防守能力排名"]
    for col in rank_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"{x}名" if pd.notna(x) and x != "" else x
            )

    return df_copy.where(pd.notna(df_copy), "--")



# --------------------------------------------------
# 10. 主流程
# --------------------------------------------------
def _build_prd_base_dict(df_base_info: pd.DataFrame) -> dict:
    """PRD_CODE -> 基础信息行 dict，O(1) 查找。"""
    if df_base_info is None or len(df_base_info) == 0:
        return {}
    return (
        df_base_info.dropna(subset=["PRD_CODE"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")
        .to_dict("index")
    )


def _base_df_for_product(base_by_code: dict, prd_code) -> pd.DataFrame | None:
    row = base_by_code.get(prd_code)
    if row is None:
        return None
    return pd.DataFrame([row])


class _RiskMpPoolCtx:
    idx_df = None
    periods = None


def _risk_mp_init(idx_df, periods):
    _RiskMpPoolCtx.idx_df = idx_df
    _RiskMpPoolCtx.periods = periods


def _risk_mp_one(task):
    g, base_df, day_type, td = task
    return calc_product_risk_metrics(
        g,
        _RiskMpPoolCtx.idx_df,
        _RiskMpPoolCtx.periods,
        base_df,
        day_type=day_type,
        trading_days_df=td,
    )


def main(index_secu_id="000300.IDX.CSIDX", day_type="自然日"):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]
    modes = ["自然日", "交易日"] if day_type is None else [day_type]

    # 全量 4 表各查一次；净值与指数按交易日历 NAV_DT/TRD_DT 闭区间过滤（与原先全表相比不丢日历覆盖内的数据）
    trading_days_cal = fetch_trading_days()
    cal = trading_days_cal["CALD_DATE"].dropna()
    nav_from = pd.Timestamp(cal.min()).strftime("%Y%m%d")
    nav_to = pd.Timestamp(cal.max()).strftime("%Y%m%d")

    prd_df = fetch_fin_prd_nav(nav_dt_from=nav_from, nav_dt_to=nav_to)
    idx_df = fetch_index_quote(index_secu_id, trd_dt_from=nav_from, trd_dt_to=nav_to)
    df_base_info = fetch_pty_prd_base_info()
    base_by_code = _build_prd_base_dict(df_base_info)
    prd_df = fill_special_prd_typ(prd_df, df_base_info)
    trading_days_df = trading_days_cal if "交易日" in modes else None

    grouped = list(prd_df.groupby("PRD_CODE", sort=False))
    all_results = []

    for m in modes:
        td_m = trading_days_df if m == "交易日" else None
        tasks = []
        for _, g in grouped:
            prd_code = g["PRD_CODE"].iloc[0]
            tasks.append((g, _base_df_for_product(base_by_code, prd_code), m, td_m))

        n_workers = min(os.cpu_count() or 4, len(tasks)) if tasks else 0
        if len(tasks) >= 6 and n_workers > 1:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_risk_mp_init,
                    initargs=(idx_df, periods),
            ) as pool:
                for chunk in pool.map(_risk_mp_one, tasks, chunksize=1):
                    all_results.extend(chunk)
        else:
            for g, base_df, m2, td_m2 in tasks:
                all_results.extend(
                    calc_product_risk_metrics(
                        g, idx_df, periods, base_df,
                        day_type=m2, trading_days_df=td_m2,
                    )
                )

    df = pd.DataFrame(all_results)

    # 为每行补充「同类平均*」「沪深300*」列（与「收益能力指标」一致，不另增同类/指数汇总行）
    df_rank = build_benchmark_and_avg(df, idx_df, df_base_info, trading_days_df)

    # 排名
    df_rank = rank_risk_df(df_rank)

    _col_order = (
        ["产品代码", "产品类型", "计算模式", "周期", "计算基准日", "成立日", "理论起始日", "符合条件"]
        + RISK_BENCHMARK_METRICS
        + [f"同类平均{m}" for m in RISK_BENCHMARK_METRICS]
        + [f"沪深300{m}" for m in RISK_BENCHMARK_METRICS]
        + [f"{c}排名" for c in RISK_BENCHMARK_METRICS]
    )
    df_rank = df_rank[[c for c in _col_order if c in df_rank.columns]]

    # 格式化输出
    df_formatted = format_output(df_rank)

    return df_formatted


# --------------------------------------------------
# 11. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    result = main()
    result.to_csv("风险控制指标.csv", index=False, encoding="utf-8-sig")
    print(result.head())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ 程序运行时间: {elapsed_time:.2f} 秒")
