import oracledb
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time

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
                 UNIT_NVAL, \
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
    prd_df = prd_df.sort_values("NAV_DT").copy()
    base_dt = prd_df["NAV_DT"].max()
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)

    prd_df_clean = prd_df
    if pd.notna(fund_established_dt):
        prd_df_clean = prd_df[prd_df["NAV_DT"] >= fund_established_dt].copy()
    if prd_df_clean.empty:
        return _empty_risk_metrics_result(period)

    actual_theoretical_start = _resolve_actual_theoretical_start(
        period, fund_established_dt, theory_start_dt, day_type, trading_days_df
    )
    pnorm = _normalize_period_key(period)
    if pnorm != "成立以来":
        ats = pd.Timestamp(actual_theoretical_start).normalize()
        if prd_df_clean[prd_df_clean["NAV_DT"].dt.normalize() == ats].empty:
            return _empty_risk_metrics_result(period)

    missing_check_start_dt = fund_established_dt if pnorm == "成立以来" else actual_theoretical_start
    if _has_missing_data_in_period(
            nav_df=prd_df_clean,
            start_dt=missing_check_start_dt,
            end_dt=period_end_dt,
            day_type=day_type,
            trading_days_df=trading_days_df,
    ):
        return _empty_risk_metrics_result(period)

    nav_start = get_nav_start(prd_df, theory_start_dt, fund_established_dt)
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

    # 区间内日收益率（与 UNIT_NVAL 区间序列一致）
    sub["日收益"] = sub["UNIT_NVAL"].pct_change()

    # ========== 最大回撤：区间累计净值（起点归一）上计算 ==========
    sub["累计收益"] = sub["UNIT_NVAL"] / nav_start
    max_dd_val = max_drawdown(sub["累计收益"])

    sub["历史最高"] = sub["累计收益"].cummax()
    sub["回撤"] = (sub["累计收益"] - sub["历史最高"]) / sub["历史最高"]

    # ==========  贝塔  ==========
    idx_sub = idx_df[
        (idx_df["NAV_DT"] >= sub["NAV_DT"].min()) &
        (idx_df["NAV_DT"] <= base_dt)
        ].copy()

    beta = np.nan
    merged_valid = None
    ann_vol = np.nan
    downside_risk = np.nan
    ann_scale = _vol_annual_factor(day_type)

    if len(idx_sub) >= 2:
        idx_sub = idx_sub.sort_values("NAV_DT")
        
        # 计算基金日收益率：处理虚拟起始点场景
        fund_nav_series = sub.set_index("NAV_DT")["UNIT_NVAL"]
        
        # 判断是否存在虚拟起始点（理论起始日早于成立日，且 nav_start=1.0）
        has_virtual_start = (theory_start_dt < fund_established_dt) and (nav_start == 1.0)
        
        if has_virtual_start and len(fund_nav_series) > 0:
            # 存在虚拟起始点：成立日收益率 = (成立日净值 - 1.0) / 1.0
            first_date = fund_nav_series.index[0]
            first_nav = fund_nav_series.iloc[0]
            first_return = (first_nav - 1.0) / 1.0
            
            # 后续日期正常计算 pct_change
            if len(fund_nav_series) > 1:
                subsequent_returns = fund_nav_series.pct_change().iloc[1:]
                fund_daily_ret = pd.concat([
                    pd.Series([first_return], index=[first_date]),
                    subsequent_returns
                ]).dropna()
            else:
                # 只有一天数据
                fund_daily_ret = pd.Series([first_return], index=[first_date])
        else:
            # 正常情况：直接使用 pct_change
            fund_daily_ret = fund_nav_series.pct_change().dropna()
        
        idx_daily_ret = idx_sub.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()
        fund_daily_ret, idx_daily_ret = _apply_day_type_to_fund_idx_returns(
            fund_daily_ret, idx_daily_ret, day_type, trading_days_df
        )
        merged = pd.DataFrame({
            "日收益": fund_daily_ret,
            "指数日收益": idx_daily_ret,
        }).dropna()

        if len(merged) >= 10:
            cov_m = np.cov(merged["日收益"], merged["指数日收益"], ddof=1)
            var_idx = cov_m[1, 1]
            if var_idx == 0:
                beta = 0.0
            else:
                beta = cov_m[0, 1] / var_idx
            merged_valid = merged

        # 年化波动率、下行风险：优先用对齐后的序列；样本过少则用基金单边日收益
        vol_src = merged["日收益"] if len(merged) >= 2 else fund_daily_ret
        if len(vol_src) >= 2:
            dv = vol_src.std()
            ann_vol = dv * ann_scale if pd.notna(dv) else np.nan
            neg = vol_src[vol_src < 0]
            if len(neg) >= 2:
                ds = neg.std()
                downside_risk = ds * ann_scale if pd.notna(ds) else np.nan

    # ==========  回撤修复时间  ==========
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
            recovered = future[future["累计收益"] >= row["历史最高"]]
            if len(recovered) > 0:
                days = (recovered.iloc[0]["NAV_DT"] - row["NAV_DT"]).days
                recovery_days.append(days)

        # 取平均修复时间
        recovery_time = np.mean(recovery_days) if recovery_days else np.nan

    # ========== 防守能力  ==========
    if merged_valid is None or len(merged_valid) < 10:
        defensive_ability = np.nan
    else:
        # 筛选市场下跌的交易日
        down_days = merged_valid[merged_valid["指数日收益"] < 0].copy()

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
    prd_df = prd_df.sort_values("NAV_DT").copy()
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
    idx_cum = idx_sub["INDEX_CLOSE"] / idx_start
    idx_max_dd = max_drawdown(idx_cum)
    
    # ========== 计算回撤修复时间 ==========
    idx_sub["历史最高"] = idx_cum.cummax()
    idx_sub["回撤"] = (idx_cum - idx_sub["历史最高"]) / idx_sub["历史最高"]
    
    significant_drawdowns = idx_sub[idx_sub["回撤"] < -0.01].copy()
    
    if len(significant_drawdowns) == 0:
        recovery_time = np.nan
    else:
        recovery_days = []
        for idx_pos, row in significant_drawdowns.iterrows():
            future = idx_sub[idx_sub["NAV_DT"] >= row["NAV_DT"]].copy()
            recovered = future[future["INDEX_CLOSE"] / idx_start >= row["历史最高"]]
            if len(recovered) > 0:
                days = (recovered.iloc[0]["NAV_DT"] - row["NAV_DT"]).days
                recovery_days.append(days)
        recovery_time = np.mean(recovery_days) if recovery_days else np.nan
    
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

    for row_idx, row in result.iterrows():
        gkey = (
            row.get("产品类型"),
            row.get("周期"),
            row.get("计算模式"),
            row.get("计算基准日"),
        )
        if gkey[0] is None or gkey[1] is None or gkey[2] is None or pd.isna(gkey[3]):
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

        base_dt = row.get("计算基准日")
        period = row.get("周期")
        day_type = row.get("计算模式", "交易日")
        if pd.isna(base_dt):
            continue
        base_dt = pd.Timestamp(base_dt)
        theory_start_dt, period_end_dt = get_period_dates_for_drawdown(base_dt, period, idx_df)
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

    smaller_is_better = ["最大回撤", "贝塔", "回撤修复", "年化波动率", "下行风险"]
    larger_is_better = ["防守能力"]

    for col in smaller_is_better + larger_is_better:
        df[f"{col}排名"] = np.nan
        df[f"{col}排名"] = df[f"{col}排名"].astype(object)

    need_cols = ["符合条件", "计算基准日", "成立日", "理论起始日"]
    if not all(c in df.columns for c in need_cols):
        return df

    benchmark_codes = {"同类平均", "沪深 300"}
    mask_special = df["产品代码"].isin(benchmark_codes)
    eligible_products = df[(~mask_special) & (df["符合条件"] == True)].copy()

    if len(eligible_products) == 0:
        return df

    for idx, row in eligible_products.iterrows():
        fund_est = row["成立日"]
        comparable_indices = []
        for comp_idx, comp_row in eligible_products.iterrows():
            if comp_row["产品类型"] != row["产品类型"]:
                continue
            if comp_row["周期"] != row["周期"]:
                continue
            if comp_row["计算模式"] != row["计算模式"]:
                continue
            if pd.Timestamp(comp_row["计算基准日"]).normalize() != pd.Timestamp(row["计算基准日"]).normalize():
                continue
            comp_est = comp_row["成立日"]
            if pd.isna(comp_est):
                continue
            if _normalize_period_key(row["周期"]) == "成立以来":
                if pd.notna(fund_est) and pd.Timestamp(comp_est).normalize() <= pd.Timestamp(fund_est).normalize():
                    comparable_indices.append(comp_idx)
            else:
                ps = row["理论起始日"]
                if pd.isna(ps):
                    continue
                if pd.Timestamp(comp_est).normalize() <= pd.Timestamp(ps).normalize():
                    comparable_indices.append(comp_idx)

        if not comparable_indices:
            continue

        comparable_df = eligible_products.loc[comparable_indices]
        total_count = len(comparable_df)

        for col in smaller_is_better:
            rk = comparable_df[col].rank(method="min", ascending=True, na_option="keep")
            my = rk.loc[idx] if idx in rk.index else np.nan
            if pd.notna(my):
                df.at[idx, f"{col}排名"] = f"{int(my)}/{total_count}"

        col = "防守能力"
        rk = comparable_df[col].rank(method="min", ascending=False, na_option="keep")
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
def main(index_secu_id="000300.IDX.CSIDX", day_type="交易日"):
    periods = ["近 1 年", "近 2 年", "近 3 年", "近 5 年", "今年以来", "成立以来"]
    modes = ["自然日", "交易日"] if day_type is None else [day_type]

    # 获取数据
    prd_df = fetch_fin_prd_nav()
    idx_df = fetch_index_quote(index_secu_id)
    df_base_info = fetch_pty_prd_base_info()
    prd_df = fill_special_prd_typ(prd_df, df_base_info)
    trading_days_df = fetch_trading_days() if "交易日" in modes else None

    # 计算所有产品的风险指标
    all_results = []
    for m in modes:
        td_m = trading_days_df if m == "交易日" else None
        for _, g in prd_df.groupby("PRD_CODE"):
            prd_code = g["PRD_CODE"].iloc[0]
            base_rows = df_base_info[df_base_info["PRD_CODE"] == prd_code]
            results = calc_product_risk_metrics(
                g, idx_df, periods,
                base_rows if len(base_rows) else None,
                day_type=m,
                trading_days_df=td_m,
            )
            all_results.extend(results)

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
