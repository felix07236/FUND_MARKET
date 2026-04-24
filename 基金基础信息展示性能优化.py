import oracledb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time
from functools import lru_cache

# -----------------------------
# 1. 数据库连接
# -----------------------------
def get_oracle_conn(schema_type: str = "original"):
    """
    获取Oracle数据库连接
    :param schema_type: 'original' - 原数据库(交易日历表), 'new' - 新数据库(产品净值表和基础信息表)
    """
    dsn = oracledb.makedsn(
        host="10.150.8.30",
        port=1521,
        service_name="DEV"
    )
    return oracledb.connect(
        user="CMSINDICATORS_GSAB",
        password="CMSINDICATORS_GSAB",
        dsn=dsn
    )


def read_sql_with_fallback(conn, table_candidates: list[str], select_cols: str, where_clause: str = "") -> pd.DataFrame:
    """
    依次尝试不同表名（含/不含 schema）以适配不同库环境，避免 ORA-00942 直接中断。
    """
    last_error = None
    where_sql = f" {where_clause.strip()}" if where_clause else ""

    for table_name in table_candidates:
        sql = f"SELECT {select_cols} FROM {table_name}{where_sql}"
        try:
            return pd.read_sql(sql, conn)
        except oracledb.DatabaseError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"所有候选表都不可访问: {table_candidates}，请确认当前账号下的表名/同义词。"
    ) from last_error


# -----------------------------
# 交易日历：进程内只拉取并排序一次（可显式传入覆盖）
# -----------------------------
@lru_cache(maxsize=1)
def get_trading_dates_sorted_np() -> tuple[np.ndarray,]:
    """
    返回 (sorted_trading_dates_np,) 元组以便 lru_cache（数组不可哈希，用单元素元组包装）。
    空日历返回空数组。
    """
    trading_days_df = fetch_trading_days()
    trading_dates = trading_days_df.loc[trading_days_df["IS_TRD_DT"] == True, "CALD_DATE"]
    if len(trading_dates) == 0:
        return (np.array([], dtype="datetime64[ns]"),)
    arr = np.array(
        [np.datetime64(pd.Timestamp(x).to_datetime64()) for x in trading_dates],
        dtype="datetime64[ns]",
    )
    arr.sort()
    return (arr,)


def get_cached_trading_dates_np() -> np.ndarray | None:
    """交易日序列（datetime64[ns]），无数据时返回 None；全进程只从库拉取一次。"""
    arr = get_trading_dates_sorted_np()[0]
    return arr if len(arr) else None


def _isin_sorted_trading_days(
    sorted_cal: np.ndarray, dates_ns: np.ndarray
) -> np.ndarray:
    """sorted_cal 升序时，判断 dates_ns 各点是否为交易日（O(n log m)，优于 np.isin）。"""
    if sorted_cal is None or len(sorted_cal) == 0:
        return np.zeros(len(dates_ns), dtype=bool)
    idx = np.searchsorted(sorted_cal, dates_ns, side="left")
    n = len(sorted_cal)
    valid = idx < n
    out = np.zeros(len(dates_ns), dtype=bool)
    out[valid] = sorted_cal[idx[valid]] == dates_ns[valid]
    return out


def vectorized_holiday_days(
    dates_ns: np.ndarray, trading_dates_sorted_np: np.ndarray
) -> np.ndarray:
    """与 calculate_holiday_days 一致：每个自然日到「下一交易日」间隔天数，下限为 1。"""
    if trading_dates_sorted_np is None or len(trading_dates_sorted_np) == 0:
        return np.ones(len(dates_ns), dtype=np.int64)
    idx = np.searchsorted(trading_dates_sorted_np, dates_ns, side="right")
    out = np.ones(len(dates_ns), dtype=np.int64)
    valid = idx < len(trading_dates_sorted_np)
    if not np.any(valid):
        return out
    next_trd = trading_dates_sorted_np[idx[valid]]
    delta = (next_trd - dates_ns[valid]).astype("timedelta64[D]").astype(np.int64)
    out[valid] = np.maximum(delta, 1)
    return out


def max_drawdown_by_prd_code(
    df: pd.DataFrame, established_dt_map: dict
) -> dict:
    """按 PRD_CODE groupby 向量化：成立日后单位净值序列上的历史最大回撤（与逐产品 cummax 一致）。"""
    need = df[["PRD_CODE", "NAV_DT", "UNIT_NVAL"]].assign(
        FOUND_DT=df["PRD_CODE"].map(established_dt_map)
    )
    need = need.dropna(subset=["FOUND_DT"])
    need = need[need["NAV_DT"] >= need["FOUND_DT"]]
    if need.empty:
        return {}
    need = need.sort_values(["PRD_CODE", "NAV_DT"], kind="mergesort")
    g = need.groupby("PRD_CODE", sort=False)["UNIT_NVAL"]
    cummax_u = g.cummax()
    dd = (need["UNIT_NVAL"] - cummax_u) / cummax_u
    return dd.groupby(need["PRD_CODE"]).min().to_dict()


# -----------------------------
# 2. 获取数据
# -----------------------------

def fetch_trading_days() -> pd.DataFrame:
    """获取交易日数据"""
    conn = get_oracle_conn("original")
    try:
        df = read_sql_with_fallback(
            conn=conn,
            table_candidates=["DATA_MART_04.TRD_CALD_DTL", "TRD_CALD_DTL"],
            select_cols="CALD_DATE, IS_TRD_DT",
            where_clause="WHERE CALD_ID = '1'",
        )
        # 转换CALD_DATE为日期类型
        df["CALD_DATE"] = pd.to_datetime(df["CALD_DATE"], format="%Y%m%d", errors="coerce")
        # 转换IS_TRD_DT为布尔类型
        df["IS_TRD_DT"] = df["IS_TRD_DT"].astype(int) == 1
        return df
    finally:
        conn.close()


def fetch_fin_prd_nav(limit: int = None) -> pd.DataFrame:
    conn = get_oracle_conn("new")
    try:
        where_clause = f"WHERE ROWNUM <= {limit}" if limit is not None else ""
        df = read_sql_with_fallback(
            conn=conn,
            table_candidates=["CMSINDICATORS_GSAB.FIN_PRD_NAV", "FIN_PRD_NAV"],
            select_cols="PRD_TYP, UNIT_NVAL, AGGR_UNIT_NVAL, NAV_DT, PRD_CODE",
            where_clause=where_clause,
        )
        return df
    finally:
        conn.close()


# 按 limit 缓存的「已解析净值」内存表，避免同进程内重复查库
_FIN_PRD_NAV_MEMORY: dict[int | None, pd.DataFrame] = {}


def clear_fin_nav_preload() -> None:
    """清空净值内存缓存（换库、换环境或需强制重拉时调用）。"""
    _FIN_PRD_NAV_MEMORY.clear()


def preload_fin_prd_nav(
    limit: int | None = None, *, force_refresh: bool = False
) -> pd.DataFrame:
    """
    预加载产品净值到内存：一次读取 FIN_PRD_NAV，并在内存中完成 NAV_DT、
    单位/累计净值列的类型解析；后续计算只使用该 DataFrame，不再访问净值表。

    :param limit: 与 fetch_fin_prd_nav 一致；None 表示全表。
    :param force_refresh: 为 True 时忽略缓存重新拉取并覆盖缓存。
    """
    if not force_refresh and limit in _FIN_PRD_NAV_MEMORY:
        return _FIN_PRD_NAV_MEMORY[limit]

    raw = fetch_fin_prd_nav(limit=limit)
    df = raw.copy()
    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"], format="%Y%m%d", errors="coerce")
    df["UNIT_NVAL"] = pd.to_numeric(df["UNIT_NVAL"], errors="coerce")
    df["AGGR_UNIT_NVAL"] = pd.to_numeric(df["AGGR_UNIT_NVAL"], errors="coerce")
    _FIN_PRD_NAV_MEMORY[limit] = df
    return df


def fetch_pty_prd_base_info() -> pd.DataFrame:
    """获取产品基础信息表"""
    conn = get_oracle_conn("new")
    try:
        df = read_sql_with_fallback(
            conn=conn,
            table_candidates=["CMSINDICATORS_GSAB.PTY_PRD_BASE_INFO", "PTY_PRD_BASE_INFO"],
            select_cols="PRD_CODE, FOUND_DT, PRD_NAME, PRD_FULL_NAME, PRD_TYP",
        )
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
        trading_days_df: pd.DataFrame = None,
        trading_dates_sorted_np: np.ndarray = None
) -> tuple:
    # 初始化实际期末日
    actual_end_dt = current_dt

    # 计算持有天数和年化天数
    if day_type == "交易日":
        if trading_dates_sorted_np is not None and len(trading_dates_sorted_np) > 0:
            v = np.datetime64(pd.Timestamp(virtual_start_dt).to_datetime64())
            c = np.datetime64(pd.Timestamp(current_dt).to_datetime64())

            i_left = np.searchsorted(trading_dates_sorted_np, v, side="right")
            i_right = np.searchsorted(trading_dates_sorted_np, c, side="right")
            days_since = int(i_right - i_left)

            j = np.searchsorted(trading_dates_sorted_np, c, side="left")
            c_is_trd = j < len(trading_dates_sorted_np) and trading_dates_sorted_np[j] == c
            if not c_is_trd and i_right > i_left:
                actual_end_dt = pd.Timestamp(trading_dates_sorted_np[i_right - 1])
        elif trading_days_df is not None:
            # 兼容旧调用方式
            trading_days = trading_days_df[
                (trading_days_df["CALD_DATE"] > virtual_start_dt) &
                (trading_days_df["CALD_DATE"] <= current_dt) &
                (trading_days_df["IS_TRD_DT"] == True)
            ]
            days_since = len(trading_days)
        else:
            days_since = (current_dt - virtual_start_dt).days
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
        trading_days_df: pd.DataFrame = None,
        trading_dates_sorted_np: np.ndarray = None
) -> int:

    # 确保current_date是Timestamp类型
    if not isinstance(current_date, pd.Timestamp):
        current_date = pd.Timestamp(current_date)

    if trading_dates_sorted_np is not None and len(trading_dates_sorted_np) > 0:
        current_np = np.datetime64(current_date.to_datetime64())
        next_idx = np.searchsorted(trading_dates_sorted_np, current_np, side="right")
        if next_idx >= len(trading_dates_sorted_np):
            return 1
        next_trading_date = pd.Timestamp(trading_dates_sorted_np[next_idx])
        return max((next_trading_date - current_date).days, 1)

    if trading_days_df is None:
        return 1

    # 兼容旧调用方式
    trading_dates = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
    next_trading_date = trading_dates[trading_dates > current_date]
    if len(next_trading_date) == 0:
        return 1

    next_trading_date = next_trading_date.iloc[0]

    days_diff = (next_trading_date - current_date).days
    return max(days_diff, 1)


# 子进程只读共享（initializer 注入，避免每个任务重复 pickle）
_WORKER_TD_NP: np.ndarray | None = None
_WORKER_YLY_RANK: dict = {}


def _init_worker_shared(trading_np: np.ndarray | None, yly_rank: dict) -> None:
    global _WORKER_TD_NP, _WORKER_YLY_RANK
    _WORKER_TD_NP = trading_np
    _WORKER_YLY_RANK = yly_rank


def _calc_product_metrics_worker(args: tuple) -> dict | None:
    trading_dates_sorted_np = _WORKER_TD_NP
    yly_rank_dict = _WORKER_YLY_RANK

    (
        prd_code,
        nav_dt_ns,
        unit_nval,
        aggr_unit_nval,
        prd_typ_scalar,
        fund_est_ns,
        annual_method,
        day_type,
        pre_max_drawdown,
    ) = args

    est = fund_est_ns if isinstance(fund_est_ns, np.datetime64) else np.datetime64(
        pd.Timestamp(fund_est_ns).to_datetime64()
    )
    mask = nav_dt_ns >= est
    if not np.any(mask):
        return None

    nav_dt_ns = nav_dt_ns[mask]
    unit_nval = unit_nval[mask]
    aggr_unit_nval = aggr_unit_nval[mask]
    order = np.argsort(nav_dt_ns, kind="mergesort")
    nav_dt_ns = nav_dt_ns[order]
    unit_nval = unit_nval[order]
    aggr_unit_nval = aggr_unit_nval[order]

    group_clean = pd.DataFrame(
        {
            "NAV_DT": pd.to_datetime(nav_dt_ns),
            "UNIT_NVAL": unit_nval,
            "AGGR_UNIT_NVAL": aggr_unit_nval,
            "PRD_TYP": prd_typ_scalar,
        }
    )
    if group_clean.empty:
        return None

    group_clean = group_clean.reset_index(drop=True)

    latest = group_clean.iloc[-1]
    current_dt = group_clean["NAV_DT"].max()
    prd_typ = prd_typ_scalar

    ytd_return = calc_ytd_return(group_clean, current_dt)

    establish_date = pd.Timestamp(est)
    virtual_start_dt = establish_date - pd.Timedelta(days=1)
    start_aggr = 1.0

    since_return, annualized, actual_end_dt = calc_established_annualized(
        virtual_start_dt=virtual_start_dt,
        current_dt=current_dt,
        end_aggr=latest["AGGR_UNIT_NVAL"],
        start_aggr=start_aggr,
        method=annual_method,
        day_type=day_type,
        trading_dates_sorted_np=trading_dates_sorted_np,
    )

    if actual_end_dt != current_dt:
        end_ns = np.datetime64(pd.Timestamp(actual_end_dt).to_datetime64())
        pos = np.searchsorted(nav_dt_ns, end_ns, side="left")
        if pos < len(nav_dt_ns) and nav_dt_ns[pos] == end_ns:
            end_aggr = float(aggr_unit_nval[pos])
            since_return, annualized, _ = calc_established_annualized(
                virtual_start_dt=virtual_start_dt,
                current_dt=actual_end_dt,
                end_aggr=end_aggr,
                start_aggr=start_aggr,
                method=annual_method,
                day_type=day_type,
                trading_dates_sorted_np=trading_dates_sorted_np,
            )

    max_drawdown = pre_max_drawdown

    sharpe_annual_factor = 250 if day_type == "交易日" else 365
    virtual_start_nav = pd.DataFrame({
        "NAV_DT": [virtual_start_dt],
        "AGGR_UNIT_NVAL": [1.0]
    })
    full_nav_series = pd.concat([virtual_start_nav, group_clean[["NAV_DT", "AGGR_UNIT_NVAL"]]], ignore_index=True)
    full_nav_series = full_nav_series.sort_values("NAV_DT").reset_index(drop=True)

    if day_type == "交易日" and trading_dates_sorted_np is not None and len(trading_dates_sorted_np) > 0:
        nav_dates_ns = full_nav_series["NAV_DT"].to_numpy(dtype="datetime64[ns]")
        virtual_start_ns = np.datetime64(pd.Timestamp(virtual_start_dt).to_datetime64())
        is_trading_day = _isin_sorted_trading_days(trading_dates_sorted_np, nav_dates_ns)
        is_virtual_start = nav_dates_ns == virtual_start_ns
        mask = is_trading_day | is_virtual_start
        full_nav_series = full_nav_series[mask].reset_index(drop=True)

    daily_ret = full_nav_series["AGGR_UNIT_NVAL"].pct_change().dropna()
    if len(daily_ret) > 0:
        # ddof=1 与 pandas 默认一致，保证夏普分母与原先相同
        daily_ret_std = daily_ret.std(ddof=1)
        ann_rf = 0.015

        if day_type == "自然日":
            daily_rf = ann_rf / 365
            daily_excess_ret_mean = (daily_ret - daily_rf).mean()
        else:
            nav_for_rf = full_nav_series["NAV_DT"].to_numpy(dtype="datetime64[ns]")
            idx_arr = daily_ret.index.to_numpy()
            nav_ns_aligned = nav_for_rf[idx_arr]
            holiday_days_arr = vectorized_holiday_days(
                nav_ns_aligned, trading_dates_sorted_np
            )
            daily_rf_series = pd.Series(
                (ann_rf / 365) * holiday_days_arr, index=daily_ret.index
            )
            daily_excess_ret_mean = (daily_ret - daily_rf_series).mean()

        sharpe = (daily_excess_ret_mean / daily_ret_std) * np.sqrt(sharpe_annual_factor) if daily_ret_std != 0 else np.nan
    else:
        sharpe = np.nan

    start_dt = current_dt.replace(year=current_dt.year - 1) - timedelta(days=1)
    nav_ns_gc = group_clean["NAV_DT"].to_numpy(dtype="datetime64[ns]")
    aggr_gc = group_clean["AGGR_UNIT_NVAL"].to_numpy(dtype=np.float64, copy=False)
    start_ns_y = np.datetime64(pd.Timestamp(start_dt).to_datetime64())
    end_ns_y = np.datetime64(pd.Timestamp(current_dt).to_datetime64())
    lo = int(np.searchsorted(nav_ns_gc, start_ns_y, side="left"))
    hi = int(np.searchsorted(nav_ns_gc, end_ns_y, side="right"))
    yly_return = np.nan
    if hi - lo >= 2:
        yly_return = (aggr_gc[hi - 1] - aggr_gc[lo]) / aggr_gc[lo]

    rank_key = (prd_typ, prd_code)
    rank_info = yly_rank_dict.get(rank_key)
    yly_rank = f"{rank_info[0]}/{rank_info[1]}名" if rank_info else "N/A"

    return {
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
    }


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

    # 一次性筛选出需要的数据
    mask_prev = grp["NAV_DT"] <= prev_year_end
    mask_curr = (grp["NAV_DT"] >= year_start) & (grp["NAV_DT"] <= base_dt)
    
    last_day_prev_year = grp.loc[mask_prev, "NAV_DT"].max()

    if pd.isna(last_day_prev_year):
        first_day = grp.loc[mask_curr, "NAV_DT"].min()

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
        day_type: str = "自然日",
        trading_calendar_np: np.ndarray | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["NAV_DT"]):
        df["NAV_DT"] = pd.to_datetime(df["NAV_DT"], format="%Y%m%d", errors="coerce")
    if "UNIT_NVAL" in df.columns and not pd.api.types.is_numeric_dtype(df["UNIT_NVAL"]):
        df["UNIT_NVAL"] = pd.to_numeric(df["UNIT_NVAL"], errors="coerce")
    if "AGGR_UNIT_NVAL" in df.columns and not pd.api.types.is_numeric_dtype(
        df["AGGR_UNIT_NVAL"]
    ):
        df["AGGR_UNIT_NVAL"] = pd.to_numeric(df["AGGR_UNIT_NVAL"], errors="coerce")

    trading_dates_sorted_np = None
    if day_type == "交易日":
        trading_dates_sorted_np = (
            trading_calendar_np
            if trading_calendar_np is not None
            else get_cached_trading_dates_np()
        )

    # 向量化填充同一产品下唯一的产品类型，避免循环筛选
    prd_typ_map = df[df["PRD_TYP"].notna()].groupby("PRD_CODE")["PRD_TYP"].first()
    df["PRD_TYP"] = df["PRD_TYP"].fillna(df["PRD_CODE"].map(prd_typ_map))

    established_dt_map = (
        df_base_info.dropna(subset=["PRD_CODE", "FOUND_DT"])
        .drop_duplicates(subset=["PRD_CODE"], keep="first")
        .set_index("PRD_CODE")["FOUND_DT"]
        .to_dict()
    )

    max_dd_by_prd = max_drawdown_by_prd_code(df, established_dt_map)

    grouped_products = list(df.groupby("PRD_CODE", sort=False))

    product_data: list[dict] = []
    payloads: list[tuple] = []

    for prd_code, group in grouped_products:
        fund_established_dt = established_dt_map.get(prd_code, pd.NaT)
        if pd.isna(fund_established_dt):
            continue

        pt_raw = group["PRD_TYP"].dropna().unique()
        prd_typ_for_rank = pt_raw[0] if len(pt_raw) > 0 else np.nan

        g = group.sort_values("NAV_DT", kind="mergesort")
        # 与子进程排名字典键一致：与原「近一年」循环相同，取 dropna().unique()[0]
        prd_typ_scalar = prd_typ_for_rank

        base_dt = g["NAV_DT"].max()
        start_dt_yly = base_dt.replace(year=base_dt.year - 1) - timedelta(days=1)
        if fund_established_dt <= start_dt_yly:
            sub_mask = (g["NAV_DT"] >= start_dt_yly) & (g["NAV_DT"] <= base_dt)
            sub = g[sub_mask]
            if len(sub) < 2:
                ret = np.nan
            else:
                end_val = float(sub.iloc[-1]["AGGR_UNIT_NVAL"])
                start_val = float(sub.iloc[0]["AGGR_UNIT_NVAL"])
                ret = (end_val - start_val) / start_val
            if len(pt_raw) > 0:
                product_data.append(
                    {
                        "PRD_TYP": prd_typ_for_rank,
                        "PRD_CODE": prd_code,
                        "base_dt": base_dt,
                        "RET": ret,
                    }
                )

        nav_dt_ns = g["NAV_DT"].to_numpy(dtype="datetime64[ns]")
        unit_nval = g["UNIT_NVAL"].to_numpy(dtype=np.float64, copy=False)
        aggr_unit_nval = g["AGGR_UNIT_NVAL"].to_numpy(dtype=np.float64, copy=False)
        pre_mdd = max_dd_by_prd.get(prd_code, np.nan)
        fund_est_ns = np.datetime64(pd.Timestamp(fund_established_dt).to_datetime64())
        payloads.append(
            (
                prd_code,
                nav_dt_ns,
                unit_nval,
                aggr_unit_nval,
                prd_typ_scalar,
                fund_est_ns,
                annual_method,
                day_type,
                pre_mdd,
            )
        )

    yly_rank_dict: dict = {}
    if product_data:
        rank_df = pd.DataFrame(product_data)
        rank_df["RANK"] = rank_df.groupby(
            ["PRD_TYP", "base_dt"], sort=False
        )["RET"].rank(method="min", ascending=False)
        rank_df["N_grp"] = rank_df.groupby(
            ["PRD_TYP", "base_dt"], sort=False
        )["PRD_CODE"].transform("size")
        valid = rank_df["RANK"].notna()
        if valid.any():
            sub = rank_df.loc[valid]
            keys = zip(sub["PRD_TYP"].to_numpy(), sub["PRD_CODE"].to_numpy())
            vals = zip(
                sub["RANK"].astype(np.int64).to_numpy(),
                sub["N_grp"].astype(np.int64).to_numpy(),
            )
            yly_rank_dict = dict(zip(keys, vals))

    cpu_n = os.cpu_count() or 4
    env_cap = os.environ.get("FUND_METRICS_MAX_WORKERS")
    if env_cap is not None and env_cap.strip().isdigit():
        max_workers = max(1, int(env_cap.strip()))
    else:
        max_workers = max(1, cpu_n)
    n_tasks = len(payloads)
    if n_tasks == 0:
        return pd.DataFrame()
    chunksize = max(1, n_tasks // (max_workers * 4))
    ctx = mp.get_context("spawn")

    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker_shared,
        initargs=(trading_dates_sorted_np, yly_rank_dict),
    ) as executor:
        for row in executor.map(
            _calc_product_metrics_worker, payloads, chunksize=chunksize
        ):
            if row is not None:
                results.append(row)

    return pd.DataFrame(results)


# -----------------------------
# 7. 示例调用
# -----------------------------
if __name__ == "__main__":
    mp.freeze_support()

    print("⏱️  开始执行...")
    t_all0 = time.perf_counter()

    t_fetch0 = time.perf_counter()
    t_nav0 = time.perf_counter()
    df = preload_fin_prd_nav()
    nav_seconds = time.perf_counter() - t_nav0
    df_base_info = fetch_pty_prd_base_info()
    df = fill_special_prd_typ(df, df_base_info)
    # 交易日历只预取一次，传入后续计算（与 get_cached_trading_dates_np 缓存一致）
    cal_np = get_cached_trading_dates_np()
    fetch_seconds = time.perf_counter() - t_fetch0

    print("\n各指标计算的结果：")
    t_calc0 = time.perf_counter()
    metrics_df = calculate_all_fund_metrics(
        df,
        df_base_info,
        annual_method="compound",
        day_type="交易日",
        trading_calendar_np=cal_np,
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

    calc_seconds = time.perf_counter() - t_calc0
    total_seconds = time.perf_counter() - t_all0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"\n⏱️  取数耗时: {fetch_seconds:.2f}s（净值预加载 {nav_seconds:.2f}s + "
        f"基础信息 / 类型填充 / 交易日历）"
    )
    print(f"   净值内存表: {len(df):,} 行")
    print(
        f"⏱️  计算耗时: {calc_seconds:.2f}s（指标计算 + 排序 + 格式化输出 + 写文件）"
    )
    print(
        f"⏱️  总运行时间: {total_seconds:.2f}s "
        f"（{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}）"
    )