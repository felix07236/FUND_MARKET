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
                 AGGR_UNIT_NVAL
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


def calc_period_annualized(
        virtual_start_dt: pd.Timestamp,
        current_dt: pd.Timestamp,
        end_aggr: float,
        start_aggr: float = 1.0,
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

    # 区间收益
    total_return = (end_aggr - start_aggr) / start_aggr

    # 年化收益（复利）
    annualized = calc_since_annualized_return(
        start_value=start_aggr,
        end_value=end_aggr,
        days=days_since,
        annual_days=annual_days
    )

    return total_return, annualized, actual_end_dt


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


def calc_alpha_and_attack(fund_daily_ret, idx_daily_ret, day_type="自然日", trading_days_df=None, prd_code=None):
    if len(fund_daily_ret) < 2 or len(idx_daily_ret) < 2:
        return {"Alpha": np.nan, "月超额胜率": np.nan, "进攻能力": np.nan}

    merged = pd.DataFrame({
        "fund_ret": fund_daily_ret,
        "idx_ret": idx_daily_ret
    }).dropna()

    if len(merged) < 2:
        return {"Alpha": np.nan, "月超额胜率": np.nan, "进攻能力": np.nan}

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

    alpha = fund_excess_ret.mean() - beta * idx_excess_ret.mean()

    win_rate = np.nan
    attack = np.nan

    return {"Alpha": alpha, "月超额胜率": win_rate, "进攻能力": attack}


def calc_monthly_metrics(fund_month, idx_month):
    merged = fund_month.merge(idx_month, on="年月", how="inner")

    if len(merged) < 2:
        return {"月超额胜率": np.nan, "进攻能力": np.nan}

    # 月超额胜率
    win_months = merged["月收益"] > merged["指数月收益"]
    win_count = win_months.sum()
    total_count = len(merged)
    win_rate = win_months.mean()

    # 进攻能力
    up = merged[merged["指数月收益"] > 0]
    if not up.empty and up["指数月收益"].mean() != 0:
        attack = up["月收益"].mean() / up["指数月收益"].mean()
    else:
        attack = np.nan

    return {"月超额胜率": win_rate, "进攻能力": attack}


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


def _get_idx_start_date_by_anchor(idx_df: pd.DataFrame, anchor_dt: pd.Timestamp) -> pd.Timestamp:
    idx_before = idx_df[idx_df["NAV_DT"] <= anchor_dt]
    if len(idx_before) == 0:
        return idx_df["NAV_DT"].min()
    return idx_before["NAV_DT"].max()


def _normalize_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    return pd.to_datetime(ts)


def _is_product_eligible_for_row(
        comp_row: pd.Series,
        target_row: pd.Series,
        product_base_info: pd.DataFrame
) -> bool:
    comp_prd_code = comp_row["产品代码"]
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
        product_base_info: pd.DataFrame
) -> list:
    comparable_indices = []
    for comp_idx, comp_row in df.iterrows():
        if comp_row["产品类型"] != target_row["产品类型"]:
            continue
        if comp_row["周期"] != target_row["周期"]:
            continue
        if comp_row["计算基准日"] != target_row["计算基准日"]:
            continue
        if _is_product_eligible_for_row(comp_row, target_row, product_base_info):
            comparable_indices.append(comp_idx)
    return comparable_indices


# --------------------------------------------------
# 5. 单产品指标计算
# --------------------------------------------------
def calc_product_metrics(prd_df, idx_df, periods, fund_established_dt,
                         day_type="自然日", trading_days_df=None):
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

        if period == "成立以来":
            virtual_start_dt = fund_established_dt - pd.Timedelta(days=1)
            nav_start = 1.0
            actual_start_dt = virtual_start_dt
            has_start_data = False
        else:
            # ========== 交易日模式：调整理论起始日到实际交易日 ==========
            if day_type == "交易日" and trading_days_df is not None:
                trading_dates_sorted = trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].sort_values()
                prev_trading_days = trading_dates_sorted[trading_dates_sorted <= theoretical_start_dt]

                if len(prev_trading_days) > 0:
                    actual_theoretical_start = prev_trading_days.max()
                else:
                    actual_theoretical_start = theoretical_start_dt
            else:
                actual_theoretical_start = theoretical_start_dt

            sub_at_start = prd_df_clean[prd_df_clean["NAV_DT"] == actual_theoretical_start]
            has_start_data = len(sub_at_start) > 0

            if not has_start_data:
                if actual_theoretical_start >= fund_established_dt:
                    first_nav_after_establish = prd_df_clean.iloc[0]
                    nav_start = first_nav_after_establish["AGGR_UNIT_NVAL"]
                    actual_start_dt = first_nav_after_establish["NAV_DT"]
                    virtual_start_dt = actual_theoretical_start
                else:
                    fund_earliest_dt = prd_df_clean["NAV_DT"].min()
                    virtual_start_dt = fund_earliest_dt - pd.Timedelta(days=1)
                    nav_start = 1.0
                    actual_start_dt = virtual_start_dt
            else:
                nav_start = sub_at_start.iloc[0]["AGGR_UNIT_NVAL"]
                actual_start_dt = actual_theoretical_start
                virtual_start_dt = actual_theoretical_start

        sub_end = prd_df_clean[prd_df_clean["NAV_DT"] <= base_dt]
        if len(sub_end) == 0:
            continue

        nav_end = sub_end.iloc[-1]["AGGR_UNIT_NVAL"]
        current_dt = sub_end.iloc[-1]["NAV_DT"]

        if pd.isna(nav_start) or pd.isna(nav_end):
            continue

        total_ret, ann_ret, actual_end_dt = calc_period_annualized(
            virtual_start_dt=virtual_start_dt,
            current_dt=current_dt,
            end_aggr=nav_end,
            start_aggr=nav_start,
            day_type=day_type,
            trading_days_df=trading_days_df
        )

        if actual_end_dt != current_dt:
            actual_end_data = prd_df_clean[prd_df_clean["NAV_DT"] == actual_end_dt]
            if len(actual_end_data) > 0:
                end_aggr_adjusted = actual_end_data.iloc[0]["AGGR_UNIT_NVAL"]
                total_ret, ann_ret, _ = calc_period_annualized(
                    virtual_start_dt=virtual_start_dt,
                    current_dt=actual_end_dt,
                    end_aggr=end_aggr_adjusted,
                    start_aggr=nav_start,
                    day_type=day_type,
                    trading_days_df=trading_days_df
                )
                actual_end_dt_for_monthly = actual_end_dt
            else:
                actual_end_dt_for_monthly = current_dt
        else:
            actual_end_dt_for_monthly = current_dt

        if period == "成立以来":
            full_nav_series = _add_virtual_start_nav(prd_df_clean, virtual_start_dt)
        elif not has_start_data and theoretical_start_dt < fund_established_dt:
            full_nav_series = _add_virtual_start_nav(prd_df_clean, virtual_start_dt)
        elif not has_start_data and theoretical_start_dt >= fund_established_dt:
            full_nav_series = prd_df_clean.copy()
            full_nav_series = full_nav_series.sort_values("NAV_DT").reset_index(drop=True)
        else:
            full_nav_series = prd_df_clean[prd_df_clean["NAV_DT"] >= theoretical_start_dt].copy()
            full_nav_series = full_nav_series.sort_values("NAV_DT").reset_index(drop=True)

        if day_type == "交易日" and trading_days_df is not None:
            trading_dates = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)
            full_nav_series = full_nav_series[
                full_nav_series["NAV_DT"].isin(trading_dates) |
                (full_nav_series["NAV_DT"] == virtual_start_dt)
                ].reset_index(drop=True)

        full_nav_series["年月"] = full_nav_series["NAV_DT"].dt.to_period("M")

        # ========== 足月逻辑：先标记哪些月份是完整月份 ==========
        complete_months = []  # 初始化为空列表

        if len(full_nav_series) >= 1:
            # 获取实际的数据起始日和结束日
            actual_data_start = full_nav_series["NAV_DT"].min()
            actual_data_end = full_nav_series["NAV_DT"].max()

            # 提取所有月份
            all_months = full_nav_series["年月"].unique()

            # 标记每个月是否完整
            for month_period in all_months:
                is_complete = True

                # 检查起始月是否完整
                if month_period == actual_data_start.to_period("M"):
                    if day_type == "交易日" and trading_days_df is not None:
                        # 交易日模式：获取该月的所有交易日
                        trading_dates_in_start_month = trading_days_df[
                            (trading_days_df["IS_TRD_DT"] == True) &
                            (trading_days_df["CALD_DATE"].dt.to_period("M") == month_period)
                            ]["CALD_DATE"]

                        if len(trading_dates_in_start_month) > 0:
                            # 该月第一个交易日
                            first_trading_day_of_month = trading_dates_in_start_month.min()
                            # 如果实际数据起始日晚于该月第一个交易日，说明起始月不完整
                            if actual_data_start > first_trading_day_of_month:
                                is_complete = False
                    else:
                        # 自然日模式：如果起始日不是该月1号，则不完整
                        if actual_data_start.day != 1:
                            is_complete = False

                # 检查结束月是否完整
                if month_period == actual_data_end.to_period("M"):
                    if day_type == "交易日" and trading_days_df is not None:
                        # 交易日模式：获取该月的所有交易日
                        trading_dates_in_end_month = trading_days_df[
                            (trading_days_df["IS_TRD_DT"] == True) &
                            (trading_days_df["CALD_DATE"].dt.to_period("M") == month_period)
                            ]["CALD_DATE"]

                        if len(trading_dates_in_end_month) > 0:
                            # 该月最后一个交易日
                            last_trading_day_of_month = trading_dates_in_end_month.max()
                            # 如果实际数据结束日早于该月最后一个交易日，说明结束月不完整
                            if actual_data_end < last_trading_day_of_month:
                                is_complete = False
                    else:
                        # 自然日模式：如果结束日不是该月最后一天，则不完整
                        end_month_last_day = (actual_data_end + pd.offsets.MonthEnd(0))
                        if actual_data_end.day != end_month_last_day.day:
                            is_complete = False

                if is_complete:
                    complete_months.append(month_period)

        # 提取每月末净值（保留所有月份用于计算）
        month_end_nav = (
            full_nav_series.groupby("年月")["AGGR_UNIT_NVAL"]
            .last()
            .reset_index(name="月末净值")
        )

        # 计算月收益：本月末净值 / 上月末净值 - 1
        month_end_nav["上月末净值"] = month_end_nav["月末净值"].shift(1)
        # 今年以来场景下，若缺少上一年12月数据，首月（通常为1月）改用区间起始净值作为基准
        if period == "今年以来" and len(month_end_nav) > 0:
            first_idx = month_end_nav.index[0]
            if pd.isna(month_end_nav.loc[first_idx, "上月末净值"]) and pd.notna(nav_start) and nav_start > 0:
                month_end_nav.loc[first_idx, "上月末净值"] = nav_start
        fund_month = _filter_complete_month_returns(month_end_nav, "月收益", complete_months)

        fund_daily_ret = full_nav_series.set_index("NAV_DT")["AGGR_UNIT_NVAL"].pct_change().dropna()

        if period == "成立以来" or (not has_start_data and theoretical_start_dt < fund_established_dt):
            idx_anchor_dt = fund_established_dt - pd.Timedelta(days=1)
        else:
            idx_anchor_dt = actual_start_dt
        idx_start_date = _get_idx_start_date_by_anchor(idx_df, idx_anchor_dt)
        idx_end_date = actual_end_dt_for_monthly

        idx_sub = idx_df[
            (idx_df["NAV_DT"] >= idx_start_date) &
            (idx_df["NAV_DT"] <= idx_end_date)
            ]

        if len(idx_sub) < 2:
            results.append({
                "产品代码": prd_code,
                "产品类型": prd_typ,
                "周期": period,
                "计算基准日": current_dt.date(),
                "THEORY_START_DT": theory_start_dt_for_rank,
                "INDEX_START_DT_CALC": idx_start_date,
                "INDEX_END_DT_CALC": idx_end_date,
                "收益": total_ret,
                "年化收益": ann_ret,
                "Alpha": np.nan,
                "月超额胜率": np.nan,
                "进攻能力": np.nan
            })
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

        # 提取指数每月末净值（保留所有月份用于计算）
        idx_month_end = (
            idx_sub_copy.groupby("年月")["INDEX_CLOSE"]
            .last()
            .reset_index(name="月末净值")
        )

        # 计算指数月收益：本月末净值 / 上月末净值 - 1
        idx_month_end["上月末净值"] = idx_month_end["月末净值"].shift(1)
        # 今年以来场景下，若缺少上一年12月数据，首月改用指数区间起始点作为基准
        if period == "今年以来" and len(idx_month_end) > 0:
            first_idx = idx_month_end.index[0]
            idx_start_row = idx_sub_copy[idx_sub_copy["NAV_DT"] == idx_start_date]
            if len(idx_start_row) > 0:
                idx_start_close = idx_start_row.iloc[0]["INDEX_CLOSE"]
            elif len(idx_sub_copy) > 0:
                idx_start_close = idx_sub_copy.iloc[0]["INDEX_CLOSE"]
            else:
                idx_start_close = np.nan
            if pd.isna(idx_month_end.loc[first_idx, "上月末净值"]) and pd.notna(
                    idx_start_close) and idx_start_close > 0:
                idx_month_end.loc[first_idx, "上月末净值"] = idx_start_close
        idx_month = _filter_complete_month_returns(idx_month_end, "指数月收益", complete_months)

        idx_daily_ret = idx_sub_copy.set_index("NAV_DT")["INDEX_CLOSE"].pct_change().dropna()

        if day_type == "交易日" and trading_days_df is not None:
            trading_dates = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)
            idx_daily_ret = idx_daily_ret[idx_daily_ret.index.isin(trading_dates)]

        alpha_result = calc_alpha_and_attack(
            fund_daily_ret=fund_daily_ret,
            idx_daily_ret=idx_daily_ret,
            day_type=day_type,
            trading_days_df=trading_days_df,
            prd_code=prd_code
        )

        monthly_metrics = calc_monthly_metrics(fund_month, idx_month)

        metrics = {
            "Alpha": alpha_result["Alpha"],
            "月超额胜率": monthly_metrics["月超额胜率"],
            "进攻能力": monthly_metrics["进攻能力"]
        }

        results.append({
            "计算模式": day_type,
            "产品代码": prd_code,
            "产品类型": prd_typ,
            "周期": period,
            "计算基准日": current_dt.date(),
            "THEORY_START_DT": theory_start_dt_for_rank,
            "INDEX_START_DT_CALC": idx_start_date,
            "INDEX_END_DT_CALC": idx_end_date,
            "收益": total_ret,
            "年化收益": ann_ret,
            **metrics
        })

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

    for metric in metrics:
        result[f"同类平均{metric}"] = np.nan
        result[f"沪深300{metric}"] = np.nan

    for idx, row in result.iterrows():
        comparable_indices = _get_comparable_product_indices(result, row, product_base_info)
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

        # ========== 交易日模式：剔除非交易日 ==========
        if day_type == "交易日" and trading_days_df is not None:
            trading_dates = set(trading_days_df[trading_days_df["IS_TRD_DT"] == True]["CALD_DATE"].values)
            idx_sub = idx_sub[idx_sub["NAV_DT"].isin(trading_dates)].copy()

        if len(idx_sub) < 2:
            continue

        ret = idx_sub["INDEX_CLOSE"].iloc[-1] / idx_sub["INDEX_CLOSE"].iloc[0] - 1

        # ========== 根据计算模式选择年化系数和天数计算方式 ==========
        if day_type == "交易日" and trading_days_df is not None:
            # 交易日模式：使用250天作为年化系数，计算交易日天数
            days = len(idx_sub)
            annual_days = 250
        else:
            # 自然日模式：使用365天作为年化系数，计算自然日天数
            days = (idx_sub["NAV_DT"].iloc[-1] - idx_sub["NAV_DT"].iloc[0]).days
            annual_days = 365

        ann_ret = (1 + ret) ** (annual_days / max(days, 1)) - 1

        result.loc[idx, "沪深300收益"] = ret
        result.loc[idx, "沪深300年化收益"] = ann_ret
        result.loc[idx, "沪深300Alpha"] = 0
        result.loc[idx, "沪深300月超额胜率"] = np.nan
        result.loc[idx, "沪深300进攻能力"] = 1

    return result


# --------------------------------------------------
# 7. 排名（使用成立日筛选逻辑）
# --------------------------------------------------
def rank_by_established_date(df: pd.DataFrame, product_base_info: pd.DataFrame) -> pd.DataFrame:
    metrics = ["收益", "年化收益", "Alpha", "月超额胜率", "进攻能力"]

    # 初始化排名列为空字符串
    for col in metrics:
        df[f"{col}排名"] = ""

    # 对每个产品单独计算排名
    for idx, row in df.iterrows():
        prd_code = row["产品代码"]
        prd_typ = row["产品类型"]
        period = row["周期"]

        # 跳过基准产品（兼容历史口径）
        if prd_code in ["同类平均", "沪深300", "沪深 300"]:
            continue

        # 获取该产品的成立日（从基础信息表）
        prd_base = product_base_info[product_base_info["PRD_CODE"] == prd_code]
        if len(prd_base) == 0:
            continue

        fund_established_dt = prd_base["FOUND_DT"].iloc[0]
        if pd.isna(fund_established_dt):
            continue

        if not _is_product_eligible_for_row(row, row, product_base_info):
            continue

        comparable_products = _get_comparable_product_indices(df, row, product_base_info)

        if len(comparable_products) == 0:
            continue

        # 在这些可比产品中计算排名
        comparable_df = df.loc[comparable_products]

        # 对每个指标分别排名
        for metric in metrics:
            rank_series = comparable_df[metric].rank(method="min", ascending=False).astype("Int64")
            my_rank = rank_series[idx]
            total_count = len(comparable_products)

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
    if day_type == "交易日":
        trading_days_df = fetch_trading_days()

    # ========== 产品类型填充 ==========
    prd_df = fill_special_prd_typ(prd_df, product_base_info)

    all_results = []
    for prd_code, g in prd_df.groupby("PRD_CODE"):
        # 从基础信息表获取该产品的成立日
        prd_base = product_base_info[product_base_info["PRD_CODE"] == prd_code]
        if len(prd_base) == 0:
            continue

        fund_established_dt = prd_base["FOUND_DT"].iloc[0]
        if pd.isna(fund_established_dt):
            continue

        all_results.extend(calc_product_metrics(g, idx_df, periods, fund_established_dt,
                                                day_type, trading_days_df))

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

    return df_rank


# --------------------------------------------------
# 10. 执行入口
# --------------------------------------------------
if __name__ == "__main__":
    print("开始计算收益能力指标...")
    result = main(day_type="交易日")

    if result is not None and len(result) > 0:
        print(f"\n计算完成！共 {len(result)} 条记录")
        print("\n前10条数据预览：")
        print(result.head(10).to_string())

        result.to_csv("收益能力指标.csv", index=False, encoding="utf-8-sig", quoting=1)
        print("\n💾 结果已保存至：收益能力指标.csv")
    else:
        print("\n⚠️ 警告：没有计算出任何数据！")
