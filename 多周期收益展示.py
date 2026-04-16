import oracledb
import pandas as pd
from datetime import datetime, timedelta


# =========================================================
# 1. 数据库连接
# =========================================================
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


# =========================================================
# 2. 获取数据
# =========================================================
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
          SELECT PRD_TYP, \
                 UNIT_NVAL, \
                 AGGR_UNIT_NVAL, \
                 NAV_DT, \
                 PRD_CODE
          FROM DATA_MART_04.FIN_PRD_NAV \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(
        df["NAV_DT"].astype(str), format="%Y%m%d"
    )
    return df


def fetch_hs300_quot() -> pd.DataFrame:
    sql = """
          SELECT SECU_ID, \
                 TRD_DT, \
                 CLS_PRC, \
                 PREV_CLS_PRC
          FROM DATA_MART_04.VAR_SECU_DQUOT
          WHERE SECU_ID = '000300.IDX.CSIDX' \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["TRD_DT"] = pd.to_datetime(
        df["TRD_DT"].astype(str), format="%Y%m%d"
    )
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


# 中文数字映射
CHINESE_NUM_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def chinese_to_int(chinese_num: str) -> int:
    if chinese_num in CHINESE_NUM_MAP:
        return CHINESE_NUM_MAP[chinese_num]

    raise ValueError(f"无法解析中文数字：{chinese_num}")


PERIODS = [
    "成立以来",
    "今年以来",
    "近一月",
    "近三月",
    "近六月",
    "近一年",
    "近三年",
    "近五年"
]


def get_period_dates(
        end_dt: datetime,
        period: str,
        df_idx: pd.DataFrame
) -> tuple:
    if period == "成立以来":
        start_dt = df_idx["TRD_DT"].min()
        return start_dt, end_dt

    elif period == "今年以来":
        return datetime(end_dt.year - 1, 12, 31), end_dt

    elif period.startswith("近") and period.endswith("年"):
        try:
            n_years = chinese_to_int(period[1:-1])
        except ValueError:
            raise ValueError(f"无法解析周期：{period}")

        target_year = end_dt.year - n_years
        target_month = end_dt.month
        target_day = end_dt.day - 1

        if target_day < 1:
            first_day_of_month = datetime(target_year, target_month, 1)
            result = first_day_of_month - timedelta(days=1)
        else:
            result = datetime(target_year, target_month, target_day)

        return result, end_dt

    elif period.startswith("近") and period.endswith("月"):
        try:
            n_months = chinese_to_int(period[1:-1])
        except ValueError:
            raise ValueError(f"无法解析周期：{period}")

        from dateutil.relativedelta import relativedelta
        result = end_dt - relativedelta(months=n_months)
        result = result - timedelta(days=1)
        return result, end_dt

    elif period.startswith("近") and period.endswith("日"):
        try:
            n_days = int(period[1:-1])
        except ValueError:
            raise ValueError(f"无法解析周期：{period}")

        result = end_dt - timedelta(days=n_days)
        result = result - timedelta(days=1)
        return result, end_dt

    else:
        raise ValueError(f"未知周期：{period}")


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


# =========================================================
# 3. 产品级计算基准日
# =========================================================
def get_product_end_dt(df_nav: pd.DataFrame) -> pd.DataFrame:
    return (
        df_nav.groupby("PRD_CODE", as_index=False)["NAV_DT"]
        .max()
        .rename(columns={"NAV_DT": "END_DT"})
    )


# =========================================================
# 5. 基金收益
# =========================================================
def calc_fund_return(
        df_nav: pd.DataFrame,
        start_dt,
        end_dt,
        period: str
) -> pd.DataFrame:
    df = df_nav.copy()

    # ========== 数据清洗：只保留成立日之后的净值数据 ==========
    mask = (df["NAV_DT"] >= start_dt) & (df["NAV_DT"] <= end_dt)
    df = df[mask]

    if df.empty:
        return pd.DataFrame()

    result = []

    for prd_code, g in df.groupby("PRD_CODE"):
        g = g.sort_values("NAV_DT")
        first = g.iloc[0]
        last = g.iloc[-1]

        # ========== 判断是否需要使用虚拟起始点 ==========
        # 检查理论 start_dt 是否有数据
        has_start_data = (first["NAV_DT"] == start_dt)

        if not has_start_data:
            # 获取产品实际最早日期
            fund_earliest_dt = g["NAV_DT"].min()

            # 如果是成立以来周期，使用成立日作为基准
            # 否则使用净值表最早日期
            if period == "成立以来":
                # 使用成立日的前一天作为虚拟起始点
                virtual_start_dt = start_dt - pd.Timedelta(days=1)
            else:
                # 使用最早数据的前一天作为起始点
                virtual_start_dt = fund_earliest_dt - pd.Timedelta(days=1)

            # 虚拟起始点的净值设为 1
            nav_start = 1.0

            # 记录实际使用的起始日期
            actual_start_dt = virtual_start_dt
        else:
            # 情况 2：数据充足 - 使用实际的期初净值
            # 累计净值优先，为空则用单位净值
            nav_start = (
                first["AGGR_UNIT_NVAL"]
                if pd.notna(first["AGGR_UNIT_NVAL"])
                else first["UNIT_NVAL"]
            )
            actual_start_dt = first["NAV_DT"]

        nav_end = (
            last["AGGR_UNIT_NVAL"]
            if pd.notna(last["AGGR_UNIT_NVAL"])
            else last["UNIT_NVAL"]
        )

        if pd.isna(nav_start) or pd.isna(nav_end):
            continue

        result.append({
            "PRD_CODE": prd_code,
            "PRD_TYP": first["PRD_TYP"],
            "START_DT": actual_start_dt,
            "END_DT": last["NAV_DT"],
            "FUND_RETURN": nav_end / nav_start - 1,
            "DAYS_IN_PERIOD": g["NAV_DT"].nunique(),
            "PERIOD": period
        })

    return pd.DataFrame(result)


# =========================================================
# 6. 沪深 300 收益
# =========================================================
def calc_hs300_return(df_idx: pd.DataFrame, start_dt, end_dt) -> float:
    df = df_idx[
        (df_idx["TRD_DT"] >= start_dt) &
        (df_idx["TRD_DT"] <= end_dt)
        ].copy()

    if df.empty:
        return None

    df = df.sort_values("TRD_DT")

    first_price = df.iloc[0]["CLS_PRC"]
    last_price = df.iloc[-1]["CLS_PRC"]

    if pd.isna(first_price) or pd.isna(last_price):
        return None

    return last_price / first_price - 1


# =========================================================
# 7. 数据完整性校验
# =========================================================
def filter_by_completeness(df: pd.DataFrame, df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PRD_TYP"] = df["PRD_TYP"].fillna("NULL")

    # 标记每个产品是否符合条件
    eligible_list = []

    for idx, row in df.iterrows():
        prd_code = row["PRD_CODE"]
        period = row["PERIOD"]

        # 从基础信息表中获取该产品的成立日期
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]

        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):

            eligible_list.append(False)
            continue
        else:
            # 使用基础信息表中的成立日期
            fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 所有周期统一逻辑：使用理论周期开始日进行判断
        target_start = row["THEORY_START_DT"]

        # 判断：成立日必须 <= 周期开始日
        if fund_established_dt <= target_start:
            eligible_list.append(True)
        else:
            eligible_list.append(False)

    df["符合条件"] = eligible_list

    return df


# =========================================================
# 8. 同类排名
# =========================================================
def calc_category_rank(df: pd.DataFrame, df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 只统计符合条件的产品（成立日 <= 周期开始日）
    eligible_mask = df["符合条件"] == True

    # 初始化排名列为空字符串
    df["RANK_NUM"] = ""

    eligible_products = df[eligible_mask]

    if len(eligible_products) > 0:
        # 对每个产品单独计算排名
        for idx, row in eligible_products.iterrows():
            prd_code = row["PRD_CODE"]
            prd_typ = row["PRD_TYP"]

            # 从基础信息表中获取该产品的成立日期
            base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]

            if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
                continue
            else:
                # 使用基础信息表中的成立日期
                fund_established_dt = base_info.iloc[0]["FOUND_DT"]

            # 找出所有同类型、同周期的产品
            comparable_products = []
            for comp_idx, comp_row in eligible_products.iterrows():
                # 必须同类型、同周期、同计算基准日
                if (comp_row["PRD_TYP"] != prd_typ or
                        comp_row["PERIOD"] != row["PERIOD"] or
                        comp_row["END_DT"] != row["END_DT"]):
                    continue

                comp_prd_code = comp_row["PRD_CODE"]

                # 从基础信息表获取可比产品的成立日期
                comp_base_info = df_base_info[df_base_info["PRD_CODE"] == comp_prd_code]

                if len(comp_base_info) == 0 or pd.isna(comp_base_info.iloc[0]["FOUND_DT"]):
                    continue
                else:
                    # 使用基础信息表中的成立日期
                    comp_established_dt = comp_base_info.iloc[0]["FOUND_DT"]

                # 成立以来：只纳入成立日 <= 该产品成立日的产品
                # 其他周期：只纳入成立日 <= 该周期开始日的产品
                if row["PERIOD"] == "成立以来":
                    if comp_established_dt <= fund_established_dt:
                        comparable_products.append(comp_idx)
                else:
                    # 其他周期使用理论开始日
                    period_start = row["THEORY_START_DT"]
                    if comp_established_dt <= period_start:
                        comparable_products.append(comp_idx)

            if len(comparable_products) == 0:
                continue

            # 在这些可比产品中计算排名
            comparable_df = eligible_products.loc[comparable_products]

            # 计算收益排名
            rank_series = comparable_df["FUND_RETURN"].rank(method="min", ascending=False).astype("Int64")

            # 获取该产品的排名
            my_rank = rank_series[idx]
            total_count = len(comparable_products)

            # 赋值排名
            df.loc[idx, "RANK_NUM"] = f"{my_rank}/{total_count}名"

    return df


# =========================================================
# 9. 主流程
# =========================================================
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_hs300 = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()
    df_nav = fill_special_prd_typ(df_nav, df_base_info)
    prd_end_dt = get_product_end_dt(df_nav)

    all_result = []

    for period in PERIODS:
        period_rows = []

        for _, row in prd_end_dt.iterrows():
            prd_code = row["PRD_CODE"]
            end_dt = row["END_DT"]

            # 获取该产品的所有净值数据
            df_prd_nav = df_nav[df_nav["PRD_CODE"] == prd_code]

            # 从基础信息表中获取该产品的成立日
            base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]

            if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
                # 如果基础信息表中没有该产品或成立日期为空，跳过该产品
                continue

            fund_established_dt = base_info.iloc[0]["FOUND_DT"]

            # 过滤掉成立日之前的脏数据，确保只使用成立日之后的净值数据
            df_prd_nav_clean = df_prd_nav[df_prd_nav["NAV_DT"] >= fund_established_dt].copy()

            if df_prd_nav_clean.empty:
                # 如果过滤后没有数据，说明该产品在成立日之前没有任何净值，跳过该产品
                continue

            # ========== 获取各周期的时间区间 ==========
            # 所有周期统一使用 get_period_dates 获取理论区间
            start_dt, _ = get_period_dates(end_dt, period, df_hs300)

            df_fund = calc_fund_return(
                df_prd_nav_clean,
                start_dt,
                end_dt,
                period
            )

            if df_fund.empty:
                # 如果收益计算结果为空，添加一条包含 NaN 的记录
                df_fund = pd.DataFrame({
                    "PRD_CODE": [prd_code],
                    "PRD_TYP": ["NULL"],
                    "START_DT": [pd.NaT],
                    "END_DT": [pd.NaT],
                    "FUND_RETURN": [None],
                    "DAYS_IN_PERIOD": [None],
                    "PERIOD": [period]
                })

            if period == "成立以来":
                # 对于成立以来，使用产品实际成立日作为筛选基准
                # 这样在 filter_by_completeness 中会被特殊处理（所有产品都符合）
                df_fund["THEORY_START_DT"] = fund_established_dt  # 使用基础信息表的成立日
            else:
                df_fund["THEORY_START_DT"] = start_dt

            # ========== 根据基金收益的计算方式确定指数收益 ==========
            # 获取基金实际使用的起始日期
            fund_start_dt_used = df_fund.iloc[0]["START_DT"]
            fund_end_dt_used = df_fund.iloc[0]["END_DT"]

            # ========== 判断是否使用了虚拟起始点 ==========
            if period == "成立以来" or (start_dt < fund_established_dt):
                # 成立以来周期：强制使用成立日的前一天作为指数起始点
                target_dt = fund_established_dt - pd.Timedelta(days=1)

                # 查找指数在该日期或之前最近的数据
                idx_before = df_hs300[df_hs300["TRD_DT"] <= target_dt].copy()

                if len(idx_before) == 0:
                    # 如果找不到前一天的数据，使用指数实际最早的数据
                    idx_start = df_hs300["TRD_DT"].min()
                    idx_sub = df_hs300[
                        (df_hs300["TRD_DT"] >= idx_start) &
                        (df_hs300["TRD_DT"] <= fund_end_dt_used)
                        ].copy()
                else:
                    # 使用成立日前一天或之前最近的数据作为起始点
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_end_date = fund_end_dt_used

                    idx_sub = df_hs300[
                        (df_hs300["TRD_DT"] >= idx_start_date) &
                        (df_hs300["TRD_DT"] <= idx_end_date)
                        ].copy()
            elif pd.isna(fund_start_dt_used) or pd.isna(fund_end_dt_used):
                # 情况 1b：收益计算结果为空（NaN），跳过指数计算
                df_fund["INDEX_RETURN"] = None
                df_fund["EXCESS_RETURN"] = None
                period_rows.append(df_fund)
                continue
            else:
                # 情况 2：数据充足 - 使用与基金相同的时间区间
                # 指数使用理论 start_dt 或之前最近的数据
                idx_before = df_hs300[df_hs300["TRD_DT"] <= fund_start_dt_used].copy()

                if len(idx_before) == 0:
                    # 如果找不到，使用指数最早数据
                    idx_start = df_hs300["TRD_DT"].min()
                    idx_sub = df_hs300[
                        (df_hs300["TRD_DT"] >= idx_start) &
                        (df_hs300["TRD_DT"] <= fund_end_dt_used)
                        ].copy()
                else:
                    # 使用 start_dt 或之前最近的数据作为起始点
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_end_date = fund_end_dt_used

                    idx_sub = df_hs300[
                        (df_hs300["TRD_DT"] >= idx_start_date) &
                        (df_hs300["TRD_DT"] <= idx_end_date)
                        ].copy()

            if len(idx_sub) < 2:

                df_fund["INDEX_RETURN"] = None
                df_fund["EXCESS_RETURN"] = None
            else:
                idx_sub = idx_sub.sort_values("TRD_DT")

                # 计算指数收益
                idx_ret = (
                                  idx_sub.iloc[-1]["CLS_PRC"] -
                                  idx_sub.iloc[0]["CLS_PRC"]
                          ) / idx_sub.iloc[0]["CLS_PRC"]

                if idx_ret is None:
                    df_fund["INDEX_RETURN"] = None
                    df_fund["EXCESS_RETURN"] = None
                else:
                    df_fund["INDEX_RETURN"] = idx_ret
                    df_fund["EXCESS_RETURN"] = df_fund["FUND_RETURN"] - idx_ret

            period_rows.append(df_fund)

        if not period_rows:
            continue

        df_period = pd.concat(period_rows, ignore_index=True)

        if df_period.empty:
            continue

        # 先添加产品类型信息
        df_period["PRD_TYP"] = df_period["PRD_TYP"].fillna("NULL")

        # 进行产品准入筛选和排名计算
        df_period = filter_by_completeness(df_period, df_nav, df_base_info)

        eligible_count = (df_period["符合条件"] == True).sum()
        total_count = len(df_period)

        if eligible_count > 0:
            # 检查排名计算
            df_period = calc_category_rank(df_period, df_nav, df_base_info)

        all_result.append(df_period)

    final_df = pd.concat(all_result, ignore_index=True)

    final_df = final_df.rename(columns={"DAYS_IN_PERIOD": "周期内天数"})

    metric_map = {
        "FUND_RETURN": "基金收益",
        "EXCESS_RETURN": "超额收益",
        "INDEX_RETURN": "指数收益",
        "RANK_NUM": "同类排名"
    }

    pivot_df = final_df.pivot(
        index=["PRD_CODE", "PRD_TYP"],
        columns="PERIOD",
        values=["FUND_RETURN", "EXCESS_RETURN", "INDEX_RETURN", "RANK_NUM"]
    )

    pivot_df.columns = [
        f"{period}_{metric_map[metric]}"
        for metric, period in pivot_df.columns
    ]

    pivot_df = pivot_df.reset_index()

    for col in pivot_df.columns:
        if col in ["PRD_CODE", "PRD_TYP"]:
            continue

        if "基金收益" in col or "超额收益" in col or "指数收益" in col:
            pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce")
            pivot_df[col] = (pivot_df[col] * 100).round(2).astype(str) + "%"

    # 确保所有包含排名数据的列都是字符串类型，防止被误识别为日期
    rank_cols = [col for col in pivot_df.columns if "同类排名" in col]
    for col in rank_cols:
        pivot_df[col] = pivot_df[col].astype(str)

    # 获取所有周期
    periods = final_df["PERIOD"].unique()

    # 构建新的列顺序：先放基本信息，然后按指标类型分组（各周期基金收益、超额收益、指数收益、同类排名）
    new_col_order = ["PRD_CODE", "PRD_TYP"]

    # 所有周期的基金收益
    for period in periods:
        new_col_order.append(f"{period}_基金收益")

    # 所有周期的超额收益
    for period in periods:
        new_col_order.append(f"{period}_超额收益")

    # 所有周期的指数收益
    for period in periods:
        new_col_order.append(f"{period}_指数收益")

    # 所有周期的同类排名
    for period in periods:
        new_col_order.append(f"{period}_同类排名")

    # 只保留实际存在的列
    available_cols = [col for col in new_col_order if col in pivot_df.columns]
    combined_df = pivot_df[available_cols]

    # 导出 CSV
    output_file = "多周期收益展示.csv"
    combined_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n结果已保存至：{output_file}")



