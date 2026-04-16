import oracledb
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

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

    # 处理"十二"、"十三"等
    if len(chinese_num) == 2 and chinese_num[0] == "十":
        return 10 + CHINESE_NUM_MAP.get(chinese_num[1], 0)

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
# 产品级计算基准日
# =========================================================
def get_product_end_dt(df_nav: pd.DataFrame) -> pd.DataFrame:
    return (
        df_nav.groupby("PRD_CODE", as_index=False)["NAV_DT"]
        .max()
        .rename(columns={"NAV_DT": "END_DT"})
    )


# =========================================================
# 基金收益计算
# =========================================================
def calc_fund_return(
        df_nav: pd.DataFrame,
        start_dt,
        end_dt,
        period: str
) -> pd.DataFrame:
    df = df_nav.copy()

    mask = (df["NAV_DT"] >= start_dt) & (df["NAV_DT"] <= end_dt)
    df = df[mask]

    if df.empty:
        return pd.DataFrame()

    result = []

    for prd_code, g in df.groupby("PRD_CODE"):
        g = g.sort_values("NAV_DT")

        first = g.iloc[0]
        last = g.iloc[-1]

        # 判断是否需要使用虚拟起始点
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
# 沪深 300 收益计算
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
# 多周期收益计算
# =========================================================
def calc_multi_period_return(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算多周期收益（基金和沪深300）
    """
    all_result = []
    prd_end_dt = get_product_end_dt(df_nav)

    for period in PERIODS:
        period_rows = []

        for _, row in prd_end_dt.iterrows():
            prd_code = row["PRD_CODE"]
            end_dt = row["END_DT"]

            # 获取该产品的所有净值数据
            df_prd_nav = df_nav[df_nav["PRD_CODE"] == prd_code]

            # 从基础信息表中获取该产品的成立日（唯一数据源）
            base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]

            if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
                # 如果基础信息表中没有该产品或成立日期为空，跳过该产品
                continue

            fund_established_dt = base_info.iloc[0]["FOUND_DT"]

            # 数据清洗：过滤掉成立日之前的脏数据
            df_prd_nav_clean = df_prd_nav[df_prd_nav["NAV_DT"] >= fund_established_dt].copy()

            if df_prd_nav_clean.empty:
                # 如果过滤后没有数据，说明该产品在成立日之前没有任何净值
                # 跳过该产品
                continue

            # 获取各周期的时间区间
            start_dt, _ = get_period_dates(end_dt, period, df_idx)

            df_fund = calc_fund_return(
                df_prd_nav_clean,  # 使用清洗后的数据
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
                df_fund["THEORY_START_DT"] = fund_established_dt  # 使用基础信息表的成立日
            else:
                df_fund["THEORY_START_DT"] = start_dt

            # 根据基金收益的计算方式确定指数收益
            # 获取基金实际使用的起始日期
            fund_start_dt_used = df_fund.iloc[0]["START_DT"]
            fund_end_dt_used = df_fund.iloc[0]["END_DT"]

            # 判断是否使用了虚拟起始点
            if period == "成立以来" or (start_dt < fund_established_dt):
                # 成立以来周期：强制使用成立日的前一天作为指数起始点
                target_dt = fund_established_dt - pd.Timedelta(days=1)

                # 查找指数在该日期或之前最近的数据
                idx_before = df_idx[df_idx["TRD_DT"] <= target_dt].copy()

                if len(idx_before) == 0:
                    # 如果找不到前一天的数据，使用指数实际最早的数据
                    idx_start = df_idx["TRD_DT"].min()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start) &
                        (df_idx["TRD_DT"] <= fund_end_dt_used)
                        ].copy()
                else:
                    # 使用成立日前一天或之前最近的数据作为起始点
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_end_date = fund_end_dt_used

                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= idx_end_date)
                        ].copy()
            elif pd.isna(fund_start_dt_used) or pd.isna(fund_end_dt_used):
                # 情况 1b：收益计算结果为空（NaN），跳过指数计算
                df_fund["IDX_RETURN"] = None
                df_fund["EXCESS_RETURN"] = None
                period_rows.append(df_fund)
                continue
            else:
                # 情况 2：数据充足 - 使用与基金相同的时间区间
                # 指数使用理论 start_dt 或之前最近的数据
                idx_before = df_idx[df_idx["TRD_DT"] <= fund_start_dt_used].copy()

                if len(idx_before) == 0:
                    # 如果找不到，使用指数最早数据
                    idx_start = df_idx["TRD_DT"].min()
                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start) &
                        (df_idx["TRD_DT"] <= fund_end_dt_used)
                        ].copy()
                else:
                    # 使用 start_dt 或之前最近的数据作为起始点
                    idx_start_date = idx_before["TRD_DT"].max()
                    idx_end_date = fund_end_dt_used

                    idx_sub = df_idx[
                        (df_idx["TRD_DT"] >= idx_start_date) &
                        (df_idx["TRD_DT"] <= idx_end_date)
                        ].copy()

            if len(idx_sub) < 2:
                df_fund["IDX_RETURN"] = None
                df_fund["EXCESS_RETURN"] = None
            else:
                idx_sub = idx_sub.sort_values("TRD_DT")

                # 计算指数收益
                idx_ret = (
                                  idx_sub.iloc[-1]["CLS_PRC"] -
                                  idx_sub.iloc[0]["CLS_PRC"]
                          ) / idx_sub.iloc[0]["CLS_PRC"]

                if idx_ret is None:
                    df_fund["IDX_RETURN"] = None
                    df_fund["EXCESS_RETURN"] = None
                else:
                    df_fund["IDX_RETURN"] = idx_ret
                    # 计算几何超额收益
                    fund_ret = df_fund.iloc[0]["FUND_RETURN"]
                    if fund_ret is not None:
                        excess_return = (1 + fund_ret) / (1 + idx_ret) - 1 if (1 + idx_ret) != 0 else 0
                        df_fund["EXCESS_RETURN"] = excess_return
                    else:
                        df_fund["EXCESS_RETURN"] = None

            period_rows.append(df_fund)

        if not period_rows:
            continue

        df_period = pd.concat(period_rows, ignore_index=True)

        if df_period.empty:
            continue

        # 先添加产品类型信息
        df_period["PRD_TYP"] = df_period["PRD_TYP"].fillna("NULL")

        all_result.append(df_period)

    if not all_result:
        return pd.DataFrame()

    final_df = pd.concat(all_result, ignore_index=True)
    return final_df


# =========================================================
# 数据完整性校验
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
# 计算同类平均收益
# =========================================================
def calc_peer_average(df_multi_period: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类平均收益
    """
    peer_avg = (
        df_multi_period.groupby(["PRD_TYP", "PERIOD"]) ["FUND_RETURN"]
        .mean()
        .reset_index(name="PEER_AVG_RETURN")
    )
    return df_multi_period.merge(peer_avg, on=["PRD_TYP", "PERIOD"], how="left")


# =========================================================
# 计算同类排名
# =========================================================
def calc_category_rank(df: pd.DataFrame, df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类排名
    """
    df = df.copy()

    # 只统计符合条件的产品（成立日 <= 周期开始日）
    eligible_mask = df["符合条件"] == True

    # 初始化排名列为空字符串
    df["CATEGORY_RANK"] = ""

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
            df.loc[idx, "CATEGORY_RANK"] = f"{my_rank}/{total_count}"

    return df


# =========================================================
# 计算四分位排名
# =========================================================
def calc_quarterly_rank(df_multi_period: pd.DataFrame) -> pd.DataFrame:
    """
    计算四分位排名
    """
    df = df_multi_period.copy()
    
    # 只对符合条件的产品计算四分位
    eligible_mask = df["符合条件"] == True
    eligible_df = df[eligible_mask]
    
    for (prd_typ, period), group in eligible_df.groupby(["PRD_TYP", "PERIOD"]):
        # 按基金收益降序排序
        sorted_group = group.sort_values("FUND_RETURN", ascending=False)
        total_count = len(sorted_group)
        
        if total_count == 0:
            continue
        
        # 计算四分位
        q1 = int(total_count * 0.25)
        q2 = int(total_count * 0.5)
        q3 = int(total_count * 0.75)
        
        # 分配四分位排名
        for i, (idx, row) in enumerate(sorted_group.iterrows()):
            if i < q1:
                rank = "1/4"
            elif i < q2:
                rank = "2/4"
            elif i < q3:
                rank = "3/4"
            else:
                rank = "4/4"
            df.loc[idx, "QUARTER_RANK"] = rank
    
    return df


# =========================================================
# 主流程
# =========================================================
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_hs300 = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()
    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 计算多周期收益
    df_multi_period = calc_multi_period_return(df_nav, df_hs300, df_base_info)

    if df_multi_period.empty:
        print("没有可用的多周期收益数据")
        exit()

    # 数据完整性校验
    df_multi_period = filter_by_completeness(df_multi_period, df_nav, df_base_info)

    # 计算同类平均收益
    df_multi_period = calc_peer_average(df_multi_period)

    # 计算同类排名
    df_multi_period = calc_category_rank(df_multi_period, df_nav, df_base_info)

    # 计算四分位排名
    df_multi_period = calc_quarterly_rank(df_multi_period)

    # 格式化输出
    for col in ["FUND_RETURN", "IDX_RETURN", "EXCESS_RETURN", "PEER_AVG_RETURN"]:
        if col in df_multi_period.columns:
            df_multi_period[col] = df_multi_period[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "--")

    # 创建产品名称映射
    prd_name_map = df_base_info.set_index("PRD_CODE")["PRD_NAME"].to_dict()
    
    # 输出结果
    output_file = "阶段收益明细结果.txt"
    output_lines = []
    output_lines.append("=" * 200)
    output_lines.append("📊 阶段收益明细（BI效果）")
    output_lines.append(f"✅ 共统计产品数量：{len(df_multi_period['PRD_CODE'].unique())}")
    output_lines.append("=" * 200)
    output_lines.append("")

    # 按产品分组输出
    for prd_code in df_multi_period["PRD_CODE"].unique():
        prd_data = df_multi_period[df_multi_period["PRD_CODE"] == prd_code]
        prd_typ = prd_data.iloc[0]["PRD_TYP"]
        prd_name = prd_name_map.get(prd_code, prd_code)
        
        output_lines.append("-" * 200)
        output_lines.append(f"产品代码: {prd_code} ({prd_name}) | 产品类型: {prd_typ}")
        output_lines.append("-" * 200)
        
        # 输出表头
        header = ["区间", "基金收益", "沪深300", "超额收益(几何)", "同类平均", "同类排名", "四分位"]
        output_lines.append("|" + "|".join([f"{h:15s}" for h in header]) + "|")
        output_lines.append("-" * 200)
        
        # 按区间顺序输出
        for period in PERIODS:
            period_data = prd_data[prd_data["PERIOD"] == period]
            if not period_data.empty:
                row = period_data.iloc[0]
                data_row = [
                    period,
                    row["FUND_RETURN"],
                    row["IDX_RETURN"],
                    row["EXCESS_RETURN"],
                    row.get("PEER_AVG_RETURN", "--"),
                    row.get("CATEGORY_RANK", "--"),
                    row.get("QUARTER_RANK", "--")
                ]
                output_lines.append("|" + "|".join([f"{str(x):15s}" for x in data_row]) + "|")
            else:
                # 无数据时显示空行
                data_row = [period, "--", "--", "--", "--", "--", "--"]
                output_lines.append("|" + "|".join([f"{str(x):15s}" for x in data_row]) + "|")
        
        output_lines.append("")

    output_lines.append("=" * 200)
    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n结果已保存至：{output_file}")