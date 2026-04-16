import oracledb
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re

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
def fetch_fin_prd_nav(prd_codes: list) -> pd.DataFrame:
    placeholders = ','.join([':' + str(i) for i in range(len(prd_codes))])
    sql = f"""
          SELECT PRD_TYP, UNIT_NVAL, AGGR_UNIT_NVAL, NAV_DT, PRD_CODE
          FROM DATA_MART_04.FIN_PRD_NAV
          WHERE PRD_CODE IN ({placeholders})
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params=prd_codes)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


def fetch_index_quot(index_code) -> pd.DataFrame:
    sql = f"""
          SELECT SECU_ID, \
                 TRD_DT, \
                 CLS_PRC, \
                 PREV_CLS_PRC
          FROM DATA_MART_04.VAR_SECU_DQUOT
          WHERE SECU_ID = :index_code \
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params={':index_code': index_code})

    df["TRD_DT"] = pd.to_datetime(
        df["TRD_DT"].astype(str), format="%Y%m%d"
    )
    return df


def fetch_pty_prd_base_info(prd_codes: list) -> pd.DataFrame:
    placeholders = ','.join([':' + str(i) for i in range(len(prd_codes))])
    sql = f"""
          SELECT PRD_CODE, FOUND_DT, PRD_NAME, PRD_FULL_NAME, PRD_TYP
          FROM DATA_MART_04.PTY_PRD_BASE_INFO
          WHERE PRD_CODE IN ({placeholders})
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params=prd_codes)

    df["FOUND_DT"] = pd.to_datetime(df["FOUND_DT"], format="%Y%m%d", errors="coerce")
    return df


# 日期验证函数
def validate_date(date_str):
    """验证日期字符串是否为有效的YYYY-MM-DD格式"""
    if not date_str:
        return True

    # 检查格式
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, date_str):
        raise ValueError(f"日期格式错误：{date_str}，应为YYYY-MM-DD格式")

    # 检查日期是否有效
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError as e:
        raise ValueError(f"无效的日期：{date_str}，{str(e)}")


# 中文数字映射
CHINESE_NUM_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
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


# 预定义周期
PREDEFINED_PERIODS = [
    "成立以来",
    "今年以来",
    "近一月",
    "近三月",
    "近六月",
    "近一年",
    "近两年",
    "近三年",
    "近五年"
]


# =========================================================
# 3. 时间周期处理
# =========================================================
def get_period_dates(
        end_dt: datetime,
        period: str,
        fund_established_dt=None
) -> tuple:
    if period == "成立以来":
        if fund_established_dt:
            return fund_established_dt, end_dt
        else:
            return datetime(1900, 1, 1), end_dt

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

    else:
        raise ValueError(f"未知周期：{period}")


# =========================================================
# 4. 收益计算
# =========================================================
def calc_fund_return(
        df_nav: pd.DataFrame,
        start_dt,
        end_dt,
        period: str,
        fund_established_dt=None
) -> pd.DataFrame:
    df = df_nav.copy()

    # 数据清洗：只保留指定时间范围内的净值数据
    mask = (df["NAV_DT"] >= start_dt) & (df["NAV_DT"] <= end_dt)
    df_filtered = df[mask]

    if df_filtered.empty:
        return pd.DataFrame()

    result = []

    for prd_code, g in df_filtered.groupby("PRD_CODE"):
        g = g.sort_values("NAV_DT")
        first = g.iloc[0]
        last = g.iloc[-1]


        # 情况1：成立以来周期 - 始终使用成立日前一天 + 净值1.0
        if period == "成立以来":
            virtual_start_dt = start_dt - pd.Timedelta(days=1)
            nav_start = 1.0
            actual_start_dt = virtual_start_dt

        else:
            # 其他周期的逻辑
            has_start_data = (first["NAV_DT"] == start_dt)

            # ========== 调试信息 ==========
            if period == "今年以来" and prd_code == "1029":
                print(f"  has_start_data: {has_start_data}")
            # ============================

            if not has_start_data:
                # 获取产品实际最早日期
                fund_earliest_dt = g["NAV_DT"].min()


                # 情况2：理论起始日 < 成立日 - 使用成立日前一天 + 净值1.0
                if fund_established_dt and start_dt < fund_established_dt:
                    virtual_start_dt = fund_established_dt - pd.Timedelta(days=1)
                    nav_start = 1.0
                    actual_start_dt = virtual_start_dt


                # 情况3：理论起始日 > 成立日 且 第一个净值日 > 理论起始日 - 使用第一个净值日的实际净值
                elif fund_established_dt and start_dt > fund_established_dt and fund_earliest_dt > start_dt:
                    nav_start = (
                        first["AGGR_UNIT_NVAL"]
                        if pd.notna(first["AGGR_UNIT_NVAL"])
                        else first["UNIT_NVAL"]
                    )
                    actual_start_dt = first["NAV_DT"]

                # 情况4：其他情况 - 使用最早数据前一天 + 净值1.0
                else:
                    virtual_start_dt = fund_earliest_dt - pd.Timedelta(days=1)
                    nav_start = 1.0
                    actual_start_dt = virtual_start_dt

            else:
                # 数据充足 - 使用理论起始日的实际净值
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

        fund_return = nav_end / nav_start - 1

        result.append({
            "PRD_CODE": prd_code,
            "PRD_TYP": first["PRD_TYP"],
            "START_DT": actual_start_dt,
            "END_DT": last["NAV_DT"],
            "FUND_RETURN": fund_return,
            "DAYS_IN_PERIOD": (last["NAV_DT"] - actual_start_dt).days,
            "PERIOD": period
        })

    return pd.DataFrame(result)


def calc_index_return(df_idx: pd.DataFrame, start_dt, end_dt) -> float:
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
# 5. 主函数
# =========================================================
def main(prd_codes, index_code, custom_start_date=None, custom_end_date=None):
    # 验证日期格式
    validate_date(custom_start_date)
    validate_date(custom_end_date)

    # 获取数据
    df_nav = fetch_fin_prd_nav(prd_codes)
    df_index = fetch_index_quot(index_code)
    df_base_info = fetch_pty_prd_base_info(prd_codes)

    # 填充产品类型
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

    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 获取产品成立日
    prd_established_dates = {}
    for _, row in df_base_info.iterrows():
        if pd.notna(row["FOUND_DT"]):
            prd_established_dates[row["PRD_CODE"]] = row["FOUND_DT"]

    # 确定计算的结束日期
    if custom_end_date:
        end_dt = datetime.strptime(custom_end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    all_results = []

    # 计算预定义周期的收益
    for period in PREDEFINED_PERIODS:
        for prd_code in prd_codes:
            # 获取产品成立日
            fund_established_dt = prd_established_dates.get(prd_code)

            # 获取该产品的净值数据
            df_prd_nav = df_nav[df_nav["PRD_CODE"] == prd_code]
            if fund_established_dt:
                df_prd_nav = df_prd_nav[df_prd_nav["NAV_DT"] >= fund_established_dt]
            
            # 如果该产品没有数据，跳过
            if df_prd_nav.empty:
                continue
            
            # ========== 关键修改：每个产品使用自己的最新日期作为基准日 ==========
            product_latest_dt = df_prd_nav["NAV_DT"].max()
            
            if custom_end_date:
                custom_end_dt_obj = datetime.strptime(custom_end_date, "%Y-%m-%d")
                end_dt = min(product_latest_dt, custom_end_dt_obj)
            else:
                end_dt = product_latest_dt
            # ============================================================

            # 获取周期的开始和结束日期
            start_dt, _ = get_period_dates(end_dt, period, fund_established_dt)

            # 计算基金收益
            df_fund = calc_fund_return(df_prd_nav, start_dt, end_dt, period, fund_established_dt)

            if not df_fund.empty:
                # 计算指数收益
                fund_start_dt_used = df_fund.iloc[0]["START_DT"]
                fund_end_dt_used = df_fund.iloc[0]["END_DT"]

                index_return = calc_index_return(df_index, fund_start_dt_used, fund_end_dt_used)

                # 添加指数收益
                df_fund["INDEX_RETURN"] = index_return

                all_results.append(df_fund)

    # 计算自定义时间区间的收益
    if custom_start_date and custom_end_date:
        custom_start_dt = datetime.strptime(custom_start_date, "%Y-%m-%d")
        custom_end_dt_limit = datetime.strptime(custom_end_date, "%Y-%m-%d")

        for prd_code in prd_codes:
            # 获取该产品的净值数据
            df_prd_nav = df_nav[df_nav["PRD_CODE"] == prd_code]
            fund_established_dt = prd_established_dates.get(prd_code)
            if fund_established_dt:
                df_prd_nav = df_prd_nav[df_prd_nav["NAV_DT"] >= fund_established_dt]
            
            if df_prd_nav.empty:
                continue
            
            # 每个产品使用自己的最新日期，但不能超过 custom_end_date
            product_latest_dt = df_prd_nav["NAV_DT"].max()
            actual_end_dt = min(product_latest_dt, custom_end_dt_limit)
            
            # 如果产品最新日期早于 custom_start_date，跳过
            if product_latest_dt < custom_start_dt:
                continue

            df_fund = calc_fund_return(df_prd_nav, custom_start_dt, actual_end_dt,
                                       f"自定义区间({custom_start_date}~{custom_end_date})",
                                       fund_established_dt)

            if not df_fund.empty:
                # 计算指数收益
                fund_start_dt_used = df_fund.iloc[0]["START_DT"]
                fund_end_dt_used = df_fund.iloc[0]["END_DT"]

                index_return = calc_index_return(df_index, fund_start_dt_used, fund_end_dt_used)

                # 添加指数收益
                df_fund["INDEX_RETURN"] = index_return

                all_results.append(df_fund)

    # 合并结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # 重命列为中文
        final_df = final_df.rename(columns={
            "PRD_CODE": "产品代码",
            "PRD_TYP": "产品类型",
            "PERIOD": "周期",
            "START_DT": "起始日期",
            "END_DT": "结束日期",
            "DAYS_IN_PERIOD": "区间天数",
            "FUND_RETURN": "基金收益",
            "INDEX_RETURN": "指数收益"
        })

        # 格式化结果
        final_df["基金收益"] = (final_df["基金收益"] * 100).round(2).astype(str) + "%"
        final_df["指数收益"] = final_df["指数收益"].apply(lambda x: f"{x * 100:.2f}%" if x is not None else "-")

        # 调整列顺序
        final_df = final_df[
            ["产品代码", "产品类型", "周期", "起始日期", "结束日期", "区间天数", "基金收益", "指数收益"]]

        # 导出结果
        output_file = "基金表现对比-区间收益计算结果.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n计算完成！结果已保存至：{output_file}")
        print("\n计算结果预览：")
        print(final_df)
    else:
        print("\n未找到符合条件的数据！")


# =========================================================
# 6. 示例调用
# =========================================================
if __name__ == "__main__":
    # 示例参数
    # 产品代码列表（2-5个）
    prd_codes = ["1011", "1012", "1013", "1022", "1029"]
    # 市场指数代码
    index_code = "000300.IDX.CSIDX"
    # 自定义时间区间（可选）
    custom_start_date = "2021-12-31"
    custom_end_date = "2023-06-30"

    # 调用主函数
    main(prd_codes, index_code, custom_start_date, custom_end_date)
