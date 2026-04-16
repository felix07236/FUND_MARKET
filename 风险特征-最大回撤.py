import oracledb
import pandas as pd
import numpy as np
from datetime import date, timedelta


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
# 2. 获取产品净值数据
# --------------------------------------------------
def fetch_fin_prd_nav() -> pd.DataFrame:
    sql = """
          SELECT PRD_CODE, \
                 NAV_DT, \
                 UNIT_NVAL
          FROM DATA_MART_04.FIN_PRD_NAV \
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


# --------------------------------------------------
# 3. 获取沪深300 行情
# --------------------------------------------------
def fetch_hs300_quote() -> pd.DataFrame:
    sql = """
          SELECT TRD_DT, \
                 CLS_PRC
          FROM VAR_SECU_DQUOT
          WHERE SECU_ID = '000300.IDX.CSIDX' \
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

    df["NAV_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
    df = df[["NAV_DT", "CLS_PRC"]].rename(columns={"CLS_PRC": "HS300"})
    return df


# --------------------------------------------------
# 获取产品基础信息
# --------------------------------------------------
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


# --------------------------------------------------
# 4. 工具函数（防 inf 版本）
# --------------------------------------------------
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


def calc_return(start_nav: float, end_nav: float) -> float:
    if start_nav == 0 or pd.isna(start_nav):
        return 0.0
    return end_nav / start_nav - 1


# --------------------------------------------------
# 5. 公共函数
# --------------------------------------------------
def get_fund_established_dt(prd_df: pd.DataFrame, df_base_info: pd.DataFrame = None) -> pd.Timestamp:
    """获取产品真实成立日"""
    if df_base_info is not None and len(df_base_info) > 0:
        # 直接使用df_base_info，因为它已经是过滤后的单个产品的基础信息
        if pd.notna(df_base_info.iloc[0]["FOUND_DT"]):
            return df_base_info.iloc[0]["FOUND_DT"]
    return prd_df["NAV_DT"].min()


def get_nav_start(prd_df: pd.DataFrame, theory_start_dt: pd.Timestamp, fund_established_dt: pd.Timestamp) -> float:
    """获取起始净值"""
    if theory_start_dt < fund_established_dt:
        # 周期起始日在成立日之前 → 使用成立日的前一天，净值设为 1
        return 1.0
    else:
        # 周期起始日 >= 成立日 → 获取该日期的实际净值
        sub_start = prd_df[prd_df["NAV_DT"] == theory_start_dt]
        if len(sub_start) > 0:
            return sub_start.iloc[0]["UNIT_NVAL"]
        else:
            # 该日期没有数据，找成立日后的第一个净值日
            sub_after = prd_df[prd_df["NAV_DT"] >= theory_start_dt].sort_values("NAV_DT")
            if len(sub_after) > 0:
                return sub_after.iloc[0]["UNIT_NVAL"]
            else:
                return 1.0


def get_nav_end(prd_df: pd.DataFrame, end_dt: pd.Timestamp) -> float:
    """获取期末净值"""
    sub_end = prd_df[prd_df["NAV_DT"] == end_dt]
    if len(sub_end) > 0:
        return sub_end.iloc[0]["UNIT_NVAL"]
    else:
        return prd_df.iloc[-1]["UNIT_NVAL"]


def format_percentage(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """格式化百分比列"""
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else x
            )
    return df


def reorder_columns(df: pd.DataFrame, first_cols: list) -> pd.DataFrame:
    """重新排序列，将涨跌幅列放在一起，最大回撤列放在一起"""
    # 分离涨跌幅列和最大回撤列
    return_cols = []
    drawdown_cols = []

    for col in df.columns:
        if col not in first_cols:
            if "涨跌幅" in col:
                return_cols.append(col)
            elif "最大回撤" in col:
                drawdown_cols.append(col)

    # 构建新的列顺序：基础列 + 所有涨跌幅列 + 所有最大回撤列
    cols = first_cols + return_cols + drawdown_cols
    return df[cols]


# =========================================================
# 周期计算逻辑
# =========================================================
def get_period_dates_for_drawdown(
        end_dt: pd.Timestamp,
        period: str,
        hs300_df: pd.DataFrame
) -> tuple:
    if period == "成立以来":
        # 成立以来：使用指数最早数据
        start_dt = hs300_df["NAV_DT"].min()
        return start_dt, end_dt

    elif period == "今年以来":
        # 今年以来：上年末最后一天
        return pd.Timestamp(year=end_dt.year - 1, month=12, day=31), end_dt

    elif period.startswith("近") and period.endswith("年"):
        # 近 n 年：如"近 1 年"、"近 3 年"
        n_years_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
        chinese_num = period[1:-1]
        n_years = n_years_map.get(chinese_num)

        # 如果是阿拉伯数字，直接转换
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

    elif period.startswith("近") and period.endswith("月"):
        # 近 n 月：如"近 1 月"、"近 3 月"
        from dateutil.relativedelta import relativedelta
        n_months_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        chinese_num = period[1:-1]
        n_months = n_months_map.get(chinese_num)

        # 如果是阿拉伯数字，直接转换
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

    else:
        raise ValueError(f"未知周期：{period}")


# --------------------------------------------------
# 6. 产品指标（成立以来 + 各区间）
# --------------------------------------------------
def calc_product_metrics(
        prd_df: pd.DataFrame,
        hs300_df: pd.DataFrame,
        df_base_info: pd.DataFrame = None
) -> dict:
    prd_df = prd_df.sort_values("NAV_DT").copy()
    end_dt = prd_df["NAV_DT"].max()

    # ========== 获取产品真实成立日（从基础信息表） ==========
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    result = {}

    # ========== 定义所有周期 ==========
    periods = [
        "成立以来",
        "今年以来",
        "近1月",
        "近3月",
        "近6月",
        "近1年",
        "近2年",
        "近3年",
        "近5年"
    ]

    for name in periods:
        # ========== 计算周期起始日 ==========
        theory_start_dt, _ = get_period_dates_for_drawdown(end_dt, name, hs300_df)

        # ========== 获取起始净值 ==========
        # 直接使用get_nav_start函数，它会处理周期起始时间在成立日之前的情况
        nav_start = get_nav_start(prd_df, theory_start_dt, fund_established_dt)

        # ========== 获取期末净值 ==========
        nav_end = get_nav_end(prd_df, end_dt)

        if pd.isna(nav_start) or pd.isna(nav_end) or nav_start <= 0:
            result[f"{name}涨跌幅"] = 0.0
            result[f"{name}最大回撤"] = 0.0
        else:
            # 计算涨跌幅
            result[f"{name}涨跌幅"] = calc_return(nav_start, nav_end)

            # 计算该区间内的最大回撤
            # 当周期起始日在成立日之前时，使用成立日作为区间的起始时间
            actual_start_dt = max(theory_start_dt, fund_established_dt)
            sub = prd_df[(prd_df["NAV_DT"] >= actual_start_dt) & (prd_df["NAV_DT"] <= end_dt)]
            if len(sub) > 0:
                sub_with_cumret = sub.copy()
                sub_with_cumret["累计收益"] = sub_with_cumret["UNIT_NVAL"] / nav_start
                result[f"{name}最大回撤"] = max_drawdown(sub_with_cumret["累计收益"])
            else:
                result[f"{name}最大回撤"] = 0.0

    return result


# --------------------------------------------------
# 7. 区间维度（产品 / 沪深300 / 超额）
# --------------------------------------------------
def calc_interval_drawdown(
        prd_df: pd.DataFrame,
        hs300_df: pd.DataFrame,
        df_base_info: pd.DataFrame = None
) -> pd.DataFrame:
    prd_df = prd_df.sort_values("NAV_DT").copy()
    end_dt = prd_df["NAV_DT"].max()

    # ========== 获取产品真实成立日（从基础信息表） ==========
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    result_records = []

    periods = [
        "成立以来",
        "今年以来",
        "近1月",
        "近3月",
        "近6月",
        "近1年",
        "近2年",
        "近3年",
        "近5年"
    ]

    for name in periods:
        period_name = "今年以来" if name == "今年来" else name
        theory_start_dt, _ = get_period_dates_for_drawdown(end_dt, period_name, hs300_df)

        # ========== 获取起始净值 ==========
        nav_start = get_nav_start(prd_df, theory_start_dt, fund_established_dt)

        # ========== 获取期末净值 ==========
        nav_end = get_nav_end(prd_df, end_dt)

        if pd.isna(nav_start) or nav_start <= 0:
            result_records.append({
                "区间": name,
                "产品最大回撤": 0.0,
                "沪深300 最大回撤": 0.0,
                "超额最大回撤": 0.0
            })
            continue

        # ========== 获取区间内数据 ==========
        actual_start_dt = max(theory_start_dt, fund_established_dt)
        sub = prd_df[(prd_df["NAV_DT"] >= actual_start_dt) & (prd_df["NAV_DT"] <= end_dt)]

        if len(sub) < 2:
            result_records.append({
                "区间": name,
                "产品最大回撤": 0.0,
                "沪深300 最大回撤": 0.0,
                "超额最大回撤": 0.0
            })
            continue

        # ========== 产品最大回撤 ==========
        sub_prod = sub.copy()
        sub_prod["累计收益"] = sub_prod["UNIT_NVAL"] / nav_start
        prd_dd = max_drawdown(sub_prod["累计收益"])

        # ========== 合并沪深300 ==========
        merged = pd.merge(sub, hs300_df, on="NAV_DT", how="inner")

        if merged.empty or len(merged) < 2:
            result_records.append({
                "区间": name,
                "产品最大回撤": prd_dd,
                "沪深300 最大回撤": 0.0,
                "超额最大回撤": 0.0
            })
            continue

        merged = merged.sort_values("NAV_DT")

        # ==========  超额最大回撤 ==========
        hs300_start_price = merged.iloc[0]["HS300"]
        merged["沪深300 累计收益"] = merged["HS300"] / hs300_start_price
        hs300_dd = max_drawdown(merged["沪深300 累计收益"])
        # 从第二天开始计算超额收益（去掉第一天）
        if len(merged) < 2:
            excess_dd = 0.0
        else:
            # 取从第二天开始的数据
            merged_excess = merged.iloc[1:].copy()
            
            # 1. 产品日收益
            merged_excess["产品日收益"] = merged_excess["UNIT_NVAL"].pct_change()
            
            # 2. 基准日收益
            merged_excess["基准日收益"] = merged_excess["HS300"].pct_change()
            
            # 3. 超额收益
            merged_excess["超额收益"] = merged_excess["产品日收益"] - merged_excess["基准日收益"]
            
            # 4. 构造超额累计净值
            excess_cumulative = pd.Series(index=merged_excess.index, dtype=float)
            excess_cumulative.iloc[0] = 1.0
            
            if len(merged_excess) > 1:
                # 获取有效的超额收益（去除NaN）
                valid_excess_returns = merged_excess["超额收益"].iloc[1:].dropna()
                
                if len(valid_excess_returns) > 0:
                    # 计算累乘结果
                    cumprod_result = (1 + valid_excess_returns).cumprod()
                    
                    # 将结果赋值给对应的位置
                    excess_cumulative.iloc[1:len(cumprod_result) + 1] = cumprod_result.values
            
            # 5. 超额最大回撤
            excess_dd = max_drawdown(excess_cumulative)

        result_records.append({
            "区间": name,
            "产品最大回撤": prd_dd,
            "沪深300 最大回撤": hs300_dd,
            "超额最大回撤": excess_dd
        })

    return pd.DataFrame(result_records)


# --------------------------------------------------
# 8. 执行入口
# --------------------------------------------------
if __name__ == "__main__":

    prd_df = fetch_fin_prd_nav()
    hs300_df = fetch_hs300_quote()
    df_base_info = fetch_pty_prd_base_info()

    results = []
    for prd_code, g in prd_df.groupby("PRD_CODE"):
        # 获取该产品的基础信息
        base_info_for_prd = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        metrics = calc_product_metrics(g, hs300_df, base_info_for_prd)
        if not metrics:
            continue
        metrics["产品代码"] = prd_code
        metrics["计算基准日"] = g["NAV_DT"].max().date()
        results.append(metrics)

    if results:
        df1 = pd.DataFrame(results)
        df1 = reorder_columns(df1, ["产品代码", "计算基准日"])
        df1 = format_percentage(df1, ["产品代码", "计算基准日"])

        df1.to_csv(
            "最大回撤数据.csv",
            index=False,
            encoding="utf-8-sig"
        )

    all_dd = []
    for prd_code, g in prd_df.groupby("PRD_CODE"):
        base_info_for_prd = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        dd_df = calc_interval_drawdown(g, hs300_df, base_info_for_prd)
        if dd_df.empty:
            continue
        dd_df["产品代码"] = prd_code
        dd_df["计算基准日"] = g["NAV_DT"].max().date()
        all_dd.append(dd_df)

    if all_dd:
        df2 = pd.concat(all_dd, ignore_index=True)
        df2 = reorder_columns(df2, ["产品代码", "计算基准日", "区间"])
        df2 = format_percentage(df2, ["产品代码", "计算基准日", "区间"])

        df2.to_csv(
            "动态回撤对比.csv",
            index=False,
            encoding="utf-8-sig"
        )
