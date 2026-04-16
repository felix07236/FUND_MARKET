import oracledb
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

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
def fetch_fin_prd_nav(prd_codes: list) -> pd.DataFrame:
    placeholders = ','.join([':' + str(i) for i in range(len(prd_codes))])
    sql = f"""
          SELECT PRD_CODE, \
                 NAV_DT, \
                 UNIT_NVAL, 
                 AGGR_UNIT_NVAL
          FROM DATA_MART_04.FIN_PRD_NAV \
          WHERE PRD_CODE IN ({placeholders})
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params=prd_codes)

    df["NAV_DT"] = pd.to_datetime(df["NAV_DT"].astype(str), format="%Y%m%d")
    return df


# --------------------------------------------------
# 3. 获取指数行情
# --------------------------------------------------
def fetch_index_quote(index_code) -> pd.DataFrame:
    sql = f"""
          SELECT TRD_DT, \
                 CLS_PRC
          FROM DATA_MART_04.VAR_SECU_DQUOT
          WHERE SECU_ID = :index_code \
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params={':index_code': index_code})

    df["NAV_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
    df = df[["NAV_DT", "CLS_PRC"]].rename(columns={"CLS_PRC": "INDEX"})
    return df


# --------------------------------------------------
# 获取产品基础信息
# --------------------------------------------------
def fetch_pty_prd_base_info(prd_codes: list) -> pd.DataFrame:
    placeholders = ','.join([':' + str(i) for i in range(len(prd_codes))])
    sql = f"""
          SELECT PRD_CODE, \
                 FOUND_DT, \
                 PRD_NAME, \
                 PRD_FULL_NAME, \
                 PRD_TYP
          FROM DATA_MART_04.PTY_PRD_BASE_INFO \
          WHERE PRD_CODE IN ({placeholders})
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params=prd_codes)

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
            # 优先使用累计净值
            if pd.notna(sub_start.iloc[0].get("AGGR_UNIT_NVAL")):
                return sub_start.iloc[0]["AGGR_UNIT_NVAL"]
            else:
                return sub_start.iloc[0]["UNIT_NVAL"]
        else:
            # 该日期没有数据，找成立日后的第一个净值日
            sub_after = prd_df[prd_df["NAV_DT"] >= theory_start_dt].sort_values("NAV_DT")
            if len(sub_after) > 0:
                # 优先使用累计净值
                if pd.notna(sub_after.iloc[0].get("AGGR_UNIT_NVAL")):
                    return sub_after.iloc[0]["AGGR_UNIT_NVAL"]
                else:
                    return sub_after.iloc[0]["UNIT_NVAL"]
            else:
                return 1.0


def get_nav_end(prd_df: pd.DataFrame, end_dt: pd.Timestamp) -> float:
    """获取期末净值"""
    sub_end = prd_df[prd_df["NAV_DT"] == end_dt]
    if len(sub_end) > 0:
        # 优先使用累计净值
        if pd.notna(sub_end.iloc[0].get("AGGR_UNIT_NVAL")):
            return sub_end.iloc[0]["AGGR_UNIT_NVAL"]
        else:
            return sub_end.iloc[0]["UNIT_NVAL"]
    else:
        # 优先使用累计净值
        if pd.notna(prd_df.iloc[-1].get("AGGR_UNIT_NVAL")):
            return prd_df.iloc[-1]["AGGR_UNIT_NVAL"]
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
        index_df: pd.DataFrame
) -> tuple:
    if period == "成立以来":
        # 成立以来：使用指数最早数据
        start_dt = index_df["NAV_DT"].min()
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
# 6. 计算各年度最大回撤
# --------------------------------------------------
def calc_annual_max_drawdown(
        prd_df: pd.DataFrame,
        index_df: pd.DataFrame,
        df_base_info: pd.DataFrame = None
) -> pd.DataFrame:
    prd_df = prd_df.sort_values("NAV_DT").copy()
    end_dt = prd_df["NAV_DT"].max()

    # ========== 获取产品真实成立日（从基础信息表） ==========
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)
    start_year = fund_established_dt.year
    end_year = end_dt.year

    result_records = []

    for year in range(start_year, end_year + 1):
        # 计算该年度的起始和结束日期
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)

        # 实际起始日期不能早于成立日
        actual_start = max(year_start, fund_established_dt)
        # 实际结束日期不能晚于当前日期
        actual_end = min(year_end, end_dt)

        # 筛选该年度的数据
        sub = prd_df[(prd_df["NAV_DT"] >= actual_start) & (prd_df["NAV_DT"] <= actual_end)]

        if len(sub) < 2:
            result_records.append({
                "年度": year,
                "产品最大回撤": 0.0,
                "指数最大回撤": 0.0,
                "超额最大回撤": 0.0
            })
            continue

        # 计算产品年度最大回撤
        # 使用年度第一天的净值作为基准
        nav_start = get_nav_start(prd_df, year_start, fund_established_dt)
        sub_prod = sub.copy()
        sub_prod["累计收益"] = sub_prod["UNIT_NVAL"] / nav_start
        prd_dd = max_drawdown(sub_prod["累计收益"])

        # 合并指数数据
        merged = pd.merge(sub, index_df, on="NAV_DT", how="inner")

        if merged.empty or len(merged) < 2:
            result_records.append({
                "年度": year,
                "产品最大回撤": prd_dd,
                "指数最大回撤": 0.0,
                "超额最大回撤": 0.0
            })
            continue

        merged = merged.sort_values("NAV_DT")
        merged["累计收益"] = merged["UNIT_NVAL"] / nav_start

        # 指数起始价格
        index_first_date = merged["NAV_DT"].min()
        idx_before = index_df[index_df["NAV_DT"] <= index_first_date].copy()

        if len(idx_before) > 0:
            idx_start_date = idx_before["NAV_DT"].max()
            index_start_price = idx_before.loc[idx_before["NAV_DT"] == idx_start_date, "INDEX"].values[0]
        else:
            index_start_price = merged.iloc[0]["INDEX"]

        merged["指数累计收益"] = merged["INDEX"] / index_start_price
        index_dd = max_drawdown(merged["指数累计收益"])

        # 超额净值
        merged["超额净值"] = (
                merged["累计收益"] / merged["指数累计收益"]
        ).replace([np.inf, -np.inf], np.nan)
        excess_dd = max_drawdown(merged["超额净值"])

        result_records.append({
            "年度": year,
            "产品最大回撤": prd_dd,
            "指数最大回撤": index_dd,
            "超额最大回撤": excess_dd
        })

    return pd.DataFrame(result_records)


# --------------------------------------------------
# 7. 产品指标（成立以来 + 各区间）
# --------------------------------------------------
def calc_product_metrics(
        prd_df: pd.DataFrame,
        index_df: pd.DataFrame,
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
        theory_start_dt, _ = get_period_dates_for_drawdown(end_dt, name, index_df)

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
# 9. 主函数
# --------------------------------------------------
def main(prd_codes, index_code):
    try:
        print("开始计算基金最大回撤指标...")
        print(f"产品代码: {prd_codes}")
        print(f"指数代码: {index_code}")
        
        # 验证输入参数
        if not prd_codes or len(prd_codes) < 2 or len(prd_codes) > 5:
            raise ValueError("产品代码数量必须在2-5个之间")
        
        # 获取数据
        df_nav = fetch_fin_prd_nav(prd_codes)
        df_index = fetch_index_quote(index_code)
        df_base_info = fetch_pty_prd_base_info(prd_codes)

        # 计算每个产品的指标
        all_metrics = []
        all_annual_dd = []
        
        for prd_code in prd_codes:
            print(f"\n4. 计算产品 {prd_code} 的指标...")
            
            # 获取该产品的数据
            prd_df = df_nav[df_nav["PRD_CODE"] == prd_code]
            base_info_for_prd = df_base_info[df_base_info["PRD_CODE"] == prd_code]
            
            if prd_df.empty:
                print(f"产品 {prd_code} 无数据，跳过")
                continue
            
            # 计算产品指标
            metrics = calc_product_metrics(prd_df, df_index, base_info_for_prd)
            if metrics:
                metrics["产品代码"] = prd_code
                metrics["计算基准日"] = prd_df["NAV_DT"].max().date()
                all_metrics.append(metrics)
            
            # 计算年度最大回撤
            annual_dd = calc_annual_max_drawdown(prd_df, df_index, base_info_for_prd)
            if not annual_dd.empty:
                annual_dd["产品代码"] = prd_code
                annual_dd["计算基准日"] = prd_df["NAV_DT"].max().date()
                all_annual_dd.append(annual_dd)
        
        # 输出结果
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            df_metrics = reorder_columns(df_metrics, ["产品代码", "计算基准日"])
            df_metrics = format_percentage(df_metrics, ["产品代码", "计算基准日"])
            
            output_file = "基金表现对比-最大回撤指标.csv"
            df_metrics.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n5. 计算完成！最大回撤指标已保存至：{output_file}")
        
        if all_annual_dd:
            df_annual = pd.concat(all_annual_dd, ignore_index=True)
            df_annual = reorder_columns(df_annual, ["产品代码", "计算基准日", "年度"])
            df_annual = format_percentage(df_annual, ["产品代码", "计算基准日", "年度"])
            
            output_file = "基金表现对比-年度最大回撤.csv"
            df_annual.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"年度最大回撤对比已保存至：{output_file}")
        
        if not all_metrics and not all_annual_dd:
            print("\n未找到符合条件的数据！")
            
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

# --------------------------------------------------
# 10. 示例调用
# --------------------------------------------------
if __name__ == "__main__":
    # 产品代码列表（2-5个）
    prd_codes = ["1011", "1012", "1022"]
    # 市场指数代码
    index_code = "000300.IDX.CSIDX"
    
    # 调用主函数
    main(prd_codes, index_code)
