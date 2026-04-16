import oracledb
import pandas as pd
import numpy as np
import sys

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
# 2. 获取所有产品净值数据
# --------------------------------------------------
def fetch_all_fin_prd_nav() -> pd.DataFrame:
    sql = f"""
          SELECT PRD_CODE, \
                 NAV_DT, \
                 UNIT_NVAL, 
                 AGGR_UNIT_NVAL
          FROM DATA_MART_04.FIN_PRD_NAV
          """

    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn)

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
# 获取所有产品基础信息
# --------------------------------------------------
def fetch_all_pty_prd_base_info() -> pd.DataFrame:
    sql = f"""
          SELECT PRD_CODE, \
                 FOUND_DT, \
                 PRD_NAME, \
                 PRD_FULL_NAME, \
                 PRD_TYP
          FROM DATA_MART_04.PTY_PRD_BASE_INFO
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
# 4. 工具函数
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
    """重新排序列"""
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


# --------------------------------------------------
# 6. 计算区间最大回撤和涨跌幅
# --------------------------------------------------
def calc_interval_metrics(
        prd_df: pd.DataFrame,
        index_df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        df_base_info: pd.DataFrame = None
) -> dict:
    prd_df = prd_df.sort_values("NAV_DT").copy()

    # ========== 获取产品真实成立日（从基础信息表） ==========
    fund_established_dt = get_fund_established_dt(prd_df, df_base_info)

    # ========== 获取起始净值（使用原始开始日期） ==========
    nav_start = get_nav_start(prd_df, start_dt, fund_established_dt)

    # ========== 获取期末净值 ==========
    nav_end = get_nav_end(prd_df, end_dt)

    result = {
        "产品涨跌幅": 0.0,
        "产品最大回撤": 0.0,
        "指数涨跌幅": 0.0,
        "指数最大回撤": 0.0,
        "超额最大回撤": 0.0
    }

    if pd.isna(nav_start) or pd.isna(nav_end) or nav_start <= 0:
        return result

    # ========== 计算产品涨跌幅（使用原始开始日期） ==========
    product_return = calc_return(nav_start, nav_end)
    result["产品涨跌幅"] = product_return

    # ========== 单独获取指数起始价格（使用原始开始日期）用于计算指数涨跌幅 ==========
    idx_on_start_dt = index_df[index_df["NAV_DT"] <= start_dt].copy()
    if len(idx_on_start_dt) > 0:
        idx_start_date_for_return = idx_on_start_dt["NAV_DT"].max()
        index_start_price_for_return = idx_on_start_dt.loc[idx_on_start_dt["NAV_DT"] == idx_start_date_for_return, "INDEX"].values[0]
    else:
        index_start_price_for_return = None

    # ========== 产品最大回撤 ==========
    product_actual_start_dt = max(start_dt + pd.Timedelta(days=1), fund_established_dt)
    sub_product = prd_df[(prd_df["NAV_DT"] >= product_actual_start_dt) & (prd_df["NAV_DT"] <= end_dt)]

    if len(sub_product) > 0:
        sub_with_cumret = sub_product.copy()
        sub_with_cumret["累计收益"] = sub_with_cumret["UNIT_NVAL"] / nav_start
        product_drawdown = max_drawdown(sub_with_cumret["累计收益"])
        result["产品最大回撤"] = product_drawdown
    else:
        return result

    # ========== 合并指数数据 ==========
    merged = pd.merge(sub_product, index_df, on="NAV_DT", how="inner")

    if merged.empty or len(merged) < 2:
        return result

    merged = merged.sort_values("NAV_DT")
    merged["累计收益"] = merged["UNIT_NVAL"] / nav_start

    # 指数结束价格
    index_end_price = merged.iloc[-1]["INDEX"]

    # ========== 计算指数涨跌幅（使用原始开始日期的指数价格） ==========
    if index_start_price_for_return is not None:
        index_return = calc_return(index_start_price_for_return, index_end_price)
    else:
        index_return = 0.0
    result["指数涨跌幅"] = index_return

    # ========== 计算指数最大回撤 ==========
    index_start_price_for_drawdown = merged.iloc[0]["INDEX"]
    merged["指数累计收益"] = merged["INDEX"] / index_start_price_for_drawdown
    index_drawdown = max_drawdown(merged["指数累计收益"])
    result["指数最大回撤"] = index_drawdown

    # ========== 计算超额最大回撤 ==========
    if len(merged) < 3:
        excess_drawdown = 0.0
    else:
        # 从第二天开始计算
        merged_excess = merged.iloc[1:].copy()
        
        if len(merged_excess) < 2:
            excess_drawdown = 0.0
        else:
            # 产品日收益
            merged_excess["产品日收益"] = merged_excess["UNIT_NVAL"].pct_change()
            # 基准日收益
            merged_excess["基准日收益"] = merged_excess["INDEX"].pct_change()
            # 超额收益
            merged_excess["超额收益"] = merged_excess["产品日收益"] - merged_excess["基准日收益"]
            
            # 构造超额累计净值
            excess_cumulative = pd.Series(index=merged_excess.index, dtype=float)
            excess_cumulative.iloc[0] = 1.0
            
            if len(merged_excess) > 1:
                # 获取有效的超额收益
                valid_excess_returns = merged_excess["超额收益"].iloc[1:].dropna()
                
                if len(valid_excess_returns) > 0:
                    # 计算累乘结果
                    cumprod_result = (1 + valid_excess_returns).cumprod()
                    # 将结果赋值给对应的位置
                    excess_cumulative.iloc[1:len(cumprod_result) + 1] = cumprod_result.values
            
            # 超额最大回撤
            excess_drawdown = max_drawdown(excess_cumulative)
    
    result["超额最大回撤"] = excess_drawdown

    return result


# --------------------------------------------------
# 7. 主函数
# --------------------------------------------------
def main(index_code, start_date, end_date):
    try:
        print("开始计算所有基金区间回撤指标...")
        print(f"指数代码: {index_code}")
        print(f"时间区间: {start_date} 至 {end_date}")
        
        # 转换日期格式
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # ========== 开始日期往前推一天 ==========
        from datetime import timedelta
        actual_start_dt = start_dt - timedelta(days=1)
        
        # 验证日期
        if actual_start_dt >= end_dt:
            raise ValueError("开始日期必须早于结束日期")
        
        # 获取数据
        df_nav = fetch_all_fin_prd_nav()
        df_index = fetch_index_quote(index_code)
        df_base_info = fetch_all_pty_prd_base_info()
        # 获取所有唯一的产品代码
        prd_codes = df_nav["PRD_CODE"].unique().tolist()
        
        # 计算每个产品的指标
        all_metrics = []

        for prd_code in prd_codes:
            # 获取该产品的数据
            prd_df = df_nav[df_nav["PRD_CODE"] == prd_code]
            base_info_for_prd = df_base_info[df_base_info["PRD_CODE"] == prd_code]
            
            if prd_df.empty:
                continue
            
            # 计算区间指标（使用调整后的开始日期）
            metrics = calc_interval_metrics(prd_df, df_index, actual_start_dt, end_dt, base_info_for_prd)
            if metrics:
                metrics["产品代码"] = prd_code
                metrics["开始日期"] = start_date
                metrics["结束日期"] = end_date
                metrics["计算基准日"] = prd_df["NAV_DT"].max().date()
                all_metrics.append(metrics)
        
        # 输出结果
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            df_metrics = reorder_columns(df_metrics, ["产品代码", "开始日期", "结束日期", "计算基准日"])
            df_metrics = format_percentage(df_metrics, ["产品代码", "开始日期", "结束日期", "计算基准日"])
            
            output_file = f"区间回撤指标.csv"
            df_metrics.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n计算完成！区间回撤指标已保存至：{output_file}")
            print(f"共计算了 {len(all_metrics)} 个产品的指标")
        else:
            print("\n未找到符合条件的数据！")
            
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


# --------------------------------------------------
# 8. 示例调用
# --------------------------------------------------
if __name__ == "__main__":
    # 从命令行获取参数
    if len(sys.argv) >= 4:
        index_code = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
    else:
        # 默认参数
        print("未提供命令行参数，使用默认参数")
        print("使用方法: python 风险特征-最大回撤-区间回撤.py <指数代码> <开始日期> <结束日期>")
        print("使用示例: python 风险特征-最大回撤-区间回撤.py 000300.IDX.CSIDX 2022-12-30 2023-06-30")
        print("-" * 80)
        
        # 市场指数代码
        index_code = "000300.IDX.CSIDX"
        # 开始日期
        start_date = "2022-12-30"
        # 结束日期
        end_date = "2023-06-30"
    
    # 调用主函数
    main(index_code, start_date, end_date)
