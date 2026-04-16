import oracledb
import pandas as pd
from datetime import datetime, timedelta
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
          SELECT SECU_ID, TRD_DT, CLS_PRC, PREV_CLS_PRC
          FROM DATA_MART_04.VAR_SECU_DQUOT
          WHERE SECU_ID = :index_code
          """
    with get_oracle_conn() as conn:
        df = pd.read_sql(sql, conn, params={':index_code': index_code})

    df["TRD_DT"] = pd.to_datetime(df["TRD_DT"].astype(str), format="%Y%m%d")
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

# =========================================================
# 3. 收益计算
# =========================================================
def calc_period_return(df_nav, start_dt, end_dt, period_type):
    """计算指定时间区间的收益"""
    df = df_nav.copy()
    mask = (df["NAV_DT"] >= start_dt) & (df["NAV_DT"] <= end_dt)
    df = df[mask]

    if df.empty:
        return None

    df = df.sort_values("NAV_DT")
    first = df.iloc[0]
    last = df.iloc[-1]

    # 使用累计净值优先，为空则用单位净值
    nav_start = (
        first["AGGR_UNIT_NVAL"]
        if pd.notna(first["AGGR_UNIT_NVAL"])
        else first["UNIT_NVAL"]
    )
    nav_end = (
        last["AGGR_UNIT_NVAL"]
        if pd.notna(last["AGGR_UNIT_NVAL"])
        else last["UNIT_NVAL"]
    )

    if pd.isna(nav_start) or pd.isna(nav_end):
        return None

    return nav_end / nav_start - 1

def calc_index_period_return(df_index, start_dt, end_dt):
    """计算指数在指定时间区间的收益"""
    df = df_index[
        (df_index["TRD_DT"] >= start_dt) &
        (df_index["TRD_DT"] <= end_dt)
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
# 4. 年度收益计算
# =========================================================
def calculate_annual_returns(df_nav, df_index, fund_established_dt, prd_code, prd_typ):
    """计算基金成立以来各完整会计年度的收益"""
    results = []
    current_year = datetime.now().year
    
    # 从成立年份的下一年开始计算完整会计年度
    start_year = fund_established_dt.year + 1
    
    for year in range(start_year, current_year + 1):
        # 会计年度：上一年12月31日到当年12月31日
        start_dt = datetime(year - 1, 12, 31)
        end_dt = datetime(year, 12, 31)
        
        # 计算基金收益
        fund_return = calc_period_return(df_nav, start_dt, end_dt, "年度")
        
        if fund_return is not None:
            # 计算指数收益
            index_return = calc_index_period_return(df_index, start_dt, end_dt)
            
            results.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": prd_typ,
                "PERIOD_TYPE": "年度",
                "PERIOD": f"{year}年",
                "START_DT": start_dt,
                "END_DT": end_dt,
                "FUND_RETURN": fund_return,
                "INDEX_RETURN": index_return
            })
    
    return results

# =========================================================
# 5. 月度收益计算
# =========================================================
def calculate_monthly_returns(df_nav, df_index, prd_code, prd_typ, product_latest_dt):
    """计算最近12个月的月度收益（基于产品最新日期）"""
    results = []
    
    # 使用产品最新日期作为基准日
    end_date = product_latest_dt
    
    # 计算最近12个月
    for i in range(11, -1, -1):
        # 计算当月的开始和结束日期
        current_date = end_date - relativedelta(months=i)
        
        # 当月最后一天
        if current_date.month == 12:
            end_dt = datetime(current_date.year, 12, 31)
        else:
            end_dt = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
        
        # 上月最后一天作为起始日（正确计算上月最后一天）
        if current_date.month == 1:
            # 如果当前是1月，上月是上一年的12月
            start_dt = datetime(current_date.year - 1, 12, 31)
        else:
            # 否则上月最后一天 = 当月1日 - 1天
            start_dt = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)
        
        # 计算基金收益
        fund_return = calc_period_return(df_nav, start_dt, end_dt, "月度")
        
        if fund_return is not None:
            # 计算指数收益
            index_return = calc_index_period_return(df_index, start_dt, end_dt)
            
            results.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": prd_typ,
                "PERIOD_TYPE": "月度",
                "PERIOD": f"{current_date.year}-{current_date.month:02d}",
                "START_DT": start_dt,
                "END_DT": end_dt,
                "FUND_RETURN": fund_return,
                "INDEX_RETURN": index_return
            })
    
    return results

# =========================================================
# 6. 主函数
# =========================================================
def main(prd_codes, index_code):
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
    prd_types = {}
    for _, row in df_base_info.iterrows():
        if pd.notna(row["FOUND_DT"]):
            prd_established_dates[row["PRD_CODE"]] = row["FOUND_DT"]
            prd_types[row["PRD_CODE"]] = row.get("PRD_TYP", "")

    all_results = []

    # 计算每个产品的年度和月度收益
    for prd_code in prd_codes:
        fund_established_dt = prd_established_dates.get(prd_code)
        prd_typ = prd_types.get(prd_code, "")
        
        if not fund_established_dt:
            print(f"产品 {prd_code} 未找到成立日期，跳过计算")
            continue
        
        # 筛选该产品的净值数据
        df_prd_nav = df_nav[df_nav["PRD_CODE"] == prd_code]
        # 只使用成立日之后的数据
        df_prd_nav = df_prd_nav[df_prd_nav["NAV_DT"] >= fund_established_dt]
        
        if df_prd_nav.empty:
            print(f"产品 {prd_code} 无净值数据，跳过计算")
            continue
        
        # ========== 获取该产品的最新日期作为基准日 ==========
        product_latest_dt = df_prd_nav["NAV_DT"].max()
        # ================================================
        
        # 计算年度收益
        annual_results = calculate_annual_returns(
            df_prd_nav, df_index, fund_established_dt, prd_code, prd_typ
        )
        all_results.extend(annual_results)
        
        # 计算月度收益（传入产品最新日期）
        monthly_results = calculate_monthly_returns(
            df_prd_nav, df_index, prd_code, prd_typ, product_latest_dt
        )
        all_results.extend(monthly_results)

    # 处理结果
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # 格式化收益数据
        final_df["FUND_RETURN"] = (final_df["FUND_RETURN"] * 100).round(2).astype(str) + "%"
        final_df["INDEX_RETURN"] = final_df["INDEX_RETURN"].apply(lambda x: f"{x*100:.2f}%" if x is not None else "-")
        
        # 调整列顺序
        final_df = final_df[[
            "PRD_CODE", "PRD_TYP", "PERIOD_TYPE", "PERIOD", 
            "START_DT", "END_DT", "FUND_RETURN", "INDEX_RETURN"
        ]]
        
        # 导出结果
        output_file = "基金表现对比-年月度收益结果.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n计算完成！结果已保存至：{output_file}")
        print("\n计算结果预览：")
        print(final_df)
    else:
        print("\n未找到符合条件的数据！")

# =========================================================
# 7. 示例调用
# =========================================================
if __name__ == "__main__":
    # 示例参数
    # 产品代码列表（2-5个）
    prd_codes = ["1011", "1012", "1022"]
    # 市场指数代码
    index_code = "000300.IDX.CSIDX"
    
    # 调用主函数
    main(prd_codes, index_code)
