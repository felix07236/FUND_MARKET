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
# 3. 月度收益计算
# =========================================================
def calc_monthly_return(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算月度收益（基金和沪深300）
    """
    results = []

    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 数据清洗：过滤掉成立日之前的脏数据
        grp_clean = grp[grp["NAV_DT"] >= fund_established_dt].copy()

        if grp_clean.empty:
            continue

        # 按月份分组
        grp_clean["MONTH"] = grp_clean["NAV_DT"].dt.to_period("M")

        for month, month_data in grp_clean.groupby("MONTH"):
            month_data = month_data.sort_values("NAV_DT")

            if len(month_data) < 2:
                continue

            # 计算基金月度收益
            start_nav = month_data.iloc[0]["AGGR_UNIT_NVAL"]
            end_nav = month_data.iloc[-1]["AGGR_UNIT_NVAL"]

            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
                continue

            fund_return = (end_nav - start_nav) / start_nav

            # 计算沪深300月度收益
            month_start = month_data.iloc[0]["NAV_DT"]
            month_end = month_data.iloc[-1]["NAV_DT"]

            idx_data = df_idx[
                (df_idx["TRD_DT"] >= month_start) &
                (df_idx["TRD_DT"] <= month_end)
            ].copy()

            if len(idx_data) < 2:
                continue

            idx_data = idx_data.sort_values("TRD_DT")
            idx_start_price = idx_data.iloc[0]["CLS_PRC"]
            idx_end_price = idx_data.iloc[-1]["CLS_PRC"]

            if pd.isna(idx_start_price) or pd.isna(idx_end_price) or idx_start_price == 0:
                continue

            idx_return = (idx_end_price - idx_start_price) / idx_start_price

            # 计算超额收益（几何）
            excess_return = (1 + fund_return) / (1 + idx_return) - 1 if (1 + idx_return) != 0 else 0

            results.append({
                "PRD_CODE": prd_code,
                "PRD_TYP": month_data.iloc[0]["PRD_TYP"],
                "MONTH": str(month),
                "MONTH_START": month_start,
                "MONTH_END": month_end,
                "FUND_RETURN": fund_return,
                "IDX_RETURN": idx_return,
                "EXCESS_RETURN": excess_return
            })

    return pd.DataFrame(results)


# =========================================================
# 4. 计算同类平均收益
# =========================================================
def calc_peer_average(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类平均收益
    """
    peer_avg = (
        df_monthly.groupby(["PRD_TYP", "MONTH"])["FUND_RETURN"]
        .mean()
        .reset_index(name="PEER_AVG_RETURN")
    )
    return df_monthly.merge(peer_avg, on=["PRD_TYP", "MONTH"], how="left")


# =========================================================
# 5. 计算四分位排名
# =========================================================
def calc_quarterly_rank(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    计算四分位排名
    """
    df = df_monthly.copy()
    
    for (prd_typ, month), group in df.groupby(["PRD_TYP", "MONTH"]):
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
# 6. 计算本年度收益
# =========================================================
def calc_year_to_date_return(df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算本年度收益
    """
    results = []
    current_year = datetime.now().year
    
    for prd_code, grp in df_nav.groupby("PRD_CODE"):
        # 从基础信息表获取成立日
        base_info = df_base_info[df_base_info["PRD_CODE"] == prd_code]
        if len(base_info) == 0 or pd.isna(base_info.iloc[0]["FOUND_DT"]):
            continue

        fund_established_dt = base_info.iloc[0]["FOUND_DT"]

        # 数据清洗：过滤掉成立日之前的脏数据
        grp_clean = grp[grp["NAV_DT"] >= fund_established_dt].copy()

        if grp_clean.empty:
            continue

        # 筛选今年的数据
        year_start = datetime(current_year, 1, 1)
        year_data = grp_clean[grp_clean["NAV_DT"] >= year_start].copy()
        
        if len(year_data) < 2:
            continue
        
        year_data = year_data.sort_values("NAV_DT")
        start_nav = year_data.iloc[0]["AGGR_UNIT_NVAL"]
        end_nav = year_data.iloc[-1]["AGGR_UNIT_NVAL"]
        
        if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
            continue
        
        ytd_return = (end_nav - start_nav) / start_nav
        
        results.append({
            "PRD_CODE": prd_code,
            "YTD_RETURN": ytd_return
        })
    
    return pd.DataFrame(results)


# =========================================================
# 7. 计算月胜率（12个月中收益为正概率）
# =========================================================
def calc_monthly_win_rate(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    计算月胜率（12个月中收益为正概率）
    """
    results = []
    
    for (prd_code, prd_typ), group in df_monthly.groupby(["PRD_CODE", "PRD_TYP"]):
        # 最近12个月的数据
        recent_12_months = group.sort_values("MONTH", ascending=False).head(12)
        total_months = len(recent_12_months)
        
        if total_months == 0:
            continue
        
        # 统计收益为正的月份数
        positive_months = len(recent_12_months[recent_12_months["FUND_RETURN"] > 0])
        win_rate = positive_months / total_months
        
        results.append({
            "PRD_CODE": prd_code,
            "PRD_TYP": prd_typ,
            "MONTHLY_WIN_RATE": win_rate
        })
    
    return pd.DataFrame(results)


# =========================================================
# 8. 主流程
# =========================================================
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_hs300 = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()
    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 计算月度收益
    df_monthly = calc_monthly_return(df_nav, df_hs300, df_base_info)

    if df_monthly.empty:
        print("没有可用的月度收益数据")
        exit()

    # 计算同类平均收益
    df_monthly = calc_peer_average(df_monthly)

    # 计算四分位排名
    df_monthly = calc_quarterly_rank(df_monthly)

    # 计算本年度收益
    df_ytd = calc_year_to_date_return(df_nav, df_base_info)

    # 计算月胜率
    df_win_rate = calc_monthly_win_rate(df_monthly)

    # 合并数据
    df_result = df_monthly.merge(df_ytd, on="PRD_CODE", how="left")
    df_result = df_result.merge(df_win_rate, on=["PRD_CODE", "PRD_TYP"], how="left")

    # 格式化输出
    for col in ["FUND_RETURN", "IDX_RETURN", "EXCESS_RETURN", "PEER_AVG_RETURN", "YTD_RETURN"]:
        if col in df_result.columns:
            df_result[col] = df_result[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    if "MONTHLY_WIN_RATE" in df_result.columns:
        df_result["MONTHLY_WIN_RATE"] = df_result["MONTHLY_WIN_RATE"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    # 创建产品名称映射
    prd_name_map = df_base_info.set_index("PRD_CODE")["PRD_NAME"].to_dict()
    
    # 输出结果
    output_file = "区间收益明细结果.txt"
    output_lines = []
    output_lines.append("=" * 200)
    output_lines.append("📊 月度区间收益明细（BI效果）")
    output_lines.append(f"✅ 共统计产品数量：{len(df_result['PRD_CODE'].unique())}")
    output_lines.append("=" * 200)
    output_lines.append("")

    # 按产品分组输出
    for prd_code in df_result["PRD_CODE"].unique():
        prd_data = df_result[df_result["PRD_CODE"] == prd_code]
        prd_typ = prd_data.iloc[0]["PRD_TYP"]
        prd_name = prd_name_map.get(prd_code, prd_code)
        
        output_lines.append("-" * 200)
        output_lines.append(f"产品代码: {prd_code} ({prd_name}) | 产品类型: {prd_typ}")
        output_lines.append("-" * 200)
        
        # 按年份分组
        prd_data["YEAR"] = prd_data["MONTH"].str[:4]
        
        for year in sorted(prd_data["YEAR"].unique(), reverse=True):
            year_data = prd_data[prd_data["YEAR"] == year]
            year_data = year_data.sort_values("MONTH")
            
            # 准备月份数据
            month_data = {}
            for _, row in year_data.iterrows():
                month = row["MONTH"][5:7]  # 提取月份
                month_data[month] = {
                    "FUND_RETURN": row["FUND_RETURN"],
                    "IDX_RETURN": row["IDX_RETURN"],
                    "EXCESS_RETURN": row["EXCESS_RETURN"],
                    "PEER_AVG_RETURN": row["PEER_AVG_RETURN"],
                    "QUARTER_RANK": row["QUARTER_RANK"]
                }
            
            # 输出年份标题
            output_lines.append("")
            output_lines.append(f"年份: {year}")
            output_lines.append("-" * 200)
            
            # 输出表头
            header = ["简称"]
            for m in range(1, 13):
                header.append(f"{m}月")
            header.append("年份统计")
            header.append("月胜率")
            output_lines.append("|".join([f"{h:12s}" for h in header]))
            output_lines.append("-" * 200)
            
            # 输出基金收益行
            fund_display_name = f"{prd_code}({prd_name[:8]})"
            fund_row = [fund_display_name]
            for m in range(1, 13):
                month_str = f"{m:02d}"
                if month_str in month_data:
                    fund_row.append(month_data[month_str]["FUND_RETURN"])
                else:
                    fund_row.append("--")
            fund_row.append(year_data.iloc[-1]["YTD_RETURN"] if "YTD_RETURN" in year_data.columns else "--")
            fund_row.append(year_data.iloc[-1]["MONTHLY_WIN_RATE"] if "MONTHLY_WIN_RATE" in year_data.columns else "--")
            output_lines.append("|".join([f"{str(x):12s}" for x in fund_row]))
            
            # 输出沪深300行
            idx_row = ["沪深300"]
            for m in range(1, 13):
                month_str = f"{m:02d}"
                if month_str in month_data:
                    idx_row.append(month_data[month_str]["IDX_RETURN"])
                else:
                    idx_row.append("--")
            idx_row.append("--")
            idx_row.append("--")
            output_lines.append("|".join([f"{str(x):12s}" for x in idx_row]))
            
            # 输出超额收益行
            excess_row = ["超额收益(几何)"]
            for m in range(1, 13):
                month_str = f"{m:02d}"
                if month_str in month_data:
                    excess_row.append(month_data[month_str]["EXCESS_RETURN"])
                else:
                    excess_row.append("--")
            excess_row.append("--")
            excess_row.append("--")
            output_lines.append("|".join([f"{str(x):12s}" for x in excess_row]))
            
            # 输出同类平均行
            peer_row = ["同类平均"]
            for m in range(1, 13):
                month_str = f"{m:02d}"
                if month_str in month_data:
                    peer_row.append(month_data[month_str]["PEER_AVG_RETURN"])
                else:
                    peer_row.append("--")
            peer_row.append("--")
            peer_row.append("--")
            output_lines.append("|".join([f"{str(x):12s}" for x in peer_row]))
            
            # 输出四分位排名行
            quarter_row = ["四分位"]
            for m in range(1, 13):
                month_str = f"{m:02d}"
                if month_str in month_data:
                    quarter_row.append(month_data[month_str]["QUARTER_RANK"])
                else:
                    quarter_row.append("--")
            quarter_row.append("--")
            quarter_row.append("--")
            output_lines.append("|".join([f"{str(x):12s}" for x in quarter_row]))
            
            output_lines.append("")

    output_lines.append("=" * 200)
    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n结果已保存至：{output_file}")
