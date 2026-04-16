import oracledb
import pandas as pd

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
# 3. 季度收益计算
# =========================================================
def calc_quarterly_return(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算季度收益（基金和沪深300）
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

        # 按季度分组
        grp_clean["QUARTER"] = grp_clean["NAV_DT"].dt.to_period("Q")

        for quarter, quarter_data in grp_clean.groupby("QUARTER"):
            quarter_data = quarter_data.sort_values("NAV_DT")

            if len(quarter_data) < 2:
                continue

            # 计算基金季度收益
            start_nav = quarter_data.iloc[0]["AGGR_UNIT_NVAL"]
            end_nav = quarter_data.iloc[-1]["AGGR_UNIT_NVAL"]

            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
                continue

            fund_return = (end_nav - start_nav) / start_nav

            # 计算沪深300季度收益
            quarter_start = quarter_data.iloc[0]["NAV_DT"]
            quarter_end = quarter_data.iloc[-1]["NAV_DT"]

            idx_data = df_idx[
                (df_idx["TRD_DT"] >= quarter_start) &
                (df_idx["TRD_DT"] <= quarter_end)
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
                "PRD_TYP": quarter_data.iloc[0]["PRD_TYP"],
                "QUARTER": str(quarter),
                "QUARTER_START": quarter_start,
                "QUARTER_END": quarter_end,
                "FUND_RETURN": fund_return,
                "IDX_RETURN": idx_return,
                "EXCESS_RETURN": excess_return
            })

    return pd.DataFrame(results)


# =========================================================
# 3.1 年度收益计算（沪深300和超额收益）
# =========================================================
def calc_yearly_idx_excess(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算年度收益（沪深300和超额收益）
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

        # 按年份分组
        grp_clean["YEAR"] = grp_clean["NAV_DT"].dt.year

        for year, year_data in grp_clean.groupby("YEAR"):
            year_data = year_data.sort_values("NAV_DT")

            if len(year_data) < 2:
                continue

            # 计算基金年度收益
            start_nav = year_data.iloc[0]["AGGR_UNIT_NVAL"]
            end_nav = year_data.iloc[-1]["AGGR_UNIT_NVAL"]

            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
                continue

            fund_return = (end_nav - start_nav) / start_nav

            # 计算沪深300年度收益
            year_start = year_data.iloc[0]["NAV_DT"]
            year_end = year_data.iloc[-1]["NAV_DT"]

            idx_data = df_idx[
                (df_idx["TRD_DT"] >= year_start) &
                (df_idx["TRD_DT"] <= year_end)
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
                "YEAR": year,
                "YEARLY_IDX_RETURN": idx_return,
                "YEARLY_EXCESS_RETURN": excess_return
            })

    return pd.DataFrame(results)


# =========================================================
# 4. 计算同类平均收益
# =========================================================
def calc_peer_average(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类平均收益
    """
    peer_avg = (
        df_quarterly.groupby(["PRD_TYP", "QUARTER"])["FUND_RETURN"]
        .mean()
        .reset_index(name="PEER_AVG_RETURN")
    )
    return df_quarterly.merge(peer_avg, on=["PRD_TYP", "QUARTER"], how="left")


# =========================================================
# 4.1 计算年度同类平均收益
# =========================================================
def calc_yearly_peer_average(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    计算年度同类平均收益
    """
    # 先计算每个产品的年度收益
    yearly_fund_return = (
        df_quarterly.groupby(["PRD_CODE", "PRD_TYP", "YEAR"])["FUND_RETURN"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="YEARLY_FUND_RETURN")
    )
    
    # 计算年度同类平均
    yearly_peer_avg = (
        yearly_fund_return.groupby(["PRD_TYP", "YEAR"])["YEARLY_FUND_RETURN"]
        .mean()
        .reset_index(name="YEARLY_PEER_AVG_RETURN")
    )
    
    return yearly_fund_return.merge(yearly_peer_avg, on=["PRD_TYP", "YEAR"], how="left")


# =========================================================
# 5. 计算四分位排名
# =========================================================
def calc_quarterly_rank(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    计算四分位排名
    """
    df = df_quarterly.copy()
    
    for (prd_typ, quarter), group in df.groupby(["PRD_TYP", "QUARTER"]):
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
# 6. 计算年度收益
# =========================================================
def calc_yearly_return(df_nav: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个年份的年度收益
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

        # 按年份分组计算年度收益
        grp_clean["YEAR"] = grp_clean["NAV_DT"].dt.year
        
        for year, year_data in grp_clean.groupby("YEAR"):
            if len(year_data) < 2:
                continue
            
            year_data = year_data.sort_values("NAV_DT")
            start_nav = year_data.iloc[0]["AGGR_UNIT_NVAL"]
            end_nav = year_data.iloc[-1]["AGGR_UNIT_NVAL"]
            
            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav == 0:
                continue
            
            yearly_return = (end_nav - start_nav) / start_nav
            
            results.append({
                "PRD_CODE": prd_code,
                "YEAR": year,
                "YEARLY_RETURN": yearly_return
            })
    
    return pd.DataFrame(results)


# =========================================================
# 7. 计算季度胜率
# =========================================================
def calc_quarterly_win_rate(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    计算季度胜率
    """
    results = []
    
    for (prd_code, prd_typ), group in df_quarterly.groupby(["PRD_CODE", "PRD_TYP"]):
        # 最近4个季度的数据
        recent_4_quarters = group.sort_values("QUARTER", ascending=False).head(4)
        total_quarters = len(recent_4_quarters)
        
        if total_quarters == 0:
            continue
        
        # 统计收益为正的季度数
        positive_quarters = len(recent_4_quarters[recent_4_quarters["FUND_RETURN"] > 0])
        win_rate = positive_quarters / total_quarters
        
        results.append({
            "PRD_CODE": prd_code,
            "PRD_TYP": prd_typ,
            "QUARTERLY_WIN_RATE": win_rate
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

    # 计算季度收益
    df_quarterly = calc_quarterly_return(df_nav, df_hs300, df_base_info)

    if df_quarterly.empty:
        print("没有可用的季度收益数据")
        exit()

    # 计算同类平均收益
    df_quarterly = calc_peer_average(df_quarterly)

    # 计算四分位排名
    df_quarterly = calc_quarterly_rank(df_quarterly)

    # 计算年度收益
    df_yearly = calc_yearly_return(df_nav, df_base_info)

    # 计算年度沪深300和超额收益
    df_yearly_idx_excess = calc_yearly_idx_excess(df_nav, df_hs300, df_base_info)

    # 计算年度同类平均收益
    df_quarterly["YEAR"] = df_quarterly["QUARTER"].str[:4].astype(int)
    df_yearly_peer = calc_yearly_peer_average(df_quarterly)

    # 计算季度胜率
    df_win_rate = calc_quarterly_win_rate(df_quarterly)

    # 合并数据
    df_result = df_quarterly.merge(df_yearly, on=["PRD_CODE", "YEAR"], how="left")
    df_result = df_result.merge(df_yearly_idx_excess, on=["PRD_CODE", "YEAR"], how="left")
    df_result = df_result.merge(df_yearly_peer, on=["PRD_CODE", "PRD_TYP", "YEAR"], how="left")
    df_result = df_result.merge(df_win_rate, on=["PRD_CODE", "PRD_TYP"], how="left")

    # 格式化输出
    for col in ["FUND_RETURN", "IDX_RETURN", "EXCESS_RETURN", "PEER_AVG_RETURN", "YEARLY_RETURN", "YEARLY_IDX_RETURN", "YEARLY_EXCESS_RETURN", "YEARLY_PEER_AVG_RETURN"]:
        if col in df_result.columns:
            df_result[col] = df_result[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    if "QUARTERLY_WIN_RATE" in df_result.columns:
        df_result["QUARTERLY_WIN_RATE"] = df_result["QUARTERLY_WIN_RATE"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    # 创建产品名称映射
    prd_name_map = df_base_info.set_index("PRD_CODE")["PRD_NAME"].to_dict()
    
    # 输出结果
    output_file = "季度区间收益明细结果.txt"
    output_lines = []
    output_lines.append("=" * 200)
    output_lines.append("📊 季度区间收益明细（BI效果）")
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
        prd_data["YEAR"] = prd_data["QUARTER"].str[:4]
        
        for year in sorted(prd_data["YEAR"].unique(), reverse=True):
            year_data = prd_data[prd_data["YEAR"] == year]
            year_data = year_data.sort_values("QUARTER")
            
            # 准备季度数据和年度数据
            quarter_data = {}
            yearly_data = {}
            
            for _, row in year_data.iterrows():
                quarter = row["QUARTER"][5:6]  # 提取季度
                quarter_data[quarter] = {
                    "FUND_RETURN": row["FUND_RETURN"],
                    "IDX_RETURN": row["IDX_RETURN"],
                    "EXCESS_RETURN": row["EXCESS_RETURN"],
                    "PEER_AVG_RETURN": row["PEER_AVG_RETURN"],
                    "QUARTER_RANK": row["QUARTER_RANK"]
                }
                
                # 保存年度数据（取最后一行）
                yearly_data = {
                    "YEARLY_RETURN": row["YEARLY_RETURN"],
                    "YEARLY_IDX_RETURN": row.get("YEARLY_IDX_RETURN", "--"),
                    "YEARLY_EXCESS_RETURN": row.get("YEARLY_EXCESS_RETURN", "--"),
                    "YEARLY_PEER_AVG_RETURN": row.get("YEARLY_PEER_AVG_RETURN", "--")
                }
            
            # 输出年份标题
            output_lines.append("")
            output_lines.append(f"年份: {year}")
            output_lines.append("-" * 200)
            
            # 输出表头
            header = ["简称"]
            header.extend(["第一季度", "第二季度", "第三季度", "第四季度"])
            header.append("全年")
            output_lines.append("|" + "|".join([f"{h:15s}" for h in header]) + "|")
            output_lines.append("-" * 200)
            
            # 输出基金收益行
            fund_display_name = f"{prd_code}({prd_name[:8]})"
            fund_row = [fund_display_name]
            for q in range(1, 5):
                q_str = str(q)
                if q_str in quarter_data:
                    fund_row.append(quarter_data[q_str]["FUND_RETURN"])
                else:
                    fund_row.append("--")
            fund_row.append(year_data.iloc[-1]["YEARLY_RETURN"] if "YEARLY_RETURN" in year_data.columns else "--")
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in fund_row]) + "|")
            
            # 输出沪深300行
            idx_row = ["沪深300"]
            for q in range(1, 5):
                q_str = str(q)
                if q_str in quarter_data:
                    idx_row.append(quarter_data[q_str]["IDX_RETURN"])
                else:
                    idx_row.append("--")
            idx_row.append(yearly_data.get("YEARLY_IDX_RETURN", "--"))
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in idx_row]) + "|")
            
            # 输出超额收益行
            excess_row = ["超额收益(几何)"]
            for q in range(1, 5):
                q_str = str(q)
                if q_str in quarter_data:
                    excess_row.append(quarter_data[q_str]["EXCESS_RETURN"])
                else:
                    excess_row.append("--")
            excess_row.append(yearly_data.get("YEARLY_EXCESS_RETURN", "--"))
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in excess_row]) + "|")
            
            # 输出同类平均行
            peer_row = ["同类平均"]
            for q in range(1, 5):
                q_str = str(q)
                if q_str in quarter_data:
                    peer_row.append(quarter_data[q_str]["PEER_AVG_RETURN"])
                else:
                    peer_row.append("--")
            peer_row.append(yearly_data.get("YEARLY_PEER_AVG_RETURN", "--"))
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in peer_row]) + "|")
            
            # 输出四分位排名行
            quarter_rank_row = ["四分位"]
            for q in range(1, 5):
                q_str = str(q)
                if q_str in quarter_data:
                    quarter_rank_row.append(quarter_data[q_str]["QUARTER_RANK"])
                else:
                    quarter_rank_row.append("--")
            # 简化年度四分位排名计算
            yearly_quarter_rank = "--"
            if len(year_data) > 0:
                try:
                    # 获取当前产品的年度收益
                    current_year_return = yearly_data.get("YEARLY_RETURN", "--")
                    if current_year_return != "--":
                        # 获取同类型产品的年度收益数据
                        prd_typ = year_data.iloc[0]["PRD_TYP"]
                        same_type_data = df_result[(df_result["YEAR"] == year) & (df_result["PRD_TYP"] == prd_typ)]
                        
                        if not same_type_data.empty:
                            # 提取有效的年度收益
                            valid_returns = []
                            for _, row in same_type_data.iterrows():
                                ret_str = row.get("YEARLY_RETURN", "--")
                                if ret_str != "--":
                                    try:
                                        valid_returns.append(float(ret_str.replace("%", "")) / 100)
                                    except:
                                        pass
                            
                            if valid_returns:
                                # 计算当前产品的收益在同类型中的排名
                                current_return_val = float(current_year_return.replace("%", "")) / 100
                                # 计算比当前收益高的产品数量
                                higher_count = sum(1 for r in valid_returns if r > current_return_val)
                                # 计算排名百分比
                                rank_percentile = higher_count / len(valid_returns) if len(valid_returns) > 0 else 0
                                # 根据百分比确定四分位
                                if rank_percentile < 0.25:
                                    yearly_quarter_rank = "1/4"
                                elif rank_percentile < 0.5:
                                    yearly_quarter_rank = "2/4"
                                elif rank_percentile < 0.75:
                                    yearly_quarter_rank = "3/4"
                                else:
                                    yearly_quarter_rank = "4/4"
                except:
                    pass
            quarter_rank_row.append(yearly_quarter_rank)
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in quarter_rank_row]) + "|")
            
            output_lines.append("")

    output_lines.append("=" * 200)
    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n结果已保存至：{output_file}")
