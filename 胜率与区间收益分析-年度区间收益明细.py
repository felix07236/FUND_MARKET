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
# 3. 年度收益计算
# =========================================================
def calc_yearly_return(df_nav: pd.DataFrame, df_idx: pd.DataFrame, df_base_info: pd.DataFrame) -> pd.DataFrame:
    """
    计算年度收益（基金和沪深300）
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
                "PRD_TYP": year_data.iloc[0]["PRD_TYP"],
                "YEAR": year,
                "YEAR_START": year_start,
                "YEAR_END": year_end,
                "FUND_RETURN": fund_return,
                "IDX_RETURN": idx_return,
                "EXCESS_RETURN": excess_return
            })

    return pd.DataFrame(results)


# =========================================================
# 4. 计算同类平均收益
# =========================================================
def calc_peer_average(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类平均收益
    """
    peer_avg = (
        df_yearly.groupby(["PRD_TYP", "YEAR"]) ["FUND_RETURN"]
        .mean()
        .reset_index(name="PEER_AVG_RETURN")
    )
    return df_yearly.merge(peer_avg, on=["PRD_TYP", "YEAR"], how="left")


# =========================================================
# 5. 计算同类排名
# =========================================================
def calc_category_rank(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    计算同类排名
    """
    df = df_yearly.copy()
    
    for (prd_typ, year), group in df.groupby(["PRD_TYP", "YEAR"]):
        # 按基金收益降序排序
        sorted_group = group.sort_values("FUND_RETURN", ascending=False)
        total_count = len(sorted_group)
        
        if total_count == 0:
            continue
        
        # 分配排名
        for i, (idx, row) in enumerate(sorted_group.iterrows()):
            rank = i + 1
            df.loc[idx, "CATEGORY_RANK"] = f"{rank}/{total_count}"
    
    return df


# =========================================================
# 6. 计算四分位排名
# =========================================================
def calc_quarterly_rank(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    计算四分位排名
    """
    df = df_yearly.copy()
    
    for (prd_typ, year), group in df.groupby(["PRD_TYP", "YEAR"]):
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
# 7. 主流程
# =========================================================
if __name__ == "__main__":
    df_nav = fetch_fin_prd_nav()
    df_hs300 = fetch_hs300_quot()
    df_base_info = fetch_pty_prd_base_info()
    df_nav = fill_special_prd_typ(df_nav, df_base_info)

    # 计算年度收益
    df_yearly = calc_yearly_return(df_nav, df_hs300, df_base_info)

    if df_yearly.empty:
        print("没有可用的年度收益数据")
        exit()

    # 计算同类平均收益
    df_yearly = calc_peer_average(df_yearly)

    # 计算同类排名
    df_yearly = calc_category_rank(df_yearly)

    # 计算四分位排名
    df_yearly = calc_quarterly_rank(df_yearly)

    # 格式化输出
    for col in ["FUND_RETURN", "IDX_RETURN", "EXCESS_RETURN", "PEER_AVG_RETURN"]:
        if col in df_yearly.columns:
            df_yearly[col] = df_yearly[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "--")

    # 创建产品名称映射
    prd_name_map = df_base_info.set_index("PRD_CODE")["PRD_NAME"].to_dict()
    
    # 输出结果
    output_file = "年度区间收益明细结果.txt"
    output_lines = []
    output_lines.append("=" * 200)
    output_lines.append("📊 年度区间收益明细（BI效果）")
    output_lines.append(f"✅ 共统计产品数量：{len(df_yearly['PRD_CODE'].unique())}")
    output_lines.append("=" * 200)
    output_lines.append("")

    # 按产品分组输出
    for prd_code in df_yearly["PRD_CODE"].unique():
        prd_data = df_yearly[df_yearly["PRD_CODE"] == prd_code]
        prd_typ = prd_data.iloc[0]["PRD_TYP"]
        prd_name = prd_name_map.get(prd_code, prd_code)
        
        output_lines.append("-" * 200)
        output_lines.append(f"产品代码: {prd_code} ({prd_name}) | 产品类型: {prd_typ}")
        output_lines.append("-" * 200)
        
        # 输出表头
        header = ["年份", "基金收益", "沪深300", "超额收益(几何)", "同类平均", "同类排名", "四分位"]
        output_lines.append("|" + "|".join([f"{h:15s}" for h in header]) + "|")
        output_lines.append("-" * 200)
        
        # 按年份降序排序
        prd_data = prd_data.sort_values("YEAR", ascending=False)
        
        # 输出数据行
        for _, row in prd_data.iterrows():
            data_row = [
                str(row["YEAR"]),
                row["FUND_RETURN"],
                row["IDX_RETURN"],
                row["EXCESS_RETURN"],
                row.get("PEER_AVG_RETURN", "--"),
                row.get("CATEGORY_RANK", "--"),
                row.get("QUARTER_RANK", "--")
            ]
            output_lines.append("|" + "|".join([f"{str(x):15s}" for x in data_row]) + "|")
        
        output_lines.append("")

    output_lines.append("=" * 200)
    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n结果已保存至：{output_file}")