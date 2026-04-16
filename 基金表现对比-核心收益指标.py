import oracledb
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
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
PERIODS = [
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
def calc_fund_metrics(df_nav, start_dt, end_dt, period, fund_established_dt):
    """计算基金的各项指标"""
    df = df_nav.copy()
    
    # 数据清洗：只保留指定时间范围内的净值数据
    mask = (df["NAV_DT"] >= start_dt) & (df["NAV_DT"] <= end_dt)
    df_filtered = df[mask]

    if df_filtered.empty:
        return None

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

        # 计算累计收益
        total_return = nav_end / nav_start - 1

        result.append({
            "prd_code": prd_code,
            "total_return": total_return,
            "nav_end": nav_end,
            "start_dt": actual_start_dt,
            "end_dt": last["NAV_DT"]
        })

    if not result:
        return None
    
    # 返回第一个产品的结果（因为每次只处理一个产品）
    return result[0]


def calc_index_return(df_idx: pd.DataFrame, start_dt, end_dt) -> float:
    """计算指数在指定时间区间的收益"""
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
    try:
        print("开始计算基金核心收益指标...")
        print(f"产品代码: {prd_codes}")
        print(f"指数代码: {index_code}")

        # 验证日期格式
        validate_date(custom_start_date)
        validate_date(custom_end_date)

        # 获取数据
        print("\n1. 获取基金净值数据...")
        df_nav = fetch_fin_prd_nav(prd_codes)
        print(f"基金净值数据行数: {len(df_nav)}")

        print("\n2. 获取指数数据...")
        df_index = fetch_index_quot(index_code)
        print(f"指数数据行数: {len(df_index)}")

        print("\n3. 获取产品基础信息...")
        df_base_info = fetch_pty_prd_base_info(prd_codes)
        print(f"产品基础信息行数: {len(df_base_info)}")

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

        # 获取产品类型
        prd_types = {}
        for _, row in df_base_info.iterrows():
            if pd.notna(row["PRD_TYP"]):
                prd_types[row["PRD_CODE"]] = row["PRD_TYP"]

        # 获取产品最新净值日期
        latest_nav_dates = {}
        for prd_code, g in df_nav.groupby("PRD_CODE"):
            latest_nav_dates[prd_code] = g["NAV_DT"].max()

        all_results = []

        # 计算每个产品的各项指标
        for prd_code in prd_codes:
            fund_established_dt = prd_established_dates.get(prd_code)
            prd_typ = prd_types.get(prd_code, "")
            latest_nav_date = latest_nav_dates.get(prd_code)
            
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
            
            # 确定计算的结束日期（使用最新净值日期）
            end_dt = latest_nav_date if latest_nav_date else datetime.now()
            
            # ========== 先计算成立来的年化收益（只计算一次）==========
            df_all = df_prd_nav.copy()
            df_all = df_all.sort_values("NAV_DT")
            first_all = df_all.iloc[0]
            last_all = df_all.iloc[-1]
            
            nav_start_all = (
                first_all["AGGR_UNIT_NVAL"]
                if pd.notna(first_all["AGGR_UNIT_NVAL"])
                else first_all["UNIT_NVAL"]
            )
            nav_end_all = (
                last_all["AGGR_UNIT_NVAL"]
                if pd.notna(last_all["AGGR_UNIT_NVAL"])
                else last_all["UNIT_NVAL"]
            )
            
            if pd.notna(nav_start_all) and pd.notna(nav_end_all):
                days_since_established = (end_dt - fund_established_dt).days
                if days_since_established > 0:
                    established_annualized = (math.pow(nav_end_all / nav_start_all, 365 / days_since_established) - 1)
                else:
                    established_annualized = 0
            else:
                established_annualized = None
            # ====================================================
            
            # 计算各周期的指标
            print(f"\n5. 计算产品 {prd_code} 的各周期指标:")
            for period in PERIODS:
                # 获取周期的开始和结束日期
                start_dt, _ = get_period_dates(end_dt, period, fund_established_dt)
                
                # 计算基金指标
                fund_metrics = calc_fund_metrics(df_prd_nav, start_dt, end_dt, period, fund_established_dt)
                
                if fund_metrics:
                    # 计算指数收益
                    index_return = calc_index_return(df_index, start_dt, end_dt)
                    
                    # 计算超额收益（几何）
                    excess_return = fund_metrics["total_return"] - index_return if index_return is not None else None
                    
                    # 构建结果（所有周期都使用同一个成立来年化收益）
                    result = {
                        "产品代码": prd_code,
                        "产品类型": prd_typ,
                        "最新净值日期": latest_nav_date,
                        "周期": period,
                        "起始日期": start_dt,
                        "结束日期": end_dt,
                        "累计净值": fund_metrics["nav_end"],
                        "累计收益": fund_metrics["total_return"],
                        "成立来年化": established_annualized,
                        "指数收益": index_return,
                        "超额收益": excess_return
                    }
                    all_results.append(result)
                    print(f"  {period}: 累计净值={fund_metrics['nav_end']:.4f}, 总收益={(fund_metrics['total_return']*100):.2f}%")

        # 处理结果
        if all_results:
            final_df = pd.DataFrame(all_results)
            
            # 格式化数据
            final_df["累计净值"] = final_df["累计净值"].round(4)
            final_df["累计收益"] = (final_df["累计收益"] * 100).round(2).astype(str) + "%"
            final_df["成立来年化"] = final_df["成立来年化"].apply(lambda x: f"{x*100:.2f}%" if x is not None else "-")
            final_df["指数收益"] = final_df["指数收益"].apply(lambda x: f"{x*100:.2f}%" if x is not None else "-")
            final_df["超额收益"] = final_df["超额收益"].apply(lambda x: f"{x*100:.2f}%" if x is not None else "-")
            
            # 调整列顺序
            final_df = final_df[[
                "产品代码", "最新净值日期", "产品类型", "周期", 
                "起始日期", "结束日期", "累计净值", "累计收益", 
                "成立来年化", "指数收益", "超额收益"
            ]]
            
            # 导出结果
            output_file = "基金表现对比-核心收益指标结果.csv"
            final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n6. 计算完成！结果已保存至：{output_file}")
            print("\n7. 计算结果预览：")
            print(final_df.head())
        else:
            print("\n未找到符合条件的数据！")
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


# =========================================================
# 6. 示例调用
# =========================================================
if __name__ == "__main__":
    # 示例参数
    # 产品代码列表（2-5个）
    prd_codes = ["1011", "1012", "1022"]
    # 市场指数代码
    index_code = "000300.IDX.CSIDX"
    # 自定义时间区间（可选）
    custom_start_date = None
    custom_end_date = None

    # 调用主函数
    main(prd_codes, index_code, custom_start_date, custom_end_date)
