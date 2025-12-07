"""
基于 README_risktraining_optimized.md 的任务 1–4：

任务 1：计算 df_r(t)、df_defect(t)、df_nondefect(t)
任务 2：计算 rtw(t)、risk_avg(t)
任务 3：生成初版烟雾词词表 v0
任务 4：用烟雾词为每条新闻计算 raw_risk(d) 并生成 0–3 档伪标签 PL1

初始数据集:
    - data_combined_4types.xlsx
    - 必需列:
        * '内容': 新闻文本
        * 'risk' : 人工标注风险 (0–3) 或 NaN
    - 可选列:
        * 'tokenized_text': 若已有分词结果 (以空格分隔词), 会优先使用

输出文件:
    1) riskword_term_stats_v0.csv
        每个 term 的:
            term
            df_0, df_1, df_2, df_3
            df_defect, df_nondefect
            rtw, risk_avg
    2) smokewords_v0.csv
        按筛选规则挑出的初版烟雾词表 (v0)
    3) data_with_raw_risk_pl1.xlsx
        在原始数据上附加:
            raw_risk, pseudo_label_pl1, high_conf_pl1 (0/1)
    4) pseudo_labels_pl1.csv
        高置信伪标签 PL1: doc_id, pseudo_label, confidence
"""

import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from smokewords_stats import tokenize  # 复用已有的分词规则


# ===================== 配置区 =====================

DATA_PATH = "data_combined_4types.xlsx"

TEXT_COL = "内容"
RISK_COL = "risk"

# 风险权重 w_r (0–3 档)
RISK_WEIGHTS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

ALPHA = 1.0  # 平滑系数
EPS = 1e-9

# 生成 v0 时的筛选阈值
MIN_DF_DEFECT = 5
RTW_THRESHOLD = 1.5
RISK_AVG_MIN = 1.0
RISK_AVG_HIGH = 2.5  # 高风险烟雾词阈值

# PL1 离散化与高置信阈值 (0–3 档)
RAW_RISK_BINS = [0.5, 1.5, 2.5]  # 对应 0/1/2/3 档的分界
HIGH_CONF_LOW = 0.3
HIGH_CONF_HIGH = 2.7


# ===================== 工具函数 =====================

def load_data_with_tokens(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")

    df = pd.read_excel(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少文本列 '{TEXT_COL}'，当前列: {df.columns.tolist()}")

    # 构建 tokens 列
    if "tokenized_text" in df.columns:
        def parse_tokens(x):
            if isinstance(x, str):
                # 假设以空格分隔
                return [t for t in x.split() if t.strip()]
            if isinstance(x, list):
                return [str(t).strip() for t in x if str(t).strip()]
            return tokenize(str(x))

        df["tokens"] = df["tokenized_text"].apply(parse_tokens)
    else:
        df["tokens"] = df[TEXT_COL].astype(str).apply(tokenize)

    return df


def compute_df_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    任务 1:
        计算每个 term 在 risk 档位 0..3 的文档频数 df_r(t)，并据此计算:
            df_nondefect(t) = df_0(t)
            df_defect(t) = Σ_r w_r * df_r(t)
    仅使用有人工 risk 的样本 (NaN 跳过)。
    """
    df_r_counters: Dict[int, Counter] = {r: Counter() for r in range(4)}

    for _, row in df.iterrows():
        risk = row.get(RISK_COL, None)
        if pd.isna(risk):
            continue
        try:
            r = int(risk)
        except Exception:
            continue
        if r not in df_r_counters:
            continue
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            continue
        unique_terms = set(tokens)
        for t in unique_terms:
            df_r_counters[r][t] += 1

    all_terms = set()
    for r in range(4):
        all_terms.update(df_r_counters[r].keys())

    records: List[Dict] = []
    for term in all_terms:
        df_0 = df_r_counters[0].get(term, 0)
        df_1 = df_r_counters[1].get(term, 0)
        df_2 = df_r_counters[2].get(term, 0)
        df_3 = df_r_counters[3].get(term, 0)

        df_nondefect = df_0
        df_defect = (
            RISK_WEIGHTS[1] * df_1
            + RISK_WEIGHTS[2] * df_2
            + RISK_WEIGHTS[3] * df_3
        )

        records.append(
            {
                "term": term,
                "df_0": int(df_0),
                "df_1": int(df_1),
                "df_2": int(df_2),
                "df_3": int(df_3),
                "df_defect": int(df_defect),
                "df_nondefect": int(df_nondefect),
            }
        )

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        raise ValueError("未能从人工标注样本中统计出任何词频，请检查数据。")

    return stats_df


def add_rtw_and_risk_avg(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    任务 2:
        基于 df_r(t) 与 df_defect/df_nondefect，计算:
            rtw(t)      = (df_defect + α) / (df_nondefect + α)
            risk_avg(t) = Σ_{r} r * df_r(t) / (Σ_{r} df_r(t) + ε)
    """
    def _calc_row(row):
        df_0 = row["df_0"]
        df_1 = row["df_1"]
        df_2 = row["df_2"]
        df_3 = row["df_3"]
        df_defect = row["df_defect"]
        df_nondefect = row["df_nondefect"]

        rtw = (df_defect + ALPHA) / (df_nondefect + ALPHA)

        num = 0.0
        den = 0.0
        for r, df_r in zip(range(4), [df_0, df_1, df_2, df_3]):
            num += r * df_r
            den += df_r
        risk_avg = num / (den + EPS)
        return rtw, risk_avg

    rtw_list = []
    risk_avg_list = []
    for _, row in stats_df.iterrows():
        rtw, risk_avg = _calc_row(row)
        rtw_list.append(rtw)
        risk_avg_list.append(risk_avg)

    stats_df = stats_df.copy()
    stats_df["rtw"] = rtw_list
    stats_df["risk_avg"] = risk_avg_list
    return stats_df


def build_smokewords_v0(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    任务 3:
        按规则筛选初版烟雾词表 v0。
        这里只做数值过滤，不直接赋予 A/B/C/D/E/F/G 语义类别，
        后续可结合 LLM 或人工在此基础上再加 category。
    """
    cond_support = stats_df["df_defect"] >= MIN_DF_DEFECT
    cond_rtw = stats_df["rtw"] >= RTW_THRESHOLD
    cond_avg = stats_df["risk_avg"] >= RISK_AVG_MIN

    v0 = stats_df[cond_support & cond_rtw & cond_avg].copy()
    v0["is_high_risk"] = v0["risk_avg"] >= RISK_AVG_HIGH
    v0 = v0.sort_values(["rtw", "risk_avg"], ascending=False).reset_index(drop=True)
    return v0


def discretize_raw_risk(raw: float) -> int:
    """
    将连续 raw_risk(d) 映射到 0–3 档。
    """
    if raw < RAW_RISK_BINS[0]:
        return 0
    elif raw < RAW_RISK_BINS[1]:
        return 1
    elif raw < RAW_RISK_BINS[2]:
        return 2
    else:
        return 3


def compute_raw_risk_and_pl1(df: pd.DataFrame, stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    任务 4:
        利用 risk_avg(t) 为每条新闻计算 raw_risk(d)，并生成 0–4 档伪标签 PL1。

    raw_risk(d) 使用词频版本:
        raw_risk(d) = Σ tf(t,d) * risk_avg(t) / Σ tf(t,d)

    高置信伪标签:
        raw_risk < HIGH_CONF_LOW  → 高置信 0 档
        raw_risk > HIGH_CONF_HIGH → 高置信 4 档
    """
    term2risk_avg = {row["term"]: float(row["risk_avg"]) for _, row in stats_df.iterrows()}

    raw_risks = []
    pseudo_labels = []
    high_conf_flags = []

    for _, row in df.iterrows():
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            raw = 0.0
        else:
            # 词频统计
            tf = Counter(tokens)
            num = 0.0
            den = 0.0
            for t, f in tf.items():
                rv = term2risk_avg.get(t, None)
                if rv is None:
                    continue
                num += f * rv
                den += f
            raw = num / den if den > 0 else 0.0

        raw_risks.append(raw)
        pseudo_labels.append(discretize_raw_risk(raw))

        # 高置信规则
        if raw < HIGH_CONF_LOW:
            high_conf = 1
        elif raw > HIGH_CONF_HIGH:
            high_conf = 1
        else:
            high_conf = 0
        high_conf_flags.append(high_conf)

    df_out = df.copy()
    df_out["raw_risk"] = raw_risks
    df_out["pseudo_label_pl1"] = pseudo_labels
    df_out["high_conf_pl1"] = high_conf_flags

    # 生成高置信伪标签集合 PL1
    pl1_records = []
    for idx, row in df_out.iterrows():
        if row["high_conf_pl1"] == 1:
            pl1_records.append(
                {
                    "doc_id": idx,
                    "pseudo_label": int(row["pseudo_label_pl1"]),
                    "confidence": float(row["raw_risk"]),
                }
            )

    pl1_df = pd.DataFrame(pl1_records)
    return df_out, pl1_df


def main():
    # 读取数据并构建 tokens
    df = load_data_with_tokens(DATA_PATH)
    print(f"[INFO] 读取数据 {DATA_PATH}，共 {len(df)} 条。")

    # 任务 1: 统计 df_r 和 df_defect/df_nondefect
    stats_df = compute_df_r(df)
    print(f"[INFO] 已计算 df_r，得到 {len(stats_df)} 个 term。")

    # 任务 2: 计算 rtw 与 risk_avg
    stats_df = add_rtw_and_risk_avg(stats_df)

    # 导出 term 统计总表 (任务 1+2)
    stats_path = "riskword_term_stats_v0.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存词统计 v0 到: {stats_path}")

    # 任务 3: 生成初版烟雾词表 v0
    v0_df = build_smokewords_v0(stats_df)
    v0_path = "smokewords_v0.csv"
    v0_df.to_csv(v0_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存初版烟雾词表 v0 到: {v0_path} (共 {len(v0_df)} 个词)")

    # 任务 4: 为每条新闻计算 raw_risk 并生成 PL1
    df_with_scores, pl1_df = compute_raw_risk_and_pl1(df, stats_df)

    out_data_path = "data_with_raw_risk_pl1.xlsx"
    df_with_scores.to_excel(out_data_path, index=False)
    print(f"[INFO] 已保存附加 raw_risk / PL1 的数据到: {out_data_path}")

    pl1_path = "pseudo_labels_pl1.csv"
    pl1_df.to_csv(pl1_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存高置信伪标签 PL1 到: {pl1_path} (共 {len(pl1_df)} 条)")

    print("\n[INFO] 例示前 10 条高风险烟雾词：")
    print(v0_df.sort_values("risk_avg", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()


