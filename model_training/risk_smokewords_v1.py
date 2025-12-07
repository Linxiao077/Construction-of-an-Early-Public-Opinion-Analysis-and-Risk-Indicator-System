"""
基于 README_risktraining_optimized.md 的任务 7–9（简化版）：

任务 7：根据 PL2（新标签集）重新计算 df_r(t)、rtw(t)、risk_avg(t)，生成烟雾词词表 v1
任务 8：可多轮迭代自训练（此处示例实现单次更新，可按需封装循环）
任务 9：输出最终烟雾词词表、新闻风险预测与训练记录（本脚本专注于词表 v1 部分）

依赖:
    1) data_combined_4types.xlsx
        - 列:
            * '内容'
            * 'risk' (人工标签, 0–3 或 NaN)
    2) pseudo_labels_pl2.csv
        - 由 risk_bert_riskmodel.py 生成
        - 列:
            * doc_id
            * pseudo_label ∈ {0..3}

逻辑:
    - 将人工 risk 与 PL2 预测标签合并:
        * 若存在人工 risk，则优先使用人工 risk
        * 否则使用 PL2 中的 pseudo_label
    - 基于新的标签集，按任务 1–2 的方法重新计算:
        df_r(t)、df_defect(t)、df_nondefect(t)、rtw(t)、risk_avg(t)
    - 按任务 3 的规则筛选生成更新后的烟雾词表 v1。

输出:
    1) riskword_term_stats_v1.csv
    2) smokewords_v1.csv
"""

import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from smokewords_stats import tokenize  # 与 v0 保持分词口径一致


DATA_PATH = "data_combined_4types.xlsx"
PL2_PATH = "pseudo_labels_pl2.csv"

TEXT_COL = "内容"
RISK_COL = "risk"

RISK_WEIGHTS = {0: 0, 1: 1, 2: 2, 3: 3}
ALPHA = 1.0
EPS = 1e-9

MIN_DF_DEFECT = 5
RTW_THRESHOLD = 1.5
RISK_AVG_MIN = 1.0
RISK_AVG_HIGH = 2.5


def load_data_and_pl2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    if TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少文本列 '{TEXT_COL}'，当前列: {df.columns.tolist()}")

    if not os.path.isfile(PL2_PATH):
        raise FileNotFoundError(f"未找到 PL2 文件: {PL2_PATH}，请先运行 risk_bert_riskmodel.py。")
    pl2 = pd.read_csv(PL2_PATH)
    return df, pl2


def attach_final_labels(df: pd.DataFrame, pl2: pd.DataFrame) -> pd.DataFrame:
    """
    将人工 risk 与 PL2 合并成一个最终标签列 'final_risk':
        - 若存在人工 risk (非 NaN)，优先使用
        - 否则使用 PL2 的 pseudo_label
    """
    df = df.copy()
    df["tokens"] = df[TEXT_COL].astype(str).apply(tokenize)

    pl2_map = {int(row["doc_id"]): int(row["pseudo_label"]) for _, row in pl2.iterrows()}

    final_labels = []
    for idx, row in df.iterrows():
        r = row.get(RISK_COL, None)
        if not pd.isna(r):
            try:
                val = int(r)
                if 0 <= val <= 3:
                    final_labels.append(val)
                    continue
            except Exception:
                pass

        # 若无人工标签，尝试使用 PL2
        if idx in pl2_map:
            val = pl2_map[idx]
            if 0 <= val <= 3:
                final_labels.append(val)
            else:
                final_labels.append(None)
        else:
            final_labels.append(None)

    df["final_risk"] = final_labels
    labeled_df = df[df["final_risk"].notna()].copy()
    labeled_df["final_risk"] = labeled_df["final_risk"].astype(int)

    print(
        f"[INFO] 合并人工 + PL2 后，可用于重新统计的样本数: {len(labeled_df)} "
        f"(其中人工标签 {df[RISK_COL].notna().sum()} 条，"
        f"PL2 补充 {len(labeled_df) - df[RISK_COL].notna().sum()} 条可能存在重叠)。"
    )
    return labeled_df


def compute_df_r_from_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 'final_risk' 重新计算:
        df_r(t)、df_defect(t)、df_nondefect(t)
    """
    df_r_counters: Dict[int, Counter] = {r: Counter() for r in range(4)}

    for _, row in df.iterrows():
        risk = row["final_risk"]
        if pd.isna(risk):
            continue
        r = int(risk)
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
        raise ValueError("基于 final_risk 统计时未得到任何词频，请检查标签。")
    return stats_df


def add_rtw_and_risk_avg(stats_df: pd.DataFrame) -> pd.DataFrame:
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


def build_smokewords_v1(stats_df: pd.DataFrame) -> pd.DataFrame:
    cond_support = stats_df["df_defect"] >= MIN_DF_DEFECT
    cond_rtw = stats_df["rtw"] >= RTW_THRESHOLD
    cond_avg = stats_df["risk_avg"] >= RISK_AVG_MIN

    v1 = stats_df[cond_support & cond_rtw & cond_avg].copy()
    v1["is_high_risk"] = v1["risk_avg"] >= RISK_AVG_HIGH
    v1 = v1.sort_values(["rtw", "risk_avg"], ascending=False).reset_index(drop=True)
    return v1


def main():
    df, pl2 = load_data_and_pl2()
    labeled_df = attach_final_labels(df, pl2)

    stats_df = compute_df_r_from_final(labeled_df)
    stats_df = add_rtw_and_risk_avg(stats_df)

    stats_path = "riskword_term_stats_v1.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存 v1 词统计到: {stats_path} (共 {len(stats_df)} 个 term)")

    v1_df = build_smokewords_v1(stats_df)
    v1_path = "smokewords_v1.csv"
    v1_df.to_csv(v1_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存烟雾词词表 v1 到: {v1_path} (共 {len(v1_df)} 个词)")

    print("\n[INFO] v1 高风险烟雾词示例前 20 行：")
    print(v1_df.sort_values("risk_avg", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()


