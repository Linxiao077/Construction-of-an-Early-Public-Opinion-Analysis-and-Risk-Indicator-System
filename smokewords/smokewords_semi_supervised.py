"""
半监督 / 弱监督方式构建“召回烟雾词表”的自训练流程。

场景:
    - XLSX 数据集中:
        * 列 '内容': 新闻文本
        * 列 'risk' : 召回风险标注 (0–4)，大部分样本未标注 (NaN)
          - 0: 无风险
          - 4: 最高风险或已召回

思路:
    1. 使用已标注样本 (有 risk 的新闻)，将 risk >= DEFECT_RISK_THRESHOLD 视为“缺陷/高风险”新闻，
       其余视为“非缺陷/低风险”新闻。
       - 统计 df_defect(term)、df_nondefect(term)，计算:
           rtw(term) = (df_defect + α) / (df_nondefect + α)
         并按 rtw 降序排序，得到初始高 rtw 词表 (初始烟雾词)。
    2. 在未标注样本上做远程监督:
       - 命中大量高 rtw 词的新闻 → 伪标为“缺陷倾向”(label=1)
       - 基本不命中或只命中低 rtw 词的新闻 → 伪标为“非缺陷/低风险”(label=0)
    3. 把伪标签加入训练集合，重新统计 df/rtw；可迭代执行多轮自训练，逐步扩充样本。
    4. 最终在 “人工标注 + 伪标注” 的样本上统计得到稳定的 rtw，并输出完整烟雾词表:
       - smokewords_term_stats_semi.csv
       - smokewords_candidates_semi_topN.csv

备注:
    - 为保持分词与清洗的一致性，直接复用 smokewords_stats.py 中的 tokenize 函数。
    - 如需进一步结合 BERT 分类器，可在每轮迭代后，用当前 labeled_df 训练文本分类模型，
      再用模型对未标注样本打分，与规则伪标结果进行融合，这里给出的是“规则+统计”的基础版本。
"""

import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from smokewords_stats import tokenize  # 复用已有的分词+清洗规则


# ===================== 配置区 =====================

# 原始数据文件 (部分样本有 risk 标注，其余为 NaN)
DATA_PATH = "data_combined.xlsx"

TEXT_COL = "内容"
RISK_COL = "risk"

# 将 risk >= DEFECT_RISK_THRESHOLD 视为“缺陷/高风险”新闻
DEFECT_RISK_THRESHOLD = 2

# 统计时最小 df_total(term) 阈值
MIN_DF = 5

# rtw(term) = (df_defect + ALPHA) / (df_nondefect + ALPHA)
ALPHA = 0.5

# 初始高 rtw 词数量 (用于构造初始烟雾词表)
TOP_RTW_TERMS = 300

# 伪标记规则参数
HIGH_RTW_THRESHOLD = 2.0  # rtw >= 2 视作高风险词
LOW_RTW_MAX = 1.2         # rtw <= 1.2 视作低风险/非缺陷指向词

POS_MIN_HITS = 2          # 文档中命中高风险词数量阈值
POS_MIN_RTW_SUM = 5.0     # 文档中高风险词 rtw 之和阈值

NEG_MIN_HITS = 3          # 文档中命中低风险词数量阈值 (且无高风险词)

# 自训练最大迭代轮数
MAX_ITER = 3

# 最终输出 Top-N 候选烟雾词
TOP_N_FINAL = 500


def split_labeled_unlabeled(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 risk 是否缺失拆分出已标注与未标注样本。"""
    labeled = df[df[RISK_COL].notna()].copy()
    unlabeled = df[df[RISK_COL].isna()].copy()
    return labeled, unlabeled


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 risk 列添加二分类标签 label:
        label = 1 表示缺陷/高风险
        label = 0 表示非缺陷/低风险
    仅处理有 risk 的行，NaN 会在外部被过滤掉。
    """
    def _map(r):
        try:
            v = int(r)
        except Exception:
            return None
        return 1 if v >= DEFECT_RISK_THRESHOLD else 0

    df = df.copy()
    df["label"] = df[RISK_COL].apply(_map)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df


def build_tokenized_column(df: pd.DataFrame) -> pd.DataFrame:
    """为 DataFrame 添加 'tokens' 列，存储分词后的结果。"""
    df = df.copy()
    df["tokens"] = df[TEXT_COL].astype(str).apply(tokenize)
    return df


def compute_df_from_labeled(df_labeled: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    """
    在带有 label 列 (0/1) 的样本上统计:
        df_defect(term), df_nondefect(term), N_defect, N_nondefect
    """
    df_defect: Counter = Counter()
    df_nondefect: Counter = Counter()
    N_defect = 0
    N_nondefect = 0

    for _, row in df_labeled.iterrows():
        label = int(row["label"])
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            continue
        unique_terms = set(tokens)
        if label == 1:
            N_defect += 1
            for t in unique_terms:
                df_defect[t] += 1
        else:
            N_nondefect += 1
            for t in unique_terms:
                df_nondefect[t] += 1

    return df_defect, df_nondefect, N_defect, N_nondefect


def build_rtw_table(
    df_defect: Dict[str, int],
    df_nondefect: Dict[str, int],
    N_defect: int,
    N_nondefect: int,
) -> pd.DataFrame:
    """
    根据 df_defect / df_nondefect 统计构造 rtw 表:
        rtw(term) = (df_defect + ALPHA) / (df_nondefect + ALPHA)
        log_rtw   = log(rtw)
    同时给出 df_total, p_defect, p_nondefect。
    """
    all_terms: Set[str] = set(df_defect.keys()) | set(df_nondefect.keys())
    records: List[Dict] = []

    for term in all_terms:
        a = df_defect.get(term, 0)
        b = df_nondefect.get(term, 0)
        df_total = a + b
        if df_total < MIN_DF:
            continue

        rtw = (a + ALPHA) / (b + ALPHA)
        log_rtw = float(math.log(max(rtw, 1e-8)))
        # 文档层面的条件频率 (可选)
        p_def = (a + ALPHA) / (N_defect + 2 * ALPHA) if N_defect > 0 else 0.0
        p_non = (b + ALPHA) / (N_nondefect + 2 * ALPHA) if N_nondefect > 0 else 0.0

        records.append(
            {
                "term": term,
                "df_defect": int(a),
                "df_nondefect": int(b),
                "df_total": int(df_total),
                "rtw": float(rtw),
                "log_rtw": log_rtw,
                "p_defect": float(p_def),
                "p_nondefect": float(p_non),
            }
        )

    rtw_df = pd.DataFrame(records)
    if rtw_df.empty:
        raise ValueError("在当前 MIN_DF 和统计条件下，没有任何 term 通过过滤。")

    rtw_df = rtw_df.sort_values("rtw", ascending=False).reset_index(drop=True)
    return rtw_df


def select_high_low_terms(rtw_df: pd.DataFrame) -> Tuple[Set[str], Set[str], Dict[str, float]]:
    """
    根据 rtw 表选出:
        high_terms: 高 rtw 词集合 (候选高风险烟雾词)
        low_terms : 低 rtw 词集合 (偏向非缺陷/低风险的词)
        term2rtw  : term -> rtw 的映射
    """
    term2rtw = {row["term"]: float(row["rtw"]) for _, row in rtw_df.iterrows()}

    # 高 rtw 词: rtw >= HIGH_RTW_THRESHOLD，且在前 TOP_RTW_TERMS 内
    top_df = rtw_df.head(TOP_RTW_TERMS)
    high_terms = {
        row["term"]
        for _, row in top_df.iterrows()
        if float(row["rtw"]) >= HIGH_RTW_THRESHOLD
    }

    # 低 rtw 词: rtw <= LOW_RTW_MAX 且 df_nondefect > df_defect
    low_df = rtw_df[rtw_df["rtw"] <= LOW_RTW_MAX]
    low_terms = {
        row["term"]
        for _, row in low_df.iterrows()
        if int(row["df_nondefect"]) > int(row["df_defect"])
    }

    print(f"[INFO] 选出高 rtw 词 {len(high_terms)} 个，低 rtw 词 {len(low_terms)} 个。")
    return high_terms, low_terms, term2rtw


def pseudo_label_unlabeled(
    df_unlabeled: pd.DataFrame,
    high_terms: Set[str],
    low_terms: Set[str],
    term2rtw: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对未标注样本进行伪标记:
        - 命中高 rtw 词较多且 rtw 之和较大 → label=1
        - 无高 rtw 词，命中低 rtw 词较多 → label=0
      其余样本保持未标注。

    返回:
        newly_labeled: 新标注出 label 的样本
        remaining_unlabeled: 仍未能确定标签的样本
    """
    labeled_rows = []
    remaining_rows = []

    for _, row in df_unlabeled.iterrows():
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            remaining_rows.append(row)
            continue

        token_set = set(tokens)

        high_hits = token_set & high_terms
        low_hits = token_set & low_terms

        if high_hits:
            rtw_sum = sum(term2rtw.get(t, 1.0) for t in high_hits)
        else:
            rtw_sum = 0.0

        label: Optional[int] = None

        # 高风险伪标规则
        if len(high_hits) >= POS_MIN_HITS and rtw_sum >= POS_MIN_RTW_SUM:
            label = 1
        # 低风险伪标规则
        elif len(high_hits) == 0 and len(low_hits) >= NEG_MIN_HITS:
            label = 0

        if label is None:
            remaining_rows.append(row)
        else:
            new_row = row.copy()
            new_row["label"] = label
            labeled_rows.append(new_row)

    newly_labeled = pd.DataFrame(labeled_rows)
    remaining_unlabeled = pd.DataFrame(remaining_rows)

    print(
        f"[INFO] 伪标记得到 {len(newly_labeled)} 条样本，其中 label=1: "
        f"{(newly_labeled['label'] == 1).sum() if not newly_labeled.empty else 0} 条，"
        f"label=0: {(newly_labeled['label'] == 0).sum() if not newly_labeled.empty else 0} 条；"
        f"剩余未标注 {len(remaining_unlabeled)} 条。"
    )
    return newly_labeled, remaining_unlabeled


def export_final_stats(rtw_df: pd.DataFrame):
    """
    将最终 rtw_df 转换为与 smokewords_stats.py 输出形式接近的表，并导出。
        - smokewords_term_stats_semi.csv
        - smokewords_candidates_semi_topN.csv
    """
    # 对齐列名: df_defect / df_normal / df_total / p_defect / p_normal / score_stat
    out_df = rtw_df.rename(
        columns={
            "df_nondefect": "df_normal",
            "p_nondefect": "p_normal",
        }
    ).copy()
    out_df["score_stat"] = out_df["log_rtw"]

    stats_path = "smokewords_term_stats_semi.csv"
    out_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存半监督版本完整词统计到: {stats_path} (共 {len(out_df)} 个 term)")

    # Top-N 候选
    topN = min(TOP_N_FINAL, len(out_df))
    top_df = out_df.sort_values("score_stat", ascending=False).head(topN).copy()
    top_path = f"smokewords_candidates_semi_top{topN}.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存半监督版本候选烟雾词 Top-{topN} 到: {top_path}")

    print("\n[INFO] 半监督候选烟雾词示例前 20 行：")
    print(top_df.head(20).to_string(index=False))


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    if TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少文本列 '{TEXT_COL}'，当前列为: {df.columns.tolist()}")

    print(f"[INFO] 读取数据 {DATA_PATH}，共 {len(df)} 行。")

    # 1. 预先分词
    df = build_tokenized_column(df)

    # 2. 拆分已标注与未标注
    labeled_raw, unlabeled = split_labeled_unlabeled(df)
    labeled = add_binary_label(labeled_raw)

    print(
        f"[INFO] 初始人工标注样本: {len(labeled)} 条 "
        f"(缺陷/高风险: {(labeled['label'] == 1).sum()} 条, "
        f"非缺陷/低风险: {(labeled['label'] == 0).sum()} 条); "
        f"未标注样本: {len(unlabeled)} 条。"
    )

    # 3. 自训练迭代
    current_labeled = labeled.copy()
    current_unlabeled = unlabeled.copy()

    for it in range(1, MAX_ITER + 1):
        print(f"\n========== 自训练第 {it} 轮 ==========")
        df_defect, df_nondefect, N_defect, N_nondefect = compute_df_from_labeled(current_labeled)

        print(
            f"[INFO] 当前统计: N_defect={N_defect}, N_nondefect={N_nondefect}, "
            f"已标注样本共 {len(current_labeled)} 条。"
        )

        if N_defect == 0 or N_nondefect == 0:
            print("[WARN] 某一类样本数为 0，无法继续自训练。")
            break

        rtw_df = build_rtw_table(df_defect, df_nondefect, N_defect, N_nondefect)
        high_terms, low_terms, term2rtw = select_high_low_terms(rtw_df)

        if current_unlabeled.empty:
            print("[INFO] 已无未标注样本，自训练提前结束。")
            break

        newly_labeled, remaining_unlabeled = pseudo_label_unlabeled(
            current_unlabeled, high_terms, low_terms, term2rtw
        )

        if newly_labeled.empty:
            print("[INFO] 本轮未产生新的伪标样本，自训练结束。")
            break

        # 更新集合
        current_labeled = pd.concat([current_labeled, newly_labeled], ignore_index=True)
        current_unlabeled = remaining_unlabeled

    # 4. 使用最终 labeled 集合重新计算 rtw，并导出结果
    print("\n[INFO] 使用最终 (人工 + 伪标) 标注样本重新统计 rtw 并导出烟雾词表...")
    df_defect, df_nondefect, N_defect, N_nondefect = compute_df_from_labeled(current_labeled)
    rtw_df_final = build_rtw_table(df_defect, df_nondefect, N_defect, N_nondefect)
    export_final_stats(rtw_df_final)


if __name__ == "__main__":
    main()


