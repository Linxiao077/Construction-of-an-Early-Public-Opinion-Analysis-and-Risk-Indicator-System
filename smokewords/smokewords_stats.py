"""
基于带 risk 标签的新闻数据，挖掘候选“召回烟雾词”（统计法部分）。

数据来源:
    - cleaned_risk_data.xlsx
      * 列 '内容': 新闻正文文本（中文）
      * 列 'risk' : 人工标注召回风险等级，0–4（0=无风险，4=最高风险或已召回）

主要步骤:
    1. 读取数据，区分“缺陷新闻”和“普通新闻”
       - 默认: risk ∈ {2,3,4} 视为缺陷新闻（可根据需要调整 DEFECT_LEVELS）
    2. 中文分词 + 去停用词 / 品牌名 / 噪音词
    3. 统计文档频数:
         df_defect(term) : 含该词的缺陷新闻数
         df_normal(term) : 含该词的普通新闻数
         df_total(term)  : 总文档频数
    4. 计算统计关联度:
         score(term) = log( P(term|defect) / P(term|normal) )
    5. 过滤低频与噪音词，按 score 降序排序，输出:
       - smokewords_term_stats.csv : 所有通过过滤的词的统计量
       - smokewords_candidates_stat_topN.csv : 统计分最高的前 N 个候选烟雾词
"""

import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

import jieba
import numpy as np
import pandas as pd


# ===================== 配置区 =====================

DATA_PATH = "cleaned_risk_data_fullymarked.xlsx"
TEXT_COL = "内容"
RISK_COL = "risk"

# 将哪些 risk 等级视为“缺陷新闻”（可根据需要调整）
DEFECT_LEVELS = {2, 3, 4}

# 最小文档频数阈值（df_total >= MIN_DF 才纳入统计）
MIN_DF = 5

# 导出候选词数量
TOP_N = 500

# 停用词文件（若不存在则忽略）
STOPWORDS_PATH = "chinese_stopwords.txt"


def load_stopwords(path: str) -> Set[str]:
    """加载停用词表，若文件不存在则返回空集合。"""
    if not os.path.isfile(path):
        print(f"[WARN] 未找到停用词文件: {path}，将不使用额外停用词。")
        return set()
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


def load_brand_set() -> Set[str]:
    """
    尝试从 NLP_car.py 中导入 CAR_BRANDS 作为品牌词表。
    若导入失败，则返回空集合（不强制依赖）。
    """
    try:
        from NLP_car import CAR_BRANDS  # type: ignore

        brand_set = set(map(str, CAR_BRANDS))
        print(f"[INFO] 成功从 NLP_car 导入 {len(brand_set)} 个品牌名称。")
        return brand_set
    except Exception:
        print("[WARN] 未能从 NLP_car 导入 CAR_BRANDS，将不额外过滤品牌名。")
        return set()


STOPWORDS: Set[str] = load_stopwords(STOPWORDS_PATH)
BRANDS: Set[str] = load_brand_set()


def tokenize(text: str) -> List[str]:
    """简单中文分词 + 基本清洗。"""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    words = jieba.lcut(text)
    cleaned = []
    for w in words:
        w = w.strip()
        # 长度过滤
        if len(w) <= 1:
            continue
        # 去除数字和纯英文
        if w.isdigit():
            continue
        if all("a" <= ch.lower() <= "z" for ch in w if ch.isalpha()):
            continue
        # 停用词和品牌过滤
        if w in STOPWORDS:
            continue
        if w in BRANDS:
            continue
        cleaned.append(w)
    return cleaned


def iter_docs(df: pd.DataFrame) -> Iterable[Tuple[bool, List[str]]]:
    """
    遍历文档，返回 (is_defect, tokens)。
    is_defect: 是否为缺陷新闻
    tokens   : 分词后的结果
    """
    for _, row in df.iterrows():
        text = row.get(TEXT_COL, "")
        risk = row.get(RISK_COL, None)
        if pd.isna(text) or pd.isna(risk):
            continue
        try:
            risk_value = int(risk)
        except Exception:
            continue
        is_defect = risk_value in DEFECT_LEVELS
        tokens = tokenize(str(text))
        if not tokens:
            continue
        yield is_defect, tokens


def compute_term_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计 df_defect / df_normal / df_total，并计算 score(term)。
    返回包含所有通过过滤的 term 的 DataFrame。
    """
    df_defect = Counter()
    df_normal = Counter()

    N_defect = 0
    N_normal = 0

    for is_defect, tokens in iter_docs(df):
        unique_terms = set(tokens)  # 文档频数，不计重复出现
        if is_defect:
            N_defect += 1
            for t in unique_terms:
                df_defect[t] += 1
        else:
            N_normal += 1
            for t in unique_terms:
                df_normal[t] += 1

    if N_defect == 0 or N_normal == 0:
        raise ValueError(f"N_defect={N_defect}, N_normal={N_normal}, 至少一类为空，无法计算关联度。")

    print(f"[INFO] 缺陷新闻数 N_defect   = {N_defect}")
    print(f"[INFO] 普通新闻数 N_normal  = {N_normal}")
    print(f"[INFO] 总文档数   N_total   = {N_defect + N_normal}")

    all_terms = set(df_defect.keys()) | set(df_normal.keys())
    records = []

    alpha = 0.5  # 拉普拉斯平滑，避免除零

    for term in all_terms:
        a = df_defect.get(term, 0)
        b = df_normal.get(term, 0)
        df_total = a + b
        if df_total < MIN_DF:
            continue

        p_def = (a + alpha) / (N_defect + 2 * alpha)
        p_norm = (b + alpha) / (N_normal + 2 * alpha)

        # 避免极端数值
        ratio = max(p_def / p_norm, 1e-8)
        score = float(math.log(ratio))

        records.append(
            {
                "term": term,
                "df_defect": int(a),
                "df_normal": int(b),
                "df_total": int(df_total),
                "p_defect": float(p_def),
                "p_normal": float(p_norm),
                "score_stat": score,
            }
        )

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        raise ValueError("在当前 MIN_DF 和过滤条件下，没有任何词通过过滤，请调低 MIN_DF 或检查数据。")

    stats_df = stats_df.sort_values("score_stat", ascending=False).reset_index(drop=True)
    return stats_df


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    if TEXT_COL not in df.columns or RISK_COL not in df.columns:
        raise ValueError(f"数据中必须包含列: '{TEXT_COL}' 和 '{RISK_COL}'，当前列为: {df.columns.tolist()}")

    print(f"[INFO] 读取数据 {DATA_PATH}，共 {len(df)} 行。")

    stats_df = compute_term_stats(df)

    # 输出完整统计表
    stats_path = "smokewords_term_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存完整词统计到: {stats_path} (共 {len(stats_df)} 个 term)")

    # 输出前 TOP_N 候选词
    topN = min(TOP_N, len(stats_df))
    top_df = stats_df.head(topN).copy()
    top_path = f"smokewords_candidates_stat_top{topN}.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存统计法候选烟雾词 Top-{topN} 到: {top_path}")

    print("\n[INFO] 示例前 20 个候选词：")
    print(top_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()


