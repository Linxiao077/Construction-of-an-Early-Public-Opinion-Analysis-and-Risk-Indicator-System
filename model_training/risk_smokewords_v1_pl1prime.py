"""
使用 v1 烟雾词表 (riskword_term_stats_v1.csv) 重新为全量新闻计算 raw_risk(d)，
并生成新一轮 0–3 档伪标签 PL1'，用于训练第二轮模型 M2。

流程: v1 → PL1'

输入:
    - data_combined_4types.xlsx
        * 列:
            - '内容': 文本
            - 'tokenized_text' (可选): 若存在则优先使用
    - riskword_term_stats_v1.csv
        * 列至少包含:
            - 'term'
            - 'risk_avg'

输出:
    - data_with_raw_risk_pl1_v1.xlsx
        * 在原始数据基础上新增:
            - raw_risk_v1
            - pseudo_label_pl1_v1  (0–3)
            - high_conf_pl1_v1     (0/1)
    - pseudo_labels_pl1_M2.csv
        * 仅包含高置信伪标签 PL1':
            - doc_id
            - pseudo_label
            - confidence (raw_risk_v1)
"""

import os
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

from smokewords_stats import tokenize  # 与前一轮保持分词规则一致


DATA_PATH = "data_combined_4types.xlsx"
TEXT_COL = "内容"

V1_STATS_PATH = "riskword_term_stats_v1.csv"

# 0–3 档离散化与高置信阈值（与 risk_smokewords_v0.py 保持一致）
RAW_RISK_BINS = [0.5, 1.5, 2.5]  # 0/1/2/3
HIGH_CONF_LOW = 0.3
HIGH_CONF_HIGH = 2.7


def load_data_with_tokens(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")

    df = pd.read_excel(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少文本列 '{TEXT_COL}'，当前列: {df.columns.tolist()}")

    if "tokenized_text" in df.columns:
        def parse_tokens(x):
            if isinstance(x, str):
                return [t for t in x.split() if t.strip()]
            if isinstance(x, list):
                return [str(t).strip() for t in x if str(t).strip()]
            return tokenize(str(x))

        df["tokens"] = df["tokenized_text"].apply(parse_tokens)
    else:
        df["tokens"] = df[TEXT_COL].astype(str).apply(tokenize)

    return df


def load_term_risk_avg(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 v1 词统计文件: {path}")
    stats = pd.read_csv(path)
    if "term" not in stats.columns or "risk_avg" not in stats.columns:
        raise ValueError(f"{path} 缺少 'term' 或 'risk_avg' 列，当前列: {stats.columns.tolist()}")

    term2risk = {str(row["term"]): float(row["risk_avg"]) for _, row in stats.iterrows()}
    print(f"[INFO] 从 {path} 载入 v1 词表，共 {len(term2risk)} 个 term。")
    return term2risk


def discretize_raw_risk_0_3(raw: float) -> int:
    if raw < RAW_RISK_BINS[0]:
        return 0
    elif raw < RAW_RISK_BINS[1]:
        return 1
    elif raw < RAW_RISK_BINS[2]:
        return 2
    else:
        return 3


def compute_raw_risk_and_pl1_prime(df: pd.DataFrame, term2risk: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_risks: List[float] = []
    pseudo_labels: List[int] = []
    high_conf_flags: List[int] = []

    for _, row in df.iterrows():
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            raw = 0.0
        else:
            tf = Counter(tokens)
            num = 0.0
            den = 0.0
            for t, f in tf.items():
                rv = term2risk.get(t, None)
                if rv is None:
                    continue
                num += f * rv
                den += f
            raw = num / den if den > 0 else 0.0

        raw_risks.append(raw)
        pseudo_labels.append(discretize_raw_risk_0_3(raw))

        if raw < HIGH_CONF_LOW:
            high_conf = 1
        elif raw > HIGH_CONF_HIGH:
            high_conf = 1
        else:
            high_conf = 0
        high_conf_flags.append(high_conf)

    df_out = df.copy()
    df_out["raw_risk_v1"] = raw_risks
    df_out["pseudo_label_pl1_v1"] = pseudo_labels
    df_out["high_conf_pl1_v1"] = high_conf_flags

    pl1_records = []
    for idx, row in df_out.iterrows():
        if row["high_conf_pl1_v1"] == 1:
            pl1_records.append(
                {
                    "doc_id": idx,
                    "pseudo_label": int(row["pseudo_label_pl1_v1"]),
                    "confidence": float(row["raw_risk_v1"]),
                }
            )

    pl1_df = pd.DataFrame(pl1_records)
    return df_out, pl1_df


def main():
    df = load_data_with_tokens(DATA_PATH)
    print(f"[INFO] 读取数据 {DATA_PATH}，共 {len(df)} 行。")

    term2risk = load_term_risk_avg(V1_STATS_PATH)

    df_with_scores, pl1_df = compute_raw_risk_and_pl1_prime(df, term2risk)

    out_data_path = "data_with_raw_risk_pl1_v1.xlsx"
    df_with_scores.to_excel(out_data_path, index=False)
    print(f"[INFO] 已保存附加 raw_risk_v1 / PL1' 的数据到: {out_data_path}")

    pl1_path = "pseudo_labels_pl1_M2.csv"
    pl1_df.to_csv(pl1_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存用于 M2 的高置信伪标签 PL1' 到: {pl1_path} (共 {len(pl1_df)} 条)")


if __name__ == "__main__":
    main()


