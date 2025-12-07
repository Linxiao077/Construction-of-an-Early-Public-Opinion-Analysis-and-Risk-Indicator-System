"""
基于 BERT / RoBERTa 等预训练模型，对候选“烟雾词”做语义扩展与合并。

输入:
    - smokewords_term_stats.csv
        * 由 smokewords_stats.py 生成
        * 至少包含列: term, df_defect, df_normal, df_total, score_stat

输出:
    - smokewords_candidates_merged.jsonl
        每行一个 JSON 对象，字段示例:
        {
          "term": "自燃",
          "freq_defect": 120,
          "freq_normal": 3,
          "df_total": 123,
          "score_stat": 3.21,
          "best_seed": "起火",
          "sim_seed": 0.87,
          "source": "both"   # stat / embed / both
        }

说明:
    - 统计法部分: 直接来自 smokewords_term_stats.csv
    - embedding 扩展部分:
        * 拼接种子词列表 (按 A/B/C/D/E/F/G 类)
        * 用 BERT 得到每个 term 的向量表示 (CLS 向量)
        * 对每个种子词，找到余弦相似度最高的前 K 个词，过滤非汽车相关 / 低相似度词
        * 将扩展出的词与统计法高分词表合并，记录 sim_seed 与 best_seed
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer


STATS_PATH = "data_combined.xlsx"

# 若已在本地完成 MLM 预训练，可使用该目录；否则自动回退到 "bert-base-chinese"
#DEFAULT_MODEL_DIR = "./bert-car-recall-pretrained"
FALLBACK_MODEL_NAME = "bert-base-chinese"

# 每个种子词要找的相似词个数
TOP_K_PER_SEED = 50

# 相似度阈值 (余弦相似度)
SIM_THRESHOLD = 0.6


# ========== 手工种子词 (可根据需要继续扩充 / 调整) ==========

SEED_TERMS_BY_CATEGORY = {
    "A": [
        "起火",
        "自燃",
        "爆炸",
        "制动失灵",
        "刹车失灵",
        "转向失控",
        "高速熄火",
        "车辆失控",
        "翻车",
        "追尾事故",
    ],
    "B": [
        "制动系统异常",
        "转向系统故障",
        "发动机熄火",
        "油路泄漏",
        "刹车偏软",
        "制动力不足",
        "转向发卡",
        "转向失灵",
        "制动踏板偏硬",
    ],
    "C": [
        "异响",
        "抖动",
        "渗油",
        "渗水",
        "空调不制冷",
        "车机卡顿",
        "顿挫",
        "油耗偏高",
        "噪音大",
    ],
    "D": [
        "启动调查",
        "缺陷调查",
        "责令召回",
        "监管约谈",
        "发布通告",
        "缺陷产品管理中心",
        "监管部门介入",
    ],
    "E": [
        "大量投诉",
        "多起事故",
        "多车事故",
        "涉及万辆",
        "涉及多批次",
        "大规模事故",
        "集中爆发",
    ],
    "F": [
        "发布召回公告",
        "启动召回",
        "实施召回",
        "扩大召回范围",
        "主动召回",
        "被动召回",
        "紧急召回",
    ],
    "G": [
        "已完成召回",
        "排除隐患",
        "不存在批量缺陷",
        "历史事件回顾",
    ],
}


@dataclass
class TermStat:
    term: str
    freq_defect: int
    freq_normal: int
    df_total: int
    score_stat: float


def load_term_stats(path: str) -> Dict[str, TermStat]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到统计文件: {path}，请先运行 smokewords_stats.py。")

    df = pd.read_csv(path)
    required_cols = {"term", "df_defect", "df_normal", "df_total", "score_stat"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path} 中缺少必要列: {required_cols}, 实际列为: {df.columns.tolist()}")

    stats: Dict[str, TermStat] = {}
    for _, row in df.iterrows():
        term = str(row["term"])
        stats[term] = TermStat(
            term=term,
            freq_defect=int(row["df_defect"]),
            freq_normal=int(row["df_normal"]),
            df_total=int(row["df_total"]),
            score_stat=float(row["score_stat"]),
        )
    print(f"[INFO] 从 {path} 读取到 {len(stats)} 个 term 的统计信息。")
    return stats


def load_model_and_tokenizer() -> Tuple[BertModel, BertTokenizer, torch.device]:
    if os.path.isdir(DEFAULT_MODEL_DIR):
        model_name = DEFAULT_MODEL_DIR
        print(f"[INFO] 使用本地预训练模型: {DEFAULT_MODEL_DIR}")
    else:
        model_name = FALLBACK_MODEL_NAME
        print(f"[WARN] 未找到本地预训练模型目录，回退到: {FALLBACK_MODEL_NAME}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def encode_terms(
    terms: List[str], model: BertModel, tokenizer: BertTokenizer, device: torch.device, batch_size: int = 64
) -> Dict[str, np.ndarray]:
    """
    对一批词 / 短语做编码，返回 term -> 向量 (CLS 表示) 的字典。
    """
    vectors: Dict[str, np.ndarray] = {}
    n = len(terms)
    for start in range(0, n, batch_size):
        batch_terms = terms[start : start + batch_size]
        inputs = tokenizer(
            batch_terms,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        cls_embeddings = cls_embeddings.cpu().numpy()
        for t, vec in zip(batch_terms, cls_embeddings):
            # 归一化，便于直接做余弦相似度
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vectors[t] = vec / norm

    print(f"[INFO] 已编码 {len(vectors)} 个 term 的向量。")
    return vectors


def build_seed_embeddings(
    model: BertModel, tokenizer: BertTokenizer, device: torch.device
) -> Dict[str, np.ndarray]:
    """
    为所有种子词构建向量表示。
    """
    all_seeds: List[str] = []
    for terms in SEED_TERMS_BY_CATEGORY.values():
        all_seeds.extend(terms)
    all_seeds = sorted(set(all_seeds))

    print(f"[INFO] 共 {len(all_seeds)} 个种子词需要编码。")
    seed_vecs = encode_terms(all_seeds, model, tokenizer, device)
    return seed_vecs


def find_neighbors_for_seeds(
    seed_vecs: Dict[str, np.ndarray],
    term_vecs: Dict[str, np.ndarray],
    top_k: int,
    sim_threshold: float,
) -> Dict[str, Tuple[str, float]]:
    """
    对每个词 w，在所有种子中找到相似度最高的种子及其相似度。

    返回:
        term -> (best_seed, sim_seed)
    """
    if not seed_vecs or not term_vecs:
        return {}

    seed_terms = list(seed_vecs.keys())
    seed_matrix = np.stack([seed_vecs[t] for t in seed_terms], axis=0)  # [S, H]

    term_list = list(term_vecs.keys())
    term_matrix = np.stack([term_vecs[t] for t in term_list], axis=0)  # [T, H]

    # 余弦相似度: term_matrix @ seed_matrix^T
    sim_matrix = np.matmul(term_matrix, seed_matrix.T)  # [T, S]

    neighbor_info: Dict[str, Tuple[str, float]] = {}

    for i, term in enumerate(term_list):
        sims = sim_matrix[i]  # [S]
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        if max_sim < sim_threshold:
            continue
        best_seed = seed_terms[max_idx]
        neighbor_info[term] = (best_seed, max_sim)

    print(f"[INFO] 共为 {len(neighbor_info)} 个 term 找到满足阈值的种子相似度。")
    return neighbor_info


def main():
    # 1. 加载统计结果
    stats = load_term_stats(STATS_PATH)

    # 2. 加载 BERT 模型与 tokenizer
    model, tokenizer, device = load_model_and_tokenizer()

    # 3. 为所有统计中出现的 term 编码
    all_terms = list(stats.keys())
    term_vecs = encode_terms(all_terms, model, tokenizer, device, batch_size=64)

    # 4. 为种子词编码
    seed_vecs = build_seed_embeddings(model, tokenizer, device)

    # 5. 计算每个 term 与各种子词的最大相似度
    neighbor_info = find_neighbors_for_seeds(
        seed_vecs=seed_vecs,
        term_vecs=term_vecs,
        top_k=TOP_K_PER_SEED,
        sim_threshold=SIM_THRESHOLD,
    )

    # 6. 合并统计信息与 embedding 相似度，输出 JSONL
    output_path = "smokewords_candidates_merged.jsonl"
    num_stat_only = 0
    num_embed_only = 0
    num_both = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for term, st in stats.items():
            best_seed: Optional[str] = None
            sim_seed: Optional[float] = None
            if term in neighbor_info:
                best_seed, sim_seed = neighbor_info[term]
                source = "both"
                num_both += 1
            else:
                source = "stat"
                num_stat_only += 1

            record = {
                "term": term,
                "freq_defect": st.freq_defect,
                "freq_normal": st.freq_normal,
                "df_total": st.df_total,
                "score_stat": st.score_stat,
                "best_seed": best_seed,
                "sim_seed": sim_seed,
                "source": source,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 可选: 如希望把“仅通过 embedding 扩展出来、统计表中没有的词”也加入，
        # 可以在此处补充逻辑（例如 term_vecs 中存在但 stats 中不存在的项）。

    print(f"[INFO] 已输出合并候选烟雾词表到: {output_path}")
    print(f"[INFO] 其中: source='both'  {num_both} 条, source='stat' {num_stat_only} 条, embed-only {num_embed_only} 条。")


if __name__ == "__main__":
    main()


