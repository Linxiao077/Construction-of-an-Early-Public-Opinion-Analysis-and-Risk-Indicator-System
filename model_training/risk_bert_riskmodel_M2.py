"""
第二轮风险模型 M2 训练脚本：v1 → PL1' → M2 → PL2'

本脚本对应 README_risktraining_optimized.md 中的任务 5–6，在“v1 词表生成的 PL1'”基础上，
训练新的四分类风险模型 M2，并对全量新闻推理生成 PL2'。

输入:
    - data_combined_4types.xlsx
        * 列:
            - '内容': 文本
            - 'risk' : 人工标注风险 (0–3) 或 NaN
    - pseudo_labels_pl1_M2.csv
        * 由 risk_smokewords_v1_pl1prime.py 生成
        * 列:
            - doc_id
            - pseudo_label ∈ {0..3}
            - confidence (raw_risk_v1)

输出:
    - ./bert-risk-model-M2/    : 第二轮 BERT 风险模型权重与 tokenizer
    - risk_predictions_M2.xlsx : 全量新闻的 0–3 风险预测
    - pseudo_labels_pl2_M2.csv : 高置信 PL2' 伪标签 (doc_id, pseudo_label, confidence)
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_PATH = "data_combined_4types.xlsx"
PL1_PATH = "pseudo_labels_pl1_M2.csv"

TEXT_COL = "内容"
RISK_COL = "risk"

MODEL_NAME = "bert-base-chinese"  # 可换成本地 v1 模型目录，如 "./bert-risk-model-M1"
OUTPUT_DIR = "./bert-risk-model-M2"

PL2_CONF_THRESHOLD = 0.85


@dataclass
class TextLabelExample:
    text: str
    label: int


class RiskDataset(Dataset):
    def __init__(self, examples: List[TextLabelExample], tokenizer: BertTokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.long)
        return item


def load_data_and_pl1() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    if TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少文本列 '{TEXT_COL}'，当前列: {df.columns.tolist()}")

    if not os.path.isfile(PL1_PATH):
        raise FileNotFoundError(f"未找到 PL1' 文件: {PL1_PATH}，请先运行 risk_smokewords_v1_pl1prime.py。")
    pl1 = pd.read_csv(PL1_PATH)

    return df, pl1


def build_training_examples(df: pd.DataFrame, pl1: pd.DataFrame) -> List[TextLabelExample]:
    """
    构造第二轮训练样本:
        - 优先使用人工标注 risk
        - 对未标注且在 PL1' 中有高置信伪标签的样本，使用 pseudo_label_pl1_M2
    """
    df = df.copy()

    has_human = df[RISK_COL].notna()
    pl1_map = {int(row["doc_id"]): int(row["pseudo_label"]) for _, row in pl1.iterrows()}

    examples: List[TextLabelExample] = []
    for idx, row in df.iterrows():
        text = str(row[TEXT_COL])

        if has_human.loc[idx]:
            try:
                label = int(row[RISK_COL])
            except Exception:
                continue
        else:
            if idx not in pl1_map:
                continue
            label = pl1_map[idx]

        if label < 0 or label > 3:
            continue
        examples.append(TextLabelExample(text=text, label=label))

    print(
        f"[INFO] 构造 M2 训练样本 {len(examples)} 条 "
        f"(人工标注 {has_human.sum()} 条, 使用 PL1' 伪标签 {len(examples) - has_human.sum()} 条可能略少/重叠)。"
    )
    return examples


def train_model_M2(examples: List[TextLabelExample]) -> Tuple[Trainer, BertTokenizer]:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    labels = [e.label for e in examples]
    train_ex, val_ex = train_test_split(examples, test_size=0.1, random_state=42, stratify=labels)

    train_ds = RiskDataset(train_ex, tokenizer)
    val_ds = RiskDataset(val_ex, tokenizer)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print("[INFO] 开始训练第二轮 BERT 风险模型 M2 ...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] M2 训练完成，模型已保存到: {OUTPUT_DIR}")

    return trainer, tokenizer


def predict_all_docs(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df[TEXT_COL].astype(str).tolist()
    preds: List[int] = []
    confidences: List[float] = []

    batch_size = 32
    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # [B, 4]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

        preds.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())

    result_df = pd.DataFrame(
        {
            "doc_id": df.index.to_list(),
            "text": texts,
            "pred_label": preds,
            "confidence": confidences,
        }
    )
    return result_df


def build_PL2_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    high_conf = pred_df[pred_df["confidence"] >= PL2_CONF_THRESHOLD].copy()
    high_conf = high_conf.rename(columns={"pred_label": "pseudo_label"})
    pl2 = high_conf[["doc_id", "pseudo_label", "confidence"]].reset_index(drop=True)
    print(f"[INFO] PL2' 高置信伪标签数量: {len(pl2)} (阈值={PL2_CONF_THRESHOLD})")
    return pl2


def main():
    df, pl1 = load_data_and_pl1()
    examples = build_training_examples(df, pl1)
    if not examples:
        raise ValueError("没有可用的 M2 训练样本，请检查 risk 和 PL1'。")

    train_model_M2(examples)

    pred_df = predict_all_docs(df, OUTPUT_DIR)
    pred_out_path = "risk_predictions_M2.xlsx"
    pred_df.to_excel(pred_out_path, index=False)
    print(f"[INFO] 已保存 M2 全量预测结果到: {pred_out_path}")

    pl2_df = build_PL2_from_predictions(pred_df)
    pl2_path = "pseudo_labels_pl2_M2.csv"
    pl2_df.to_csv(pl2_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存 PL2' 伪标签到: {pl2_path}")


if __name__ == "__main__":
    main()


