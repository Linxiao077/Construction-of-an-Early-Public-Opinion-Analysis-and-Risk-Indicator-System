"""
基于 README_risktraining_optimized.md 的任务 5–6：

任务 5：在人工标注 + PL1 上训练四分类风险模型 M1 (0–3)
任务 6：使用模型 M1 对全量新闻推理 → 生成新伪标签 PL2

数据依赖:
    1) data_combined_4types.xlsx
        - 必需列:
            * '内容': 文本
            * 'risk' : 人工标注风险 (0–3) 或 NaN
    2) pseudo_labels_pl1.csv
        - 由 risk_smokewords_v0.py 生成
        - 列:
            * doc_id
            * pseudo_label ∈ {0..3}
            * confidence (raw_risk)

输出:
    1) 训练好的 BERT 分类模型目录: ./bert-risk-model-M1
    2) 全量新闻的预测结果: risk_predictions_M1.xlsx
        - 列:
            * doc_id
            * text
            * pred_label
            * confidence
    3) 高置信伪标签 PL2: pseudo_labels_pl2.csv
        - 列:
            * doc_id
            * pseudo_label
            * confidence
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_PATH = "data_combined_4types.xlsx"
PL1_PATH = "pseudo_labels_pl1.csv"

TEXT_COL = "内容"
RISK_COL = "risk"

MODEL_NAME = "bert-base-chinese"  # 可换成本地预训练目录
OUTPUT_DIR = "./bert-risk-model-M1"

# PL2 高置信阈值
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
        raise FileNotFoundError(f"未找到 PL1 文件: {PL1_PATH}，请先运行 risk_smokewords_v0.py。")
    pl1 = pd.read_csv(PL1_PATH)

    return df, pl1


def build_training_examples(df: pd.DataFrame, pl1: pd.DataFrame) -> List[TextLabelExample]:
    """
    构造训练样本:
        - 优先使用人工标注 risk
        - 对未标注且在 PL1 中有高置信伪标签的样本，使用 pseudo_label_pl1
    """
    df = df.copy()

    # 标记人工标签
    has_human = df[RISK_COL].notna()

    # 处理 PL1 伪标签
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

    print(f"[INFO] 构造训练样本 {len(examples)} 条 "
          f"(人工标注 {has_human.sum()} 条, 使用 PL1 伪标签 {len(examples) - has_human.sum()} 条可能略少/重叠)。")
    return examples


def train_model_M1(examples: List[TextLabelExample]) -> Tuple[Trainer, BertTokenizer]:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 划分训练 / 验证集
    train_ex, val_ex = train_test_split(examples, test_size=0.1, random_state=42, stratify=[e.label for e in examples])

    train_ds = RiskDataset(train_ex, tokenizer)
    val_ds = RiskDataset(val_ex, tokenizer)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

    def compute_metrics(eval_pred):
        """
        在验证集上计算 Accuracy 和 Macro-F1，用于随训练过程记录。
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=30,  # 每 30 step 记录一次训练 loss
        eval_strategy="steps",  # 按 step 触发验证
        eval_steps=30,  # 在 step=30,60,90,120,150 ... 计算一次验证集指标（包含 Macro-F1）
        save_strategy="steps",
        save_steps=30,  # 保存步长也设为 30，满足 load_best_model_at_end 的约束
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
        compute_metrics=compute_metrics,
    )

    print("[INFO] 开始训练 BERT 风险模型 M1 ...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] 训练完成，模型已保存到: {OUTPUT_DIR}")

    # 训练完成后，根据 Trainer 的 log_history 绘制 loss 与 Macro-F1 曲线
    plot_training_curves(trainer.state.log_history, output_prefix="M1_training")

    return trainer, tokenizer


def plot_training_curves(
    log_history: List[dict],
    output_prefix: str = "M1_training",
) -> None:
    """
    基于 Trainer 的 log_history 绘制：
        - 训练 loss vs. global step
        - 验证 loss vs. global step
        - 验证 Macro-F1 vs. global step
    """
    train_steps: List[int] = []
    train_losses: List[float] = []

    eval_steps: List[int] = []
    eval_losses: List[float] = []
    eval_macro_f1: List[float] = []

    for entry in log_history:
        # 训练阶段的 loss 日志
        if "loss" in entry and "learning_rate" in entry:
            step = entry.get("step")
            loss = entry.get("loss")
            if step is not None and loss is not None:
                train_steps.append(step)
                train_losses.append(loss)

        # 验证阶段的指标日志
        if "eval_loss" in entry:
            step = entry.get("step")
            loss = entry.get("eval_loss")
            macro_f1 = entry.get("eval_macro_f1")
            if step is not None and loss is not None:
                eval_steps.append(step)
                eval_losses.append(loss)
                # Macro-F1 可能在早期未记录到，这里做一次空值判断
                if macro_f1 is not None:
                    eval_macro_f1.append(macro_f1)
                else:
                    eval_macro_f1.append(np.nan)

    if not train_steps and not eval_steps:
        print("[WARN] log_history 中没有找到训练或验证日志，无法绘制曲线。")
        return

    plt.figure(figsize=(10, 6))

    # (1) 训练 loss 与 验证 loss
    plt.subplot(2, 1, 1)
    if train_steps:
        plt.plot(train_steps, train_losses, marker="o", label="Train loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, marker="s", label="Val loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss (M1)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) 验证 Macro-F1 曲线
    plt.subplot(2, 1, 2)
    if eval_steps:
        plt.plot(eval_steps, eval_macro_f1, marker="o", color="tab:orange")
    plt.xlabel("Global step")
    plt.ylabel("Macro-F1")
    plt.title("Validation Macro-F1 (M1)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = f"{output_prefix}_curves_from_training.png"
    plt.savefig(png_path, dpi=150)
    print(f"[INFO] 已保存训练过程曲线图到: {png_path}")


def predict_all_docs(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df[TEXT_COL].astype(str).tolist()
    preds = []
    confidences = []

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
            logits = outputs.logits  # [B, 5]
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
    print(f"[INFO] PL2 高置信伪标签数量: {len(pl2)} (阈值={PL2_CONF_THRESHOLD})")
    return pl2


def main():
    # 1. 读取数据 & PL1
    df, pl1 = load_data_and_pl1()

    # 2. 构建训练样本 (人工标注 + PL1)
    examples = build_training_examples(df, pl1)
    if not examples:
        raise ValueError("没有可用的训练样本，请检查 risk 和 PL1。")

    # 3. 训练模型 M1
    trainer, tokenizer = train_model_M1(examples)

    # 4. 对全量新闻推理
    pred_df = predict_all_docs(df, OUTPUT_DIR)

    pred_out_path = "risk_predictions_M1.xlsx"
    pred_df.to_excel(pred_out_path, index=False)
    print(f"[INFO] 已保存全量预测结果到: {pred_out_path}")

    # 5. 构建新伪标签 PL2
    pl2_df = build_PL2_from_predictions(pred_df)
    pl2_path = "pseudo_labels_pl2.csv"
    pl2_df.to_csv(pl2_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存 PL2 伪标签到: {pl2_path}")


if __name__ == "__main__":
    main()


