"""
使用大模型对候选“烟雾词”进行过滤与分类，生成最终烟雾词表。

输入:
    - smokewords_candidates_merged.jsonl
        * 由 smokewords_embed_expand.py 生成
        * 每行包含: term, freq_defect, freq_normal, df_total, score_stat, best_seed, sim_seed, source

处理:
    - 按批次将候选词及其统计特征传给大模型
    - 让大模型判断:
        * 是否保留为“烟雾词”
        * 分类: A/B/C/D/E/F/G/OTHER
        * 给出简短说明

输出:
    - smokewords_final.jsonl
        每行字段示例:
        {
          "term": "自燃",
          "category": "A",
          "description": "描述车辆自燃或起火等严重安全后果",
          "source": "both",
          "freq_defect": 120,
          "freq_normal": 3,
          "score_stat": 3.21,
          "sim_seed": 0.87,
          "keep_or_drop": "keep"
        }

说明:
    - 该脚本以 OpenAI 接口为例，需在环境变量中配置 OPENAI_API_KEY。
    - 你也可以将 call_llm 函数替换为任意国产大模型 / 自有模型的调用逻辑。
"""

import json
import os
import time
from typing import Dict, List

try:
    import openai
except ImportError:
    openai = None  # type: ignore


INPUT_PATH = "smokewords_candidates_merged.jsonl"
OUTPUT_PATH = "smokewords_final.jsonl"

# 每批发送给大模型的候选词数量（可根据 token 限制调整）
BATCH_SIZE = 40

# 使用的模型名称（根据实际账户支持情况调整）
OPENAI_MODEL_NAME = "gpt-4.1-mini"


PROMPT_INSTRUCTIONS = """
你是一名汽车质量与召回领域的数据挖掘专家，正在整理“召回烟雾词表”。

烟雾词是指：在与车辆缺陷、事故、安全风险和召回相关的新闻中，具有较强指示性的词或短语。

根据以下字段，对每个候选词进行判断：
- term: 词或短语
- freq_defect: 在缺陷新闻中的文档频数
- freq_normal: 在普通新闻中的文档频数
- df_total: 总文档频数
- score_stat: 统计关联度 (score(term) = log(P(term|defect) / P(term|normal))，数值越大越偏向缺陷新闻
- sim_seed: 与已知种子词的最大语义相似度（0~1，数值越大语义越接近缺陷/召回相关种子）
- source: 词的来源（stat/embed/both）

你的任务：
1. 判断该词是否保留为“烟雾词”（keep 或 drop）。
2. 如保留，则按如下类别之一进行分类:
   - A: 严重安全后果（如起火、自燃、爆炸、制动失灵、转向失控、高速熄火、车辆失去控制等）
   - B: 安全相关功能故障（如制动系统异常、转向系统故障、发动机熄火、油路泄漏、安全气囊故障等）
   - C: 一般质量与性能问题（如异响、抖动、渗油、渗水、空调不制冷、车机系统卡顿等）
   - D: 官方调查与监管行为（如启动调查、责令召回、缺陷调查、监管约谈等）
   - E: 规模与频次描述（如大量投诉、多起事故、涉及×万辆车辆、多批次问题等）
   - F: 召回动作（如发布召回公告、启动召回、实施召回、扩大召回范围等）
   - G: 否定 / 历史语境（如已完成召回且排除隐患、不存在批量缺陷、历史事件回顾等）
   - OTHER: 非烟雾词，与车辆缺陷、事故、召回无直接关系，或歧义太大不适合做规则关键词

3. 为每个保留词写一句中文说明 (description)，简要说明该词对应的典型风险或场景。

输出格式:
对于输入的每个 term，请严格按 JSON 对象列表返回，列表长度与输入候选词数量一致。
每个对象包含字段:
  - term: 原始词
  - keep_or_drop: "keep" 或 "drop"
  - category: "A"/"B"/"C"/"D"/"E"/"F"/"G"/"OTHER"
  - description: 简短中文说明（若 drop，可写空字符串）
仅返回 JSON 数组，不要添加其他解释性文字。
"""


def call_llm(batch_items: List[Dict]) -> List[Dict]:
    """
    调用大模型对一批候选词进行过滤与分类。
    返回与 batch_items 等长的结果列表，每个元素包含 term/keep_or_drop/category/description。
    """
    if openai is None:
        raise ImportError("未安装 openai 库，请先 pip install openai，或自行替换为其他大模型 SDK。")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("未在环境变量中找到 OPENAI_API_KEY，请配置后再运行。")

    openai.api_key = api_key

    content = {
        "candidates": batch_items,
    }

    messages = [
        {"role": "system", "content": PROMPT_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(content, ensure_ascii=False),
        },
    ]

    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        temperature=0.0,
    )
    text = resp["choices"][0]["message"]["content"]

    # 期望模型返回一个 JSON 数组
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回内容非合法 JSON，需人工检查:\n{text}") from e

    if not isinstance(result, list):
        raise ValueError(f"LLM 返回结果不是列表: {result}")

    return result


def read_candidates(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到候选词文件: {path}，请先运行 smokewords_embed_expand.py。")

    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "term" in obj:
                    items.append(obj)
            except Exception:
                continue

    print(f"[INFO] 从 {path} 读取到 {len(items)} 个候选词条。")
    return items


def main():
    candidates = read_candidates(INPUT_PATH)
    total = len(candidates)
    if total == 0:
        print("[WARN] 候选词为空，终止。")
        return

    out_f = open(OUTPUT_PATH, "w", encoding="utf-8")

    processed = 0
    for start in range(0, total, BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]

        # 只将大模型需要的字段传给它
        llm_batch_input = [
            {
                "term": item["term"],
                "freq_defect": item.get("freq_defect", 0),
                "freq_normal": item.get("freq_normal", 0),
                "df_total": item.get("df_total", 0),
                "score_stat": item.get("score_stat", 0.0),
                "sim_seed": item.get("sim_seed", None),
                "source": item.get("source", "stat"),
            }
            for item in batch
        ]

        print(f"[INFO] 处理 batch {start // BATCH_SIZE + 1}, 大小={len(llm_batch_input)} ...")
        llm_results = call_llm(llm_batch_input)

        if len(llm_results) != len(batch):
            raise ValueError(
                f"LLM 返回数量与输入不一致: 输入 {len(batch)}, 返回 {len(llm_results)}，需要人工检查。"
            )

        # 合并 LLM 输出与原始统计信息，写入最终 JSONL
        for orig, cls in zip(batch, llm_results):
            term = orig["term"]
            if cls.get("term") and cls["term"] != term:
                # 防御性检查: 若返回的 term 与原始不一致，则以原始为准
                cls["term"] = term

            record = {
                "term": term,
                "category": cls.get("category", "OTHER"),
                "description": cls.get("description", ""),
                "source": orig.get("source", "stat"),
                "freq_defect": orig.get("freq_defect", 0),
                "freq_normal": orig.get("freq_normal", 0),
                "df_total": orig.get("df_total", 0),
                "score_stat": orig.get("score_stat", 0.0),
                "sim_seed": orig.get("sim_seed", None),
                "best_seed": orig.get("best_seed", None),
                "keep_or_drop": cls.get("keep_or_drop", "keep"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed += len(batch)
        print(f"[INFO] 当前已处理 {processed}/{total} 条候选词。")

        # 简单的限速保护，避免触发 API 频率限制
        time.sleep(1.0)

    out_f.close()
    print(f"[INFO] 已写入最终烟雾词表到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


