#这是处理数据的第三个文件。
#在汽车品牌字段中提取所有品牌，去重并拆分成不同的行，用于风险标记和训练。根据每条新闻的汽车品牌数决定权重（可以不加权重）。

import ast
import pandas as pd

df = pd.read_excel("D:\pycharm\workplace\deduplicated_news.xlsx")

# 1. 把字符串转成真实的 list
def parse_brand_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            # 如果解析失败，可以按逗号拆，或者设为单品牌
            return [x]
    return []

df['品牌列表'] = df['汽车品牌'].apply(parse_brand_list)

# 2. 记录原始新闻 id
df = df.reset_index().rename(columns={'index': 'news_id'})

# 3. explode 按品牌展开
df_brand = df.explode('品牌列表')
df_brand = df_brand.rename(columns={'品牌列表': '汽车品牌'})
df_brand = df_brand.loc[:, ~df_brand.columns.duplicated(keep='last')]

# 4. 计算每条新闻的品牌数，用于样本权重
# brand_counts = df_brand.groupby('news_id')['汽车品牌'].count()
# print(df_brand.columns)
# print(type(brand_counts))
# print(brand_counts.head())
# df_brand['sample_weight'] = df_brand['news_id'].map(1.0 / brand_counts)

# 5. 可选：主要品牌先设为汽车品牌本身
df_brand['主要品牌'] = df_brand['汽车品牌']

# 6. 按你需要的列顺序重排
cols = [
    'news_id', '媒体名称', '媒体类型', '内容', '日期', '标题', '版面', '作者',
    '链接', '字数', '国家地区/省份', '城市', '数据采集来源', '年月',
    '汽车品牌', '主要品牌',
    'tokenized_text', '编码序列',
    # 召回风险等级（你标注好后加进来）
    'sample_weight'
]
df_brand_final = df_brand[[c for c in cols if c in df_brand.columns]]

# 删去主要品牌列
df_brand_final = df_brand_final.drop(columns=['主要品牌'])
# 删除为空或为 NaN 的行
df_brand_final = df_brand_final[df_brand_final['汽车品牌'].notna() & (df_brand_final['汽车品牌'] != "")]

df_brand_final['risk_level'] = None  # 先占位，后面人工标注['risk_level'] = None  # 先占位，后面人工标注

# 假设 df 中有 index 字段（或者叫 news_id）
df_dedup = df_brand_final.drop_duplicates(subset=['news_id', '汽车品牌'], keep='first')

# 4. 计算每条新闻的品牌数，用于样本权重
brand_counts = df_dedup.groupby('news_id')['汽车品牌'].count()
print(df_dedup .columns)
print(type(brand_counts))
print(brand_counts.head())
df_dedup['sample_weight'] = df_dedup['news_id'].map(1.0 / brand_counts)

# 7. 保存
output_path = r"D:\pycharm\workplace\news_brand_level.xlsx"
df_dedup.to_excel(output_path, index=False)
print(f"品牌展开后的表已保存到: {output_path}")
