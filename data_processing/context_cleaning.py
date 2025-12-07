#这是处理数据的第二个文件
#定义媒体权威度映射字典，分为7个权威度。利用hamming距离找出相似新闻，根据相关策略，保留权威度高，发布时间早的新闻。最后批量处理。

import pandas as pd
import numpy as np
from simhash import Simhash
from datetime import datetime
import re
from collections import defaultdict

# 媒体权威度映射字典
media_authority_ranking = {
    '最高权威媒体': [
        '国家市场监督管理总局',  # 汽车召回官方发布机构
        '国务院发展研究中心信息网',
        '福建省市场监督管理局（知识产权局）',
        '海南省市场监督管理局',
        '通化市人民政府',
        '芜湖市人民政府',

        # 国家级主流媒体
        '人民日报 APP',
        '新华网',
        '人民网',
        '央视网',
        '中国网',
        '光明网',
        '中国青年网',
        '中青在线',
        '中工网',
        '中国新闻社',
        '参考消息网',

        # 专业质量监管
        '中国质量新闻网',
        '中国质量报(数字报)'
    ],  # 第一层级
    '高权威媒体': [
        # 专业财经媒体
        '财联社',
        '第一财经 APP',
        '财经网',
        '澎湃新闻 APP',
        '界面新闻 APP',
        '每日经济新闻 APP',
        '经济观察网',
        '中国经营报',
        '经济参考报',
        '华夏时报',
        '投资者网',

        # 证券权威媒体
        '上海证券报 APP',
        '证券时报 APP',
        '证券时报网',
        '中国证券网',
        '中国基金报',
        '期货日报',
        '中国银行保险报',

        # 专业汽车媒体
        '中国汽车召回网',  # 专业召回信息网站
        '车质网',  # 专业汽车质量平台
        '盖世汽车网',
        '中国汽车报网'
    ],  # 第二层级
    '中等权威媒体': [
        # 主流综合新闻
        '腾讯新闻 APP',
        '网易新闻 APP',
        '新浪新闻 APP',
        '今日头条',
        '百度 APP',
        '凤凰新闻 APP',
        '环球网',

        # 专业资讯平台
        '36氪',
        '虎嗅网 APP',
        '钛媒体网',
        '创业邦 APP',
        '品玩 APP',
        '亿欧网',
        '智通财经网',
        '智通财经 APP',
        '科创板日报',

        # 垂直汽车媒体
        '懂车帝 APP',
        '汽车之家',
        '易车网',
        '太平洋汽车网',
        '爱卡汽车网',
        '新浪汽车',
        '搜狐汽车',
        '网上车市 APP',
        '汽车头条 APP'
    ],  # 第三层级
    '基础权威媒体': [
        # 财经数据服务
        '东方财富网',
        '东方财富 APP',
        '同花顺财经',
        '同花顺 APP',
        '同花顺iFinD APP',
        '雪球',
        '金融界',
        '和讯网',
        '证券之星',
        '华尔街见闻',
        '中新经纬 APP',
        '每经网',

        # 地方主流媒体
        '北京日报 APP',
        '湖北日报 APP',
        '海南日报 APP',
        '南方财经网',
        '红网',
        '云南网',
        '大众网',
        '河北新闻网',
        '荆楚网-湖北日报网',
        '山西晚报',
        '合肥在线',
        '安徽网',
        '河南一百度',

        # 券商研究
        '中信建投证券 APP',
        '国泰君安证券 APP',
        '海通e海通财 APP'
    ],  # 第四层级
    '一般权威媒体': [
        # 财经媒体
        '财经头条 APP',
        '新浪财经 APP',
        '金投网',
        '汇通财经',
        '格隆汇',
        '富途牛牛',
        '大智慧 APP',
        '中国财富 APP',
        '掌证宝天玑版 APP',
        '腾讯自选股 APP',
        'BBAE必贝证券 APP',
        '智远壹户通 APP',
        '天天基金网 APP',
        '见闻VIP APP',
        '英为财情',
        '汇金网',
        '全景网',
        '中财网',

        # 行业垂直
        '我的钢铁网',
        '长江有色金属网',
        '中国钢铁工业信息网',
        '中国产业经济信息网',

        # 专业资讯
        '商业新知',
        'DoNews',
        '未央网',
        '支付百科网',
        '企业时报网'
    ],  # 第五层级
    '较低权威媒体': [
        # 内容聚合平台
        '一点资讯',
        '一点资讯 APP',
        'UC头条 APP',
        'ZAKER APP',
        '趣头条 APP',
        '看点快报 APP',
        '快资讯',
        '云掌财经',
        '中金在线',
        'hao123新闻',
        '手机新浪网',
        '手机搜狐 APP',
        '手机网易网',

        # 专业社交平台
        '知乎',
        '知乎专栏',
        '雪球',  # 兼具社交属性

        # 汽车资讯
        '车讯网',
        '汽车投诉网',
        '电车之家',
        '第1电动',
        '315汽车网',
        '汽车新看点',
        '中国汽车头条'
    ],  # 第六层级
    '最低权威媒体': [
        # 社交娱乐平台
        '新浪微博',
        '抖音 APP',
        '快手 APP',
        '哔哩哔哩APP',
        '小红书 APP',
        '虎扑社区',

        # 论坛社区
        '新浪股市汇',
        '同花顺圈子论坛',
        '东方财富网股吧',
        '太平洋汽车网论坛',
        '汽车之家论坛',
        '股吧 APP',
        '思否',

        # 自媒体与小型网站
        '微信公众号',
        '金融八卦女 APP',
        '财经钻',
        '思维财经',
        '顶尖财经网',
        '百家财富网',
        '中国财讯网',
        '潮起网',
        '财经产业网',
        '文财网',
        '优财网',
        '发现网',
        '成功财经网',
        '博览黄页',
        '最资讯',
        '中投网',

        # 其他
        '花花女性网',
        'Bianews',
        'xw.qq.com',
        '宜春新闻网',
        '钢企网',
        '云聚网',
        '中搜搜悦 APP',
        '中青看点APP',
        '环球TIME APP',
        '央广网 APP',
        '封面',
        '周到上海网',
        '乐居财经 APP',
        '市值风云',
        '面包财经',
        '节点财经',
        '鲸准',
        '烯牛数据',
        '企查查',
        '天眼查',
        '启信宝',
        '爱企查',
        '企信宝'
    ]  # 第七层级
}

# 构建媒体到权威度分数的映射字典
media_authority_map = {}
authority_scores = {
    '最高权威媒体': 1,
    '高权威媒体': 2,
    '中等权威媒体': 3,
    '基础权威媒体': 4,
    '一般权威媒体': 5,
    '较低权威媒体': 6,
    '最低权威媒体': 7
}

for level_name, media_list in media_authority_ranking.items():
    score = authority_scores[level_name]
    for media in media_list:
        media_authority_map[media] = score


def get_media_authority(media_name):
    """获取媒体权威度分数（数值越小权威度越高）"""
    return media_authority_map.get(media_name, 7)  # 默认最低权威度


class AdvancedNewsDeduplicator:
    def __init__(self, simhash_threshold=3, min_content_length=50):
        """
        初始化高级去重器

        Args:
            simhash_threshold: SimHash汉明距离阈值
            min_content_length: 最小内容长度，短于此长度的不参与去重
        """
        self.simhash_threshold = simhash_threshold
        self.min_content_length = min_content_length

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        # 去除HTML标签、特殊字符等
        text = re.sub(r'<[^>]+>', '', str(text))
        text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_simhash(self, text):
        """计算文本的SimHash值"""
        text = self.preprocess_text(text)
        if len(text) < self.min_content_length:
            return None  # 内容太短，不参与去重
        return Simhash(text).value

    def hamming_distance(self, hash1, hash2):
        """计算两个SimHash的汉明距离"""
        if hash1 is None or hash2 is None:
            return float('inf')  # 内容太短的返回无限大距离
        return bin(hash1 ^ hash2).count('1')

    def find_duplicate_groups(self, df, content_col='内容'):
        print("开始计算SimHash...")
        df = df.copy()
        df['simhash'] = df[content_col].apply(self.calculate_simhash)

        print("构建SimHash索引...")
        hash_to_indices = defaultdict(list)
        for idx, simhash in zip(df.index, df['simhash']):
            # 过滤掉 NaN 和 None
            if pd.isna(simhash):
                continue
            # 确保是 int
            simhash = int(simhash)
            hash_to_indices[simhash].append(idx)

        print("寻找相似新闻组...")
        duplicate_groups = []
        processed_hashes = set()

        for base_hash, base_indices in hash_to_indices.items():
            if base_hash in processed_hashes:
                continue

            current_group = set(base_indices)

            for other_hash, other_indices in hash_to_indices.items():
                if other_hash in processed_hashes:
                    continue
                if other_hash == base_hash:
                    continue

                if self.hamming_distance(base_hash, other_hash) <= self.simhash_threshold:
                    current_group.update(other_indices)
                    processed_hashes.add(other_hash)

            if len(current_group) > 1:
                duplicate_groups.append(list(current_group))

            processed_hashes.add(base_hash)

        print(f"找到 {len(duplicate_groups)} 组重复新闻")
        return duplicate_groups

    def select_best_article(self, df, duplicate_indices, date_col='日期', word_count_col='字数'):
        """
        从重复新闻组中选择最佳文章

        选择策略（优先级顺序）：
        1. 媒体权威度（数值越小越好）
        2. 发布时间（越早越好）
        3. 文章字数（越多越好）
        4. 媒体类型优先级（官方机构 > 专业媒体 > 综合媒体 > 社交平台）
        """
        if len(duplicate_indices) == 0:
            return None

        articles = df.loc[duplicate_indices].copy()

        # 计算权威度分数
        articles['authority_score'] = articles['媒体名称'].apply(get_media_authority)

        # 媒体类型优先级映射
        media_type_priority = {
            '官方机构': 1,
            '财经媒体': 2,
            '垂直汽车': 3,
            '专业资讯': 4,
            '传统媒体转型': 5,
            '综合新闻': 6,
            '行业垂直': 7,
            '地方媒体': 8,
            '券商平台': 9,
            '内容聚合': 10,
            '社交平台': 11,
            '其他': 12
        }

        articles['type_priority'] = articles.get('媒体类型', '其他').apply(
            lambda x: media_type_priority.get(x, 12)
        )

        # 多重排序：权威度 → 日期 → 字数 → 媒体类型
        articles_sorted = articles.sort_values(
            by=['authority_score', date_col, word_count_col, 'type_priority'],
            ascending=[True, True, False, True]
        )

        best_article_idx = articles_sorted.index[0]

        # 记录选择原因
        best_article = articles_sorted.iloc[0]
        selection_reason = self._get_selection_reason(best_article, articles_sorted)

        return best_article_idx, selection_reason

    def _get_selection_reason(self, best_article, all_articles):
        """生成选择原因说明"""
        reasons = []

        # 检查权威度是否最优
        min_authority = all_articles['authority_score'].min()
        if best_article['authority_score'] == min_authority:
            reasons.append(f"权威度最高({best_article['authority_score']})")

        # 检查是否最早发布
        min_date = all_articles['日期'].min()
        if best_article['日期'] == min_date:
            reasons.append("发布时间最早")

        # 检查字数是否最多
        max_words = all_articles['字数'].max()
        if best_article['字数'] == max_words:
            reasons.append("内容最完整")

        return " + ".join(reasons) if reasons else "默认选择"

    def deduplicate_news(self, df, content_col='内容', date_col='日期', word_count_col='字数'):
        """
        主函数：执行完整的新闻去重流程
        """
        print("开始新闻去重处理...")
        print(f"原始数据量: {len(df)} 条")

        # 数据预处理
        df = df.copy()
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 找出重复组
        duplicate_groups = self.find_duplicate_groups(df, content_col)

        # 处理重复组
        to_remove_indices = set()
        selection_log = []
        stats = {
            'total_groups': len(duplicate_groups),
            'authority_distribution': defaultdict(int),
            'removed_count': 0
        }

        for group_id, group_indices in enumerate(duplicate_groups):
            best_idx, reason = self.select_best_article(df, group_indices, date_col, word_count_col)

            if best_idx is not None:
                best_article = df.loc[best_idx]
                authority_level = get_media_authority(best_article['媒体名称'])

                # 记录统计信息
                stats['authority_distribution'][authority_level] += 1

                # 记录选择日志
                selection_log.append({
                    'group_id': group_id,
                    'best_index': best_idx,
                    'best_media': best_article['媒体名称'],
                    'authority_level': authority_level,
                    'selection_reason': reason,
                    'duplicate_count': len(group_indices),
                    'removed_count': len(group_indices) - 1
                })

                # 标记删除
                for idx in group_indices:
                    if idx != best_idx:
                        to_remove_indices.add(idx)

        stats['removed_count'] = len(to_remove_indices)

        # 创建去重后的DataFrame
        deduplicated_df = df[~df.index.isin(to_remove_indices)].copy()

        # 生成详细报告
        self._generate_detailed_report(df, deduplicated_df, selection_log, stats)

        return deduplicated_df, selection_log

    def _generate_detailed_report(self, original_df, deduplicated_df, selection_log, stats):
        """生成详细的去重报告"""
        print("\n" + "=" * 60)
        print("新闻去重详细报告")
        print("=" * 60)

        # 基础统计
        print(f"\n📊 基础统计:")
        print(f"  原始数据量: {len(original_df):,} 条")
        print(f"  去重后数据量: {len(deduplicated_df):,} 条")
        print(f"  删除重复新闻: {stats['removed_count']:,} 条")
        print(f"  重复新闻组数: {stats['total_groups']:,} 组")
        print(f"  去重率: {stats['removed_count'] / len(original_df) * 100:.2f}%")

        # 权威度分布
        print(f"\n🏅 权威度分布 (被保留的新闻组):")
        authority_names = {
            1: "最高权威", 2: "高权威", 3: "中等权威",
            4: "基础权威", 5: "一般权威", 6: "较低权威", 7: "最低权威"
        }

        for auth_level in sorted(stats['authority_distribution'].keys()):
            count = stats['authority_distribution'][auth_level]
            percentage = count / stats['total_groups'] * 100 if stats['total_groups'] > 0 else 0
            print(f"  {authority_names[auth_level]}({auth_level}): {count} 组 ({percentage:.1f}%)")

        # 选择原因统计
        print(f"\n🎯 选择原因统计:")
        reason_stats = defaultdict(int)
        for log in selection_log:
            reason_stats[log['selection_reason']] += 1

        for reason, count in sorted(reason_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(selection_log) * 100 if selection_log else 0
            print(f"  {reason}: {count} 组 ({percentage:.1f}%)")

        # 显示前几个处理示例
        if selection_log:
            print(f"\n📋 前10组处理示例:")
            for i, log in enumerate(selection_log[:10]):
                print(f"  组{log['group_id'] + 1}: 保留 '{log['best_media']}' "
                      f"(权威度{log['authority_level']}), 原因: {log['selection_reason']}, "
                      f"删除 {log['removed_count']} 个重复")


# 批量处理函数
def process_large_dataset(df, batch_size=10000):
    """
    处理大型数据集的函数（分批处理）
    """
    print(f"开始分批处理数据集，每批 {batch_size} 条...")

    deduplicator = AdvancedNewsDeduplicator(simhash_threshold=3)
    all_deduplicated = []
    all_selection_logs = []

    # 按时间分批（假设数据按日期排序）
    if '年月' in df.columns:
        unique_months = df['年月'].unique()
        print(f"按月份分批处理: {len(unique_months)} 个月")

        for month in sorted(unique_months):
            monthly_data = df[df['年月'] == month].copy()
            print(f"处理 {month}: {len(monthly_data)} 条数据")

            if len(monthly_data) > 0:
                deduplicated_month, logs = deduplicator.deduplicate_news(monthly_data)
                all_deduplicated.append(deduplicated_month)
                all_selection_logs.extend(logs)
    else:
        # 简单分批
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_data = df.iloc[start_idx:end_idx].copy()

            print(f"处理批次 {i + 1}/{total_batches}: {len(batch_data)} 条数据")
            deduplicated_batch, logs = deduplicator.deduplicate_news(batch_data)
            all_deduplicated.append(deduplicated_batch)
            all_selection_logs.extend(logs)

    # 合并结果
    final_deduplicated = pd.concat(all_deduplicated, ignore_index=True)

    print(f"\n批量处理完成!")
    print(f"最终数据量: {len(final_deduplicated):,} 条")
    print(f"总删除数量: {len(df) - len(final_deduplicated):,} 条")

    return final_deduplicated, all_selection_logs


# 使用示例
if __name__ == "__main__":
    # 示例用法
    print("高级新闻去重器初始化完成!")
    print(f"媒体权威度分级: 共{len(media_authority_map)}个媒体")

    # 统计各权威度媒体数量
    authority_counts = defaultdict(int)
    for media, score in media_authority_map.items():
        authority_counts[score] += 1

    print("\n媒体权威度分布:")
    for score in sorted(authority_counts.keys()):
        print(f"  权威度{score}: {authority_counts[score]}个媒体")


df = pd.read_excel("cleaned_normalized_data_full.xlsx")
deduplicator = AdvancedNewsDeduplicator(simhash_threshold=3)
result_df, selection_logs = deduplicator.deduplicate_news(df)

result_df.to_excel("deduplicated_news_full.xlsx", index=False)
print("已成功保存为 deduplicated_news_full.xlsx")
