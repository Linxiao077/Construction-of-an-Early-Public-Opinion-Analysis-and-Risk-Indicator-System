#第一个数据集处理文件，完成了汽车品牌的提取、新闻发布媒体的分类和提取、删除了没数据的5行。最后完成了文本清洗，分词，提取词向量。
#在defect_categories数据框中本来想实现分类，但是代码有问题。后面的置信度分析也暂时没用。
#日期提取目前没成功


from typing import Union, Any

import pandas as pd
import jieba
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

# 加载原始数据
df = pd.read_excel("2022-01.xlsx")

# ==================== 数据规范化处理 ====================

# 1. 定义标准化映射和列表
CAR_BRANDS = [
    '大众', '丰田', '本田', '日产', '别克', '奔驰', '宝马', '奥迪', '现代', '起亚',
    '福特', '雪佛兰', '吉利', '比亚迪', '哈弗', '长安', '奇瑞', '传祺', '荣威', '名爵',
    '沃尔沃', '雷克萨斯', '凯迪拉克', '保时捷', '路虎', '捷豹', '林肯', '英菲尼迪', '讴歌', '特斯拉',
    '马自达', '三菱', '斯巴鲁', '铃木', '标致', '雪铁龙', '雷诺', '菲亚特', 'Jeep', '玛莎拉蒂',
    '法拉利', '兰博基尼', '宾利', '劳斯莱斯', '迈凯伦', '阿斯顿马丁', '蔚来', '小鹏', '理想', '威马',
    '领克', 'WEY', '星途', '捷途', '东风', '江淮', '北汽', '一汽', '上汽', '广汽',
    '宝骏', '五菱', '欧拉', '几何', '极氪', '岚图', '仰望', '腾势', '方程豹', '深蓝',
    '阿维塔', '极狐', '智己', '飞凡', '昊铂', '远航', '合创', '创维', '天际', '爱驰',
    '威麟', '开瑞', '观致', '宝沃', '汉腾', '君马', '比速', '幻速', '潍柴', '黄海',
    '中兴', '猎豹', '陆风', '江铃', '华颂', '之诺', '华骐', '理念', '思铭', '启辰',
    '捷达', '思皓', '捷恩斯', '罗伦士', '巴博斯', '卡尔森', '泰卡特', '路特斯', '阿尔法·罗密欧', 'DS',
    '迈巴赫', 'smart', 'MINI', '欧宝', '霍顿', '西雅特', '达契亚', '拉达', '莫斯科人', '大宇',
    '双龙', '捷尼赛思', '科尼赛克', '帕加尼', '布加迪', '世爵', '萨林', '摩根', '卡特汉姆', '诺贝尔',
    '光冈', '汤姆', '永源', '吉奥', '美亚', '九龙', '少林', '女神', '飞碟', '新凯',
    '安驰', '富奇', '黑豹', '鲁滨逊', '燕京', '长城', '解放', '红旗', '奔腾', '夏利',
    '北京', '勇士', '战旗', '枭龙', '猛士', '东风风光', '东风风行', '东风风神', '东风小康', '东风风度',
    '江淮瑞风', '江淮帅铃', '江淮康铃', '江淮骏铃', '北汽制造', '北汽幻速', '北汽威旺', '北汽昌河', '北汽福田', '北汽新能源',
    '一汽吉林', '一汽佳宝', '一汽红塔', '一汽通用', '上汽大通', '上汽科莱威', '上汽跃进', '广汽传祺', '广汽本田', '广汽丰田',
    '广汽三菱', '广汽菲克', '广汽日野', '广汽比亚迪', '吉利',
    '比亚迪','长安', '奇瑞', '哈弗','蔚来', '小鹏', '理想ONE','理想', '威马', '哪吒', '零跑',
    '高合HiPhi', '赛力斯SF5', '问界M5', '问界M7', '极氪001', '极氪009', '极氪X', '岚图',
    '智己', '飞凡', '阿维塔', '极狐阿尔法', '极狐阿尔法', '创维', '天际',
    '爱驰', '威马', '云度', '前途', '国机智骏', '华晨鑫源', '御捷新能源', '雷丁汽车', '康迪']
MEDIA_TYPES = {
    # 财经媒体
    '东方财富 APP': '财经媒体',
    '同花顺财经': '财经媒体',
    '财经头条 APP': '财经媒体',
    '新浪财经 APP': '财经媒体',
    '雪球': '财经媒体',
    '东方财富网': '财经媒体',
    '金投网': '财经媒体',
    '和讯网': '财经媒体',
    '金融界': '财经媒体',
    '证券之星': '财经媒体',
    '财联社': '财经媒体',
    '汇通财经': '财经媒体',
    '格隆汇': '财经媒体',
    '华尔街见闻': '财经媒体',
    '富途牛牛': '财经媒体',
    '老虎证券 APP': '财经媒体',
    '大智慧 APP': '财经媒体',
    '股吧 APP': '财经媒体',

    # 综合新闻
    '腾讯新闻 APP': '综合新闻',
    '网易新闻 APP': '综合新闻',
    '新浪新闻 APP': '综合新闻',
    '搜狐新闻 APP': '综合新闻',
    '今日头条': '综合新闻',
    '百度 APP': '综合新闻',
    'UC头条 APP': '综合新闻',
    '一点资讯': '综合新闻',
    '凤凰新闻 APP': '综合新闻',
    '人民日报 APP': '综合新闻',
    '央视财经 APP': '综合新闻',
    '新华财经 APP': '综合新闻',
    '澎湃新闻 APP': '综合新闻',
    '界面新闻 APP': '综合新闻',
    'ZAKER APP': '综合新闻',
    '趣头条 APP': '综合新闻',

    # 垂直汽车媒体
    '懂车帝 APP': '垂直汽车',
    '汽车之家': '垂直汽车',
    '车质网': '垂直汽车',
    '易车网': '垂直汽车',
    '太平洋汽车网': '垂直汽车',
    '爱卡汽车网': '垂直汽车',
    '新浪汽车': '垂直汽车',
    '搜狐汽车': '垂直汽车',
    '网上车市 APP': '垂直汽车',
    '汽车头条 APP': '垂直汽车',
    '车讯网': '垂直汽车',
    '中国汽车召回网': '垂直汽车',
    '汽车投诉网': '垂直汽车',
    '电车之家': '垂直汽车',
    '第1电动': '垂直汽车',

    # 社交平台
    '新浪微博': '社交平台',
    '知乎': '社交平台',
    '知乎专栏': '社交平台',
    '抖音 APP': '社交平台',
    '快手 APP': '社交平台',
    '哔哩哔哩APP': '社交平台',
    '小红书 APP': '社交平台',
    '虎扑社区': '社交平台',

    # 专业资讯平台
    '虎嗅网 APP': '专业资讯',
    '36氪': '专业资讯',
    '钛媒体网': '专业资讯',
    '创业邦 APP': '专业资讯',
    '品玩 APP': '专业资讯',
    '亿欧网': '专业资讯',
    '商业新知': '专业资讯',
    '智通财经网': '专业资讯',

    # 官方机构
    '国家市场监督管理总局': '官方机构',
    '国务院发展研究中心信息网': '官方机构',
    '中国质量新闻网': '官方机构',
    '央视网': '官方机构',
    '新华网': '官方机构',
    '人民网': '官方机构',
    '中国网': '官方机构',
    '中国经济网': '官方机构',

    # 地方媒体
    '周到上海 APP': '地方媒体',
    '齐鲁壹点 APP': '地方媒体',
    '扬子扬眼 APP': '地方媒体',
    '羊城派 APP': '地方媒体',
    '大河新闻 APP': '地方媒体',
    '开屏新闻 APP': '地方媒体',
    '南方财经网': '地方媒体',
    '北京日报 APP': '地方媒体',
    '湖北日报 APP': '地方媒体',
    '海南日报 APP': '地方媒体',

    # 其他专业平台
    '我的钢铁网': '行业垂直',
    '长江有色金属网': '行业垂直',
    '中国钢铁工业信息网': '行业垂直',
    '盖世汽车网': '行业垂直',
    '中国汽车报网': '行业垂直',

    # 传统媒体数字版
    '上海证券报 APP': '传统媒体转型',
    '证券时报 APP': '传统媒体转型',
    '国际金融报网': '传统媒体转型',
    '每日经济新闻 APP': '传统媒体转型',
    '经济观察网': '传统媒体转型',
    '中国青年报 APP': '传统媒体转型',
    '中国新闻社': '传统媒体转型',
    '参考消息网': '传统媒体转型',

    # 券商平台
    '中信建投证券 APP': '券商平台',
    '国泰君安证券 APP': '券商平台',
    '华福证券 APP': '券商平台',
    '华鑫证券 APP': '券商平台',
    '海通e海通财 APP': '券商平台',
    '首创证券 APP': '券商平台',

    # 其他
    '微信公众号': '内容聚合',
    '看点快报 APP': '内容聚合',
    '快资讯': '内容聚合',
    '云掌财经': '内容聚合',
    '中金在线': '内容聚合',
    '金融八卦女 APP': '内容聚合',
    '虎嗅': '专业资讯',
    '虎嗅网 APP': '专业资讯',
    '一点资讯 APP': '综合新闻',
    '手机新浪网': '综合新闻',
    'hao123新闻': '内容聚合',
    '深圳热线': '地方媒体',
    '环球网': '综合新闻',
    '中新经纬 APP': '财经媒体',
    '每经网': '财经媒体',
    '国际金融报网': '财经媒体',
    '证券时报网': '财经媒体',
    '中国财富 APP': '财经媒体',
    '同花顺iFinD APP': '财经媒体',
    '东方财富 APP': '财经媒体',
    '掌证宝天玑版 APP': '财经媒体',
    '腾讯自选股 APP': '财经媒体',
    'BBAE必贝证券 APP': '财经媒体',
    '智远壹户通 APP': '财经媒体',
    '天天基金网 APP': '财经媒体',
    '见闻VIP APP': '财经媒体',
    '英为财情': '财经媒体',
    '汇金网': '财经媒体',
    '财经钻': '财经媒体',
    '思维财经': '财经媒体',
    '顶尖财经网': '财经媒体',
    '全景网': '财经媒体',
    '中财网': '财经媒体',
    '百家财富网': '财经媒体',
    '中国财讯网': '财经媒体',
    '潮起网': '财经媒体',
    '财经产业网': '财经媒体',
    '文财网': '财经媒体',
    '优财网': '财经媒体',
    '发现网': '财经媒体',
    '成功财经网': '财经媒体',
    '博览黄页': '财经媒体',
    '最资讯': '财经媒体',
    '中投网': '财经媒体',
    '商业新知': '专业资讯',
    '虎嗅网 APP': '专业资讯',
    '36氪': '专业资讯',
    '钛媒体网': '专业资讯',
    '创业邦 APP': '专业资讯',
    '品玩 APP': '专业资讯',
    '亿欧网': '专业资讯',
    '智通财经网': '专业资讯',
    '智通财经 APP': '专业资讯',
    'DoNews': '专业资讯',
    '未央网': '专业资讯',
    '支付百科网': '专业资讯',
    '企业时报网': '专业资讯',
    '今报网': '专业资讯',
    '博思网': '专业资讯',
    '电子工程世界网': '专业资讯',
    'TSC技术性贸易措施经济网': '专业资讯',
    '视点陜西网': '专业资讯',
    '中国财经时报网': '专业资讯',
    '315汽车网': '专业资讯',
    '汽车新看点': '专业资讯',
    '中国汽车头条': '专业资讯',
    '环球商讯网': '专业资讯',
    '凯图新闻网': '专业资讯',
    '汽车网评': '专业资讯',
    '网通社.汽车': '专业资讯',
    '零排放汽车网': '专业资讯',
    '水滴汽车': '专业资讯',
    '汽场': '专业资讯',
    '中国产业经济信息网': '专业资讯',
    '中国质量报(数字报)': '官方机构',
    '海东日报（数字报）': '官方机构',
    '今晚报': '官方机构',
    '福建省市场监督管理局（知识产权局）': '官方机构',
    '海南省市场监督管理局': '官方机构',
    '通化市人民政府': '官方机构',
    '芜湖市人民政府': '官方机构',
    '沈阳网': '官方机构',
    '新疆汽车网': '官方机构',
    '北部湾在线-新媒体': '官方机构',
    '西北网讯': '官方机构',
    '中青在线': '官方机构',
    '光明网': '官方机构',
    '中工网': '官方机构',
    '中国青年网': '官方机构',
    '中原汽车网': '官方机构',
    '八桂网': '官方机构',
    '红网': '官方机构',
    '山西晚报': '官方机构',
    '荆楚网-湖北日报网': '官方机构',
    '云南网': '官方机构',
    '合肥在线': '官方机构',
    '安徽网': '官方机构',
    '大众网': '官方机构',
    '河南一百度': '官方机构',
    '河北新闻网': '官方机构',
    '丹阳翼网': '官方机构',
    '尊流汽车网': '官方机构',
    '二三里': '官方机构',
    '我车网': '官方机构',
    '中原汽车网': '官方机构',
    '海角汽车': '官方机构',
    '新浪股市汇': '社交平台',
    '同花顺圈子论坛': '社交平台',
    '雪球': '社交平台',
    '东方财富网股吧': '社交平台',
    '太平洋汽车网论坛': '社交平台',
    '汽车之家论坛': '社交平台',
    '思否': '社交平台',
    '虎扑社区': '社交平台',
    '花花女性网': '其他',
    '金融八卦女 APP': '其他',
    'Bianews': '其他',
    'xw.qq.com': '其他',
    '手机搜狐 APP': '其他',
    '手机网易网': '其他',
    '宜春新闻网': '其他',
    '钢企网': '其他',
    '云聚网': '其他',
    '华福证券 APP': '券商平台',
    '华鑫证券 APP': '券商平台',
    '海通e海通财 APP': '券商平台',
    '首创证券 APP': '券商平台',
    '中信建投证券 APP': '券商平台',
    '国泰君安证券 APP': '券商平台',
    '中金在线 APP': '券商平台',
    '掌证宝天玑版 APP': '券商平台',
    '智远壹户通 APP': '券商平台',
    '天天基金网 APP': '券商平台',
    '见闻VIP APP': '券商平台',
    'BBAE必贝证券 APP': '券商平台',
    '老虎证券 APP': '券商平台',
    '富途牛牛': '券商平台',
    '大智慧 APP': '券商平台',
    '股吧 APP': '券商平台',
    '腾讯自选股 APP': '券商平台',
    '同花顺 APP': '券商平台',
    '同花顺iFinD APP': '券商平台',
    '东方财富 APP': '券商平台',
    '中搜搜悦 APP': '其他',
    '中青看点APP': '其他',
    '环球TIME APP': '其他',
    '央广网 APP': '其他',
    '封面': '其他',
    '周到上海网': '其他',
    '第一财经 APP': '其他',
    '财经网': '其他',
    '科创板日报': '其他',
    '乐居财经 APP': '其他',
    '每日经济新闻 APP': '其他',
    '每经网': '其他',
    '国际金融报网': '其他',
    '上海证券报.中国证券网': '其他',
    '证券时报网': '其他',
    '中国证券网': '其他',
    '中国基金报': '其他',
    '期货日报': '其他',
    '中国银行保险报': '其他',
    '中国经营报': '其他',
    '经济参考报': '其他',
    '华夏时报': '其他',
    '投资者网': '其他',
    '市值风云': '其他',
    '面包财经': '其他',
    '节点财经': '其他',
    '鲸准': '其他',
    '烯牛数据': '其他',
    '企查查': '其他',
    '天眼查': '其他',
    '启信宝': '其他',
    '爱企查': '其他',
    '企信宝': '其他',
}
CATEGORY_MAPPING = {
    '新车': '新车资讯',
    '上市': '新车资讯',
    '发布': '新车资讯',
    '评测': '汽车评测',
    '试驾': '汽车评测',
    '对比': '汽车评测',
    '召回': '质量问题',
    '投诉': '质量问题',
    '故障': '质量问题',
    '维权': '质量问题',
    '销量': '行业数据',
    '数据': '行业数据',
    '排行': '行业数据',
    '政策': '行业政策',
    '法规': '行业政策',
    '新能源': '新能源汽车',
    '电动': '新能源汽车',
    '电池': '新能源汽车',
    '充电': '新能源汽车'
}

defect_categories = {
    # 质量问题类
    '质量问题': {
        'keywords': ['故障', '损坏', '断裂', '漏油', '漏水', '异响', '抖动', '失灵', '失效', '破损',
                     '开裂', '变形', '生锈', '腐蚀', '脱落', '松动', '卡滞', '磨损', '烧毁', '短路'],
        'sub_categories': ['制造缺陷', '装配问题', '材料问题', '工艺问题']
    },

    # 性能问题类
    '性能问题': {
        'keywords': ['加速无力', '油耗高', '动力不足', '刹车距离长', '操控差', '续航短', '充电慢',
                     '过热', '过热保护', '性能衰减', '功率不足', '扭矩不足', '最高速度低'],
        'sub_categories': ['动力性能', '制动性能', '能耗性能', '操控性能']
    },

    # 安全问题类
    '安全问题': {
        'keywords': ['自燃', '起火', '爆炸', '刹车失灵', '转向失灵', '气囊不弹', '安全带失效',
                     '失控', '抱死', '失速', '碰撞安全', '视野盲区', '儿童安全'],
        'sub_categories': ['主动安全', '被动安全', '功能安全', '电气安全']
    },

    # 舒适性问题类
    '舒适性问题': {
        'keywords': ['噪音大', '振动强', '异味', '空调不制冷', '座椅不舒服', '隔音差', '颠簸',
                     '风噪', '胎噪', '异响', '空间狭小', '悬挂硬', '减震差'],
        'sub_categories': ['NVH问题', '空调系统', '座椅舒适性', '悬挂舒适性']
    },

    # 电子系统问题类
    '电子系统问题': {
        'keywords': ['屏幕黑屏', '死机', '卡顿', '系统崩溃', '导航不准', '蓝牙连接', '音响问题',
                     '传感器故障', '雷达失效', '摄像头模糊', '软件bug', '系统升级失败'],
        'sub_categories': ['信息娱乐系统', '驾驶辅助系统', '车身电子系统', '软件系统']
    },

    # 服务问题类
    '服务问题': {
        'keywords': ['售后差', '维修贵', '等待时间长', '配件缺货', '技术不行', '态度不好',
                     '保养贵', '索赔困难', '服务网点少', '响应慢', '维修质量差'],
        'sub_categories': ['售后服务', '维修质量', '配件供应', '客户服务']
    },

    # 设计缺陷类
    '设计缺陷': {
        'keywords': ['设计不合理', '人机工程差', '操作不便', '视野不好', '空间设计', '储物空间少',
                     '按键布局', '界面设计', '人体工学', '使用不便', '设计缺陷'],
        'sub_categories': ['人机工程设计', '空间设计', '操作设计', '外观设计']
    },

    # 环保问题类
    '环保问题': {
        'keywords': ['排放超标', '噪音污染', '尾气问题', '环保标准', '碳排放', '能耗高',
                     '材料环保', '回收利用', '污染环境'],
        'sub_categories': ['排放问题', '噪音污染', '材料环保性', '能耗环保']
    },

    # 无问题类
    '无问题': {
        'keywords': ['满意', '很好', '优秀', '推荐', '超值', '性价比高', '不错', '好评',
                     '物超所值', '值得购买', '完美', '无可挑剔'],
        'sub_categories': ['正面评价', '推荐评价', '满意度高']
    }
}

#分类器
class CarDefectClassifier:
    def __init__(self, defect_categories):
        self.defect_categories = defect_categories
        self.keyword_mapping = self._build_keyword_mapping()

    def _build_keyword_mapping(self):
        """构建关键词到类别的映射"""
        mapping = {}
        for category, info in self.defect_categories.items():
            for keyword in info['keywords']:
                mapping[keyword] = category
        return mapping

    def classify_text(self, tokenized_text):
        """对分词后的文本进行分类"""
        category_scores = defaultdict(int)

        for word in tokenized_text:
            if word in self.keyword_mapping:
                category = self.keyword_mapping[word]
                category_scores[category] += 1

        if not category_scores:  # 没有匹配到任何关键词
            return '无问题', 0

        # 返回得分最高的类别
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0], best_category[1]

    def classify_with_details(self, tokenized_text):
        """详细分类，包含所有匹配信息"""
        matches = defaultdict(list)
  
        for word in tokenized_text:
            if word in self.keyword_mapping:
                category = self.keyword_mapping[word]
                matches[category].append(word)

        if not matches:
            return {
                'main_category': '无问题',
                'confidence': 0,
                'matched_keywords': [],
                'all_matches': matches
            }

        # 找到主要类别
        main_category = max(matches.items(), key=lambda x: len(x[1]))
        confidence = len(main_category[1]) / len(tokenized_text) if tokenized_text else 0

        return {
            'main_category': main_category[0],
            'confidence': round(confidence, 3),
            'matched_keywords': main_category[1],
            'all_matches': dict(matches)
        }


# 对DataFrame中的文本进行分类
def add_classification_to_df(df, classifier):
    """为DataFrame添加分类结果"""
    classifications = []
    confidences = []
    matched_keywords_list = []

    for tokens in df['tokenized_text']:
        result = classifier.classify_with_details(tokens)
        classifications.append(result['main_category'])
        confidences.append(result['confidence'])
        matched_keywords_list.append(result['matched_keywords'])

    df['defect_category'] = classifications
    df['classification_confidence'] = confidences
    df['matched_keywords'] = matched_keywords_list

    return df


# 2. 统一时间格式
def normalize_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        # 尝试多种日期格式
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日', '%Y.%m.%d']:
            try:
                return datetime.strptime(str(date_str), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        # 提取日期部分（适用于含时间的字符串）
        match = re.search(r'(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})', str(date_str))
        if match:
            return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        return None
    except:
        return None

df['年月'] = df['年月'].apply(normalize_date)

# 3. 标准化媒体类型
def normalize_media_type(media_type):
    if pd.isna(media_type):
        return '其他'
    media_type = str(media_type).strip()
    for key, value in MEDIA_TYPES.items():
        if key in media_type:
            return value
    return '其他'

df['媒体类型'] = df['媒体名称'].apply(normalize_media_type)

# 4. 提取汽车品牌
def extract_car_brand(text):
    if pd.isna(text):
        return []
    text = str(text)
    return [brand for brand in CAR_BRANDS if brand in text]

df['汽车品牌'] = df['内容'].apply(extract_car_brand)
df['主要品牌'] = df['汽车品牌'].apply(lambda x: x[0] if len(x) > 0 else '其他')

# 4. 问题分类
# def determine_question_type(content):
#     """
#     根据内容判断问题类型
#     """
#     if not isinstance(content, str):
#         return None
#
#     content_lower = content.lower()
#
#     # 检查内容中是否包含关键词
#     for keyword, q_type in CATEGORY_MAPPING.items():
#         if keyword in content_lower:
#             return q_type
#
#     return '其他问题'  # 默认分类
#
# df['问题类型'] = df['内容'].apply(determine_question_type)

# 删除多余行
df = df.drop(columns=['host'], errors='ignore')
df = df.drop(columns=['发布日期'], errors='ignore')
df = df.drop(columns=['点赞'], errors='ignore')
df = df.drop(columns=['评论'], errors='ignore')
df = df.drop(columns=['转发'], errors='ignore')

# ==================== 原有文本处理流程 ====================

# 1. 文本清洗和分词
def load_stopwords(path='chinese_stopwords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

stopwords = load_stopwords()

def clean_and_tokenize(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 去除非中文字符
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords and len(w.strip()) > 1]
    return words

texts = df['内容'].astype(str).tolist()
tokenized_texts = [clean_and_tokenize(t) for t in texts]

df['tokenized_text'] = [' '.join(tokens) for tokens in tokenized_texts]  # 用空格分隔词语

# 2. 构建词汇表
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

all_words = [word for sent in tokenized_texts for word in sent]
word_freq = Counter(all_words)
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, freq in word_freq.items():
    if freq >= 2:  # 过滤低频词
        vocab[word] = len(vocab)

# 3. 文本编码
def encode(sentence, vocab):
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in sentence]

encoded_texts = [encode(sent, vocab) for sent in tokenized_texts]

df['编码序列'] = [','.join(map(str, seq)) for seq in encoded_texts]  # 用逗号分隔数字

# 4. 序列填充
tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in encoded_texts]
padded_tensors = pad_sequence(tensor_sequences, batch_first=True, padding_value=vocab[PAD_TOKEN])

# 5. 缺陷分类
classifier = CarDefectClassifier(defect_categories)
df = add_classification_to_df(df, classifier)

#return df, padded_tensors, vocab
# ==================== 保存处理结果 ====================

# 1. 保存规范化后的数据
df.to_excel('cleaned_normalized_data_full.xlsx', index=False)
#df.to_csv('cleaned_normalized_data.csv', index=False, encoding='utf-8-sig')

# 2. 保存词汇表
with open('vocab.txt', 'w', encoding='utf-8') as f:
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{word}\t{idx}\n")

# 3. 保存处理后的张量
torch.save(padded_tensors, 'processed_tensors.pt')

# ==================== 输出统计信息 ====================

print("\n=== 数据统计信息 ===")
print(f"总样本数: {len(df)}")
print(f"媒体类型分布:\n{df['媒体类型'].value_counts()}")
print(f"汽车品牌分布:\n{df['主要品牌'].value_counts()}")
print(f"词汇表大小: {len(vocab)}")
print(f"张量形状: {padded_tensors.shape}")

print("\n=== 示例数据 ===")
print("原始文本:", texts[0][:50] + "...")
print("分词结果:", tokenized_texts[0][:5], "...")
print("编码序列:", encoded_texts[0][:10], "...")
print("填充后的张量:", padded_tensors[0][:10])


# 分析分类结果
def analyze_classification_results(df):
    """分析分类结果"""
    print("=== 分类结果分析 ===")
    print(f"总文本数: {len(df)}")
    print("\n各类别分布:")
    category_stats = df['defect_category'].value_counts()
    for category, count in category_stats.items():
        percentage = count / len(df) * 100
        print(f"{category}: {count}条 ({percentage:.1f}%)")

    print("\n置信度统计:")
    print(f"平均置信度: {df['classification_confidence'].mean():.3f}")
    print(f"置信度中位数: {df['classification_confidence'].median():.3f}")

    # 显示每个类别的示例
    print("\n各类别示例:")
    for category in df['defect_category'].unique():
        examples = df[df['defect_category'] == category].head(2)
        print(f"\n{category}:")
        for idx, row in examples.iterrows():
            print(f"  - {row['内容'][:50]}... (置信度: {row['classification_confidence']})")


# 执行分析
analyze_classification_results(df)


# 检查分词结果是否包含关键词
def debug_tokenization(texts, tokenized_texts, defect_categories):
    """调试分词结果"""
    all_keywords = set()
    for category_info in defect_categories.values():
        all_keywords.update(category_info['keywords'])

    print("=== 分词调试 ===")
    print("所有关键词:", all_keywords)

    # 检查前几个样本的分词结果
    for i in range(min(5, len(texts))):
        print(f"\n样本 {i}:")
        print(f"原始文本: {texts[i][:100]}...")
        print(f"分词结果: {tokenized_texts[i]}")
        print(f"匹配到的关键词: {set(tokenized_texts[i]) & all_keywords}")


