import pandas as pd
import numpy as np
import gc
import cudf, itertools
import os
import argparse
from tqdm import tqdm
from utils import Logger

parser = argparse.ArgumentParser(description='生成候选集')
parser.add_argument('--logfile', default='candidate.log')
parser.add_argument('--eval', default='True')

raw_opt_path = '../../data/train_test/' # 训练测试集路径
preprocess_path = '../../data/train_valid/' # 训练验证集路径
datamart_path = '../../data/feature/'   # datamart路径
output_path = '../../data/candidate/'   # 召回结果输出路径

args = parser.parse_args()
logfile = args.logfile
is_calc_recall = args.eval.lower() == 'true'  #是否计算召回率

# 初始化日志
os.makedirs('../../log', exist_ok=True)
log = Logger(f'../../log/{logfile}').logger

def create_rule_dicts(prefix):
    """
    构建用于不同推荐任务的筛选规则字典集合，包括订单推荐（order_dict）、购物车推荐（cart_dict）和点击预测（click_dict）。

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，通常为特征文件所在目录的路径。

    返回:
        字典格式为 {候选商品数据文件路径: N(排名截断)}，value的值代表取这个共现矩阵推荐的前N个候选物品作为candidate
        - order_dict (dict): 订单预测任务所用的筛选规则字典。

        - cart_dict (dict): 购物车预测任务所用的筛选规则字典，直接复制自 order_dict，使用与订单推荐相同的特征集合。

        - click_dict (dict): 点击预测任务所用的筛选规则字典，包含点击行为相关的特征文件和参数。
                                与 order_dict 不同，其特征文件和参数专为点击预测任务设计。
    """
    order_dict = {
        prefix + 'click_click_allterm_last.parquet': 100,
        prefix + 'click_click_allterm_top.parquet': 20,
        prefix + 'click_click_allterm_hour.parquet': 100,
        prefix + 'click_click_allterm_day.parquet': 30,

        prefix + 'click_buy_allterm_last.parquet': 40,
        prefix + 'click_buy_allterm_top.parquet': 40,
        prefix + 'click_buy_allterm_hour.parquet': 40,
        prefix + 'click_buy_allterm_day.parquet': 10,

        prefix + 'buy_click_allterm_all.parquet': 40,
        prefix + 'buy_buy_allterm_all.parquet': 40,

        prefix + 'click_click_dup_last.parquet': 20,
        prefix + 'click_click_dup_top.parquet': 10,
        prefix + 'click_click_dup_hour.parquet': 20,

        prefix + 'click_buy_dup_last.parquet': 20,
        prefix + 'click_buy_dup_top.parquet': 10,
        prefix + 'click_buy_dup_hour.parquet': 20,
        prefix + 'buy_click_dup_all.parquet': 20,
        prefix + 'buy_buy_dup_all.parquet': 20,

        prefix + 'click_click_dup_wlen_last.parquet': 20,
        prefix + 'click_click_dup_wlen_hour.parquet': 20,
        prefix + 'click_buy_dup_wlen_last.parquet': 20,
        prefix + 'click_buy_dup_wlen_hour.parquet': 20,

        prefix + 'click_click_base_last.parquet': 50,
        prefix + 'click_click_base_top.parquet': 10,
        prefix + 'click_click_base_hour.parquet': 50,
        prefix + 'buy_click_base_all.parquet': 40,
        prefix + 'buy_buy_base_all.parquet': 40,

        prefix + 'click_click_base_wlen_last.parquet': 40,
        prefix + 'click_click_base_wlen_top.parquet': 10,
        prefix + 'click_click_base_wlen_hour.parquet': 30,
        prefix + 'buy_click_base_wlen_all.parquet': 20,
        prefix + 'buy_buy_base_wlen_all.parquet': 20,

        prefix + 'click_click_base_hour_last.parquet': 15,
        prefix + 'click_click_base_hour_hour.parquet': 15,
        prefix + 'click_click_dup_hour_last.parquet': 5,
        prefix + 'click_click_dup_hour_hour.parquet': 5
    }

    cart_dict = order_dict.copy()

    click_dict = {
        prefix + 'click_click_allterm_last.parquet': 30,
        prefix + 'click_click_allterm_top.parquet': 5,
        prefix + 'click_click_allterm_hour.parquet': 30,
        prefix + 'click_click_allterm_day.parquet': 10,
        
        prefix + 'click_click_dup_last.parquet': 30,
        prefix + 'click_click_dup_top.parquet': 5,
        prefix + 'click_click_dup_hour.parquet': 30,
        prefix + 'click_click_dup_day.parquet': 10,

        prefix + 'click_click_dup_wlen_last.parquet': 70,
        prefix + 'click_click_dup_wlen_hour.parquet': 50,
        prefix + 'click_click_dup_wlen_day.parquet': 10,

        prefix + 'click_click_base_hour_last.parquet': 90,
        prefix + 'click_click_base_hour_top.parquet': 5,
        prefix + 'click_click_base_hour_hour.parquet': 60,
        prefix + 'click_click_base_hour_day.parquet': 20,

        prefix + 'click_click_dup_hour_last.parquet': 30,
        prefix + 'click_click_dup_hour_hour.parquet': 30,

        prefix + 'click_click_base_last.parquet': 30,
        prefix + 'click_click_base_top.parquet': 5,
        prefix + 'click_click_base_hour.parquet': 30,
        prefix + 'click_click_base_day.parquet': 10,

        prefix + 'click_click_w2v_last_w2v.parquet': 10,
        prefix + 'click_click_w2v_hour_w2v.parquet': 5
    }

    return order_dict, cart_dict, click_dict


def load_data(prefix):
    """
    数据读取

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。

    返回:
        - data (df.dataframe): 如果prefix=='train_'返回验证集数据，如果prefix=='test_'返回测试集数据。
    """
    if prefix == 'test_':
        data = pd.read_parquet(raw_opt_path + 'test.parquet')
    else:
        data = pd.read_parquet(preprocess_path + 'test.parquet')
    return data


def gen_interaction_groundtruth_set():
    """
    ****该函数用于提取验证集的交互过的样本的groundtruth（并没有区分交互类别，只关注是否交互过，只要在之后交互过的样本就算正样本）。****
    
    因为训练验证集是由训练测试集中的训练集划分出来的，验证集中的groundtruth位于训练测试集中的训练集样本中
    该函数提取出训练测试集中训练集作为验证集交互过的groundtruth的那一部分样本，可以用做验证。
    返回的 all_data 包含每个 session 后续点击的 aid，作为交互预测的正样本 (label=1)。

    返回:
        all_data (cudf.DataFrame): 包含 [session, aid, label=1] 的带标签的交互数据。即每个session后续交互的groundtruth数据。
    """
    # 读取训练集和验证集，按 session 和时间戳排序
    train = cudf.read_parquet(preprocess_path + 'train.parquet')
    test = cudf.read_parquet(preprocess_path + 'test.parquet')
    train = train.sort_values(['session', 'ts'])
    test = test.sort_values(['session', 'ts'])

    # 将 训练集和验证集 合并成一个完整行为数据（用于计算每个 session 最后一个已知行为的位置）
    # 合并后的merge是完整训练集的一个子集，完整训练集中包含了后续每个session的一些交互行为
    # columns = [session, aid, ts, type]
    merge = cudf.concat([train, test])
    merge = merge.sort_values(['session', 'ts'])
    
    del train, test  
    gc.collect()

    # 读取完整的训练集（包含验证集中“未来可能发生”的行为）
    # columns = [session, aid, ts, type]
    all_data = cudf.read_parquet(raw_opt_path + 'train.parquet')
    all_data = all_data.sort_values(['session', 'ts'])
    all_data = all_data[['session', 'aid']]
    all_data['session'] = all_data['session'].astype('int32')
    all_data['aid'] = all_data['aid'].astype('int32')

    # 为了确定哪些交互是“未来行为”，先对每个 session 的交互打上顺序 rank
    # columns = [session, aid, ts, type, rank]
    all_data['rank'] = 1
    merge['rank'] = 1

    # 按 session 累加 rank（计算点击行为的顺序）
    # 时间越早，rank越小
    all_data['rank'] = all_data.groupby('session')['rank'].cumsum()
    merge['rank'] = merge.groupby('session')['rank'].cumsum()

    # 获取每个 session 在训练/验证集中最后一条交互的 rank
    # columns = ['session', 'rank_max']
    merge = merge.groupby('session')['rank'].max().reset_index()
    merge.columns = ['session', 'rank_max']  # 重命名为 rank_max 以区分

    # 将 rank_max 合并到 all_data(完整训练集)，用于筛选未来行为
    # columns = [session, aid, ts, type, rank, rank_max]
    all_data = all_data.merge(merge, on='session', how='left')

    # 保留那些在 merge 之后才发生的行为，作为点击预测目标（这些是“未观察行为”）
    # columns = [session, aid, ts, type, rank, rank_max]
    all_data = all_data[all_data['rank'] > all_data['rank_max']]

    del merge
    gc.collect()

    # 去除重复行为，保留唯一的 session-aid 对
    # columns = [session, aid]
    all_data = all_data[['session', 'aid']].drop_duplicates()

    # 标记这些是正样本，即真实发生过的点击目标
    # columns = [session, aid, label]
    all_data['label'] = 1
    all_data['label'] = all_data['label'].fillna(0).astype(np.int16)
    # 返回每个session后续交互的groundtruth数据
    return all_data


def agg_filtered_candidate(rule_dict):
    """
    该函数根据给定筛选规则字典，读取多个候选商品数据文件，对每个候选数据按 rank 截断保留 top-N 商品，
    最后合并这些筛选过后的 session-aid 组合，形成总候选集。

    参数:
        path_name_dict (dict): 键是候选文件名，值是截断值N。

    返回:
        candidate_all (cudf.DataFrame): 合并后的总候选集, [session, aid]
    """
    # 初始化候选数据框(综合了所有候选商品的筛选结果)
    candidate_all = None
    # 遍历字典中的每个候选商品数据文件
    for i, file_path in enumerate(tqdm(rule_dict.keys())):
        # log.info(f'[{i+1}/{len(rule_dict)}] 正在处理: {file_path}')
        # 读取候选商品数据文件
        candidate_raw = cudf.DataFrame(pd.read_parquet(datamart_path + file_path))
        # 仅保留排名小于等于指定阈值的记录，并重置索引
        candidate_raw = candidate_raw[candidate_raw['rank'] <= rule_dict[file_path]].reset_index(drop=True)

        # 合并多个来源的候选数据
        if i == 0:
            # columns = ['session', 'aid']
            candidate_all = candidate_raw[['session', 'aid']]  # 第一组直接作为初始候选
        else:
            # 相当于去除candidata_raw中与candidata_all中重复的行，再拼接起来
            # columns = ['session', 'aid']
            candidate_all = candidate_all.merge(candidate_raw[['session', 'aid']], on = ['session', 'aid'], how = 'outer')

        del candidate_raw
        gc.collect()
    
    return candidate_all


def gen_all_candidate(prefix, rule_dict, cand_type, hist_interaction_cnt):
    """
    生成最终用于训练或预测的候选集数据，并根据需要打上标签。
    - 读取根据 rule_dict 生成的候选集合，并与历史交互表做合并，以确保历史热门商品也能被纳入召回候选；
    - 若为训练集（prefix == 'train_'），则根据 cand_type 打标签（正样本），以供模型监督训练使用；
      - 若 cand_type != 'click_all'，则读取 test_labels.parquet 并展开 label；
      - 若 cand_type == 'click_all'，则使用函数生成后续交互的 aid 作为正样本；
    - 若为测试集，则只保留候选集合（不打标签）；

    参数：
        prefix (str) : 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        rule_dict (dict) : 候选生成规则的字典，不同规则用于不同召回方法的筛选。
        cand_type (str) : 候选集的类型，支持 'clicks', 'carts', 'orders' 或 'click_all' 等。
        hist_interaction_cnt (pd.DataFrame) : 历史交互统计数据，包含每个 session 与 aid（商品）之间的交互次数。[session, aid, count]。

    返回：
        - candidate_all (pd.DataFrame) : 候选集数据，每个session后续进行cand_type的预测召回结果。
                                        训练验证集(prefix = train_): columns = [session, aid, labels]
                                        训练测试集(prefix = test_): columns = [session, aid]
        - 最终保存处理后的候选集合为 parquet 文件，供后续模型训练或预测使用。
    """
    # 根据所有候选数据文件依据rule_dict进行筛选生成最终的总的召回结果
    # columns = ['session', 'aid']
    log.info(f'正在按规则筛选{cand_type}候选集')
    candidate_all = agg_filtered_candidate(rule_dict)
    candidate_all['session'] = candidate_all['session'].astype(np.int32)
    candidate_all['aid'] = candidate_all['aid'].astype(np.int32)

    # 计算所有 session 召回的 aid 数量的平均值
    avg_aid_count = candidate_all.groupby('session')['aid'].count().mean()
    log.info(f'平均每个session召回了{avg_aid_count}个的{cand_type}类物品.')


    # 将候选数据与历史交互表进行外连接合并，保留所有 session 和 aid 组合
    # 目的是为了将每个 session 中历史交互过的热门物品也纳入候选集。
    # columns = ['session', 'aid']
    candidate_all = candidate_all.merge(hist_interaction_cnt[['session', 'aid']], on=['session', 'aid'], how='outer')
    
    del hist_interaction_cnt
    gc.collect()

    # 如果是训练验证数据，则需要将验证集标签（label）合并进候选集中，供模型训练用
    if prefix == 'train_':
        log.info(f'正在构建{cand_type}类样本的样本标签...')
        # 对于 click_all 情况单独处理，其余情况（click/cart/order）处理方式如下
        if cand_type != 'click_all':
            
            # 读取包含真实标签的文件
            # columns = [session, type, groundtruth]
            labels = pd.read_parquet(preprocess_path + 'test_labels.parquet')
            # 根据当前行为类型（click/cart/order）进行过滤
            labels = labels[labels['type'] == cand_type]

            # 初始化两个列表，用于展开 ground truth 中的 aid 列表
            session_list = []
            aid_list = []
            for session_id, action_type, aids in tqdm(labels.values):
                for aid in aids:
                    session_list.append(session_id)
                    aid_list.append(aid)

            # 将 session 和 aid 列表转换成 DataFrame 形式
            # 新增一列 label，用于标记该组合是正样本
            # columns = [session, aid, label]
            labels_df = pd.DataFrame(session_list, columns = ['session'])
            labels_df['aid'] = aid_list
            labels_df['label'] = 1
            labels_df['label'] = labels_df['label'].astype(np.int16)
            labels_df = cudf.DataFrame(labels_df)

            # 与候选集 candidate_all 进行合并，保留所有候选对，并打上标签
            # 正样本label列值为1而负样本列label值为NAN。
            # columns = [session, aid, label]
            log.info('正在进行labels合并')
            candidate_all = candidate_all.merge(labels_df, on = ['session', 'aid'], how = 'left')
            # 排序
            candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
            # 转换为 pandas 格式
            candidate_all = candidate_all.to_pandas()
        else:
            # 如果是 click_all，从原始数据中生成正样本标签，正样本定义为后续交互过的物品（不单指点击的物品）
            # columns = [session, aid, label]
            click_all_labels = gen_interaction_groundtruth_set()
            # 给候选集中的数据打上相应的标签（正样本为session后续交互过的物品）
            # columns = [session, aid, label]
            log.info('正在进行labels合并')
            candidate_all = candidate_all.merge(click_all_labels, on = ['session', 'aid'], how = 'left')
            candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
            candidate_all = candidate_all.to_pandas()
            del click_all_labels
            gc.collect()
        candidate_all['label'] = candidate_all['label'].fillna(0).astype(np.int16)
    # 处理训练测试数据集，可用于最后的预测，作为召回的结果
    else:
        # columns = ['session', 'aid']
        candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
        candidate_all = candidate_all.to_pandas()

    log.info(f'正在保存{cand_type}类的预测召回候选集...')
    # 保存候选数据到指定输出路径的 parquet 文件
    candidate_all.to_parquet(output_path + prefix + f'{cand_type}_candidate.parquet')
    log.info(f'结果已保存到{output_path+prefix+cand_type}_candidate.parquet')

    return candidate_all


def calc_recall(gt_df, pred_df):
    """
    计算召回率（Recall）:
    对于每个 session，比较预测结果和 ground truth，
    统计命中个数，并汇总成全局召回率。
    
    参数:
        gt_df (pd.DataFrame): 包含真实标签的 DataFrame，至少包含列 ['session', 'ground_truth']
        pred_df (pd.DataFrame): 包含预测结果的 DataFrame，至少包含列 ['session', 'pred_labels']
                                其中 pred_labels 是预测召回的物品列表

    返回:
        recall_rate (float): 所有 session 总体的召回率
    """
    # 将 ground truth 和预测结果按 session 合并，以便每一行都能对比真实标签和预测标签
    # columns = [session, ground_truth, pred_labels]
    test_labels = gt_df.merge(pred_df, how='inner', on=['session'])

    # 对每个 session 的预测和真实标签进行“命中数(hits列)”计算
    # 命中数(hits列) = 两个集合的交集大小，即预测对了多少个 aid
    # apply函数指定axis=1是对每一行进行apply函数处理，axis=0是对每一列
    # columns = [session, ground_truth, pred_labels, hits]
    test_labels['hits'] = test_labels.apply(
        lambda df: len(set(df['ground_truth']) & set(df['pred_labels'])), axis=1
    )

    # 统计每个 session 中真实标签的数量（最多限制为20个），用于做分母
    test_labels['gt_count'] = test_labels['ground_truth'].apply(len).clip(0, 20)

    # 总命中 / 总 ground truth 数量，就是 recall
    recall_rate = test_labels['hits'].sum() / test_labels['gt_count'].sum()
    return recall_rate


if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)
    # 对训练测试集和训练验证集都进行处理
    for prefix in ['train_', 'test_']:
        if prefix == 'train_':
            log.info('正在处理训练验证集...')
        else:
            log.info('正在处理训练测试集...')

        # 创建筛选规则字典
        order_dict, cart_dict, click_dict = create_rule_dicts(prefix)
        
        # 构建对应关系
        recall_rule_dict = {
            'orders': order_dict,
            'carts': cart_dict,
            'clicks': click_dict,
            'click_all': click_dict
        }

        log.info('正在加载test数据...')
        # 读取验证集或测试集数据
        # columns = [session, aid, ts, type]
        df_test = load_data(prefix)


        log.info('正在构建验证/测试集的历史交互表...')
        # 统计每个 session 中 aid 出现的次数，形成历史交互表
        # columns = [session, aid, count]
        hist_interaction_cnt = cudf.DataFrame(df_test[['session', 'aid']].value_counts().reset_index())

        # orders/carts/clicks：
        #   进行召回的结果分别是作为购买行为预测召回、加购行为预测召回、点击行为预测召回，
        #   groundtruth labels是后续时间中购买的物品、加购的物品和点击的物品
        # click_all:
        #   进行召回的结果是作为点击行为预测召回，但是其groundtruth labels的正样本是后续时间中交互过的物品（不单单是点击的）
        for cand_type in ['orders', 'carts', 'clicks', 'click_all']:
            log.info(f'正在生成{cand_type}的召回结果...')
            # 获取召回结果并保存为.parquet文件
            # 训练验证集(prefix = train_): columns = [session, aid, labels]
            # 训练测试集(prefix = test_): columns = [session, aid]
            candidate_all = gen_all_candidate(prefix, recall_rule_dict[cand_type], cand_type, hist_interaction_cnt)
            # 如果是训练验证集（这种情况才有label）且 calc_recall 为 True
            # 计算召回率
            if prefix == 'train_' and is_calc_recall == True:
                log.info(f'=== 计算{cand_type}类型召回结果的召回率 ===')
                # 把 candidate_all 中每个 session 的 aid 合并为一个整数列表
                # columns=[session, pred_labels]
                pred_temp = candidate_all[['session', 'aid']].groupby('session')['aid'].apply(list).reset_index()
                pred_temp.columns = ['session', 'pred_labels']
                del candidate_all
                gc.collect()

                log.info('正在生成ground_truth集合...')
                if cand_type == 'click_all':
                    # click_all需要用原始训练集生成训练验证集的交互ground_truth
                    # columns = [session, aid, label=1]
                    labels = gen_interaction_groundtruth_set()
                    # columns = [session, aid(list)]
                    labels = labels[['session', 'aid']].groupby('session').agg({'aid': list}).reset_index()
                    # columns = [session, ground_truth(list)]
                    labels.columns = ['session', 'ground_truth']
                    labels = labels.to_pandas()

                else:
                    # 读取包含真实标签的文件
                    # columns = [session, type, groundtruth]
                    labels = pd.read_parquet(preprocess_path + 'test_labels.parquet')
                    # 根据当前行为类型（click/cart/order）进行过滤
                    labels = labels[labels['type'] == cand_type]

                log.info(f'正在计算{cand_type}类型召回结果的召回率...')
                recall_rate = calc_recall(labels, pred_temp)
                log.info(f'{cand_type}类型的召回率为{recall_rate}.')


        del hist_interaction_cnt
        gc.collect()





