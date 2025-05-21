import numpy as np 
import pandas as pd 
import os
import gc
import polars as pl
from collections import Counter
from tqdm import tqdm

CLICK = 0
CART = 1
ORDER = 2
ALL = 3

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
candidate_path = '../../data/candidate/'
output_path = '../../data/feature/'
sampling = True # 是否启用采样（用于调试）


def load_data(prefix, test_only=False):
    """
    数据读取

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。

    返回:
        train_actions (df.dataframe): 如果prefix=='train_'返回训练验证集中的训练数据，如果prefix=='test_'返回训练测试集中的训练数据。
        test_actions (df.dataframe): 如果prefix=='train_'返回训练验证集中的验证数据，如果prefix=='test_'返回训练测试集中的测试数据。
    """
    # 读取数据
    # [session, aid, ts, type]
    if prefix == 'test_':
        train_actions = pd.read_parquet(raw_opt_path + 'train.parquet')
        test_actions = pd.read_parquet(raw_opt_path + 'test.parquet')
    else:
        train_actions = pd.read_parquet(preprocess_path + 'train.parquet')
        test_actions = pd.read_parquet(preprocess_path + 'test.parquet')
    if test_only == True:
        return test_actions
    else:
        return train_actions, test_actions


def gen_item_transition_scores(prefix):
    """
    构建物品之间的 转移分数，(aid -> aid) ，转移分数是由 aid → aid 的转移总数归一化后得到，
    这里的转移对仅由aid和aid下一个被交互的aid_next构成，比如session1点击了aid1,aid2,aid3，那么构建成的转移对为(aid1,aid2)和(aid2,aid3)。
    参数:
    prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
    
    返回值：
    item_trans_scores (dict[tuple[int, int], float]) : 归一化后的 bigram 统计（转移分数）字典。
                                                    key 是 (aid1, aid2) 形式的物品转移对，value 是它们的归一化转移分数。
    """
    print('正在加载数据...')
    # train_actions (df.dataframe): [session, aid, ts, type], 如果prefix=='train_', 为训练验证集中的训练数据，如果prefix=='test_', 为训练测试集中的训练数据。
    # test_actions (df.dataframe): [session, aid, ts, type], 如果prefix=='train_', 为训练验证集中的验证数据，如果prefix=='test_', 为训练测试集中的测试数据。
    train_actions, test_actions = load_data(prefix)
    # 统计每个物品的出现频次（所有行为 / 每种行为）
    # [session, aid, ts, type]
    df = pd.concat([train_actions, test_actions])
    # 统计每个item被交互的总次数
    # dict{'all':dict{aid1:cnt1, aid2:cnt2, ...}}
    item_cnt = {ALL: df.groupby('aid').size().to_dict()}
    # 统计每个item被点击/加购/购买的总次数
    # item_cnt = dict{
    #               3(ALL):dict{aid1:cnt1, aid2:cnt2, ...}, 
    #               0(CLICK):dict{aid1:cnt1, aid2:cnt2, ...},
    #               1(CART):dict{aid1:cnt1, aid2:cnt2, ...},
    #               2(ORDER):dict{aid1:cnt1, aid2:cnt2, ...},
    #               }
    print('正在统计item被点击/加购/购买的总次数...')
    for t in tqdm([CLICK, CART, ORDER]):
        item_cnt[t] = df.loc[df['type'] == t].groupby('aid').size().to_dict()
    
    # 构造 aid → aid 的转移对
    # [session, aid]
    trans_pairs = pd.concat([train_actions, test_actions])[['session', 'aid']]

    del train_actions, test_actions
    gc.collect()

    # aid_next列表示session在与当前aid交互后下一个交互的物品aid，如果是最后一个交互的物品,aid_next=NAN
    # [session, aid, aid_next]
    trans_pairs['aid_next'] = trans_pairs.groupby('session')['aid'].shift(-1)
    # 将最后一个交互的物品无法形成aid-aid pair，丢弃
    trans_pairs = trans_pairs.dropna()
    trans_pairs['aid_next'] = trans_pairs['aid_next'].astype('int32')
    # 仅保留相关列
    # [aid, aid_next]
    trans_pairs = trans_pairs[['aid', 'aid_next']]

    # 统计物品转移对的出现次数
    # dict{tuple(aid, aid_next): cnt(总出现次数)}
    print('正在统计物品转移对的出现次数...')
    # item_trans_counter = trans_pairs.groupby(['aid', 'aid_next']).size().to_dict()
    item_trans_counter = Counter(zip(trans_pairs['aid'], trans_pairs['aid_next']))

    del trans_pairs
    gc.collect()

    # 对 转移次数 进行归一化：(a1,a2)转移对出现次数 除以 两者各自的总出现次数的乘积的平方根
    # dict{tuple(aid, aid_next): cnt(归一化后的结果)}
    print('正在进行归一化...')
    item_trans_scores = {}
    for (a1, a2), cnt in tqdm(item_trans_counter.items()):
        item_trans_scores[(a1, a2)] = cnt/np.sqrt(item_cnt[ALL].get(a1, 1) * item_cnt[ALL].get(a2, 1))
    
    return item_trans_scores


def gen_bigram_features(order_candidate, user_2_test_action_with_type, item_trans_scores, name):
    """
    item_trans_scores总结所有session的交互历史，得到了两两物品的转换分数，也就是交互了物品A之后再交互物品B的数量有多少（数量经过标准化得到分数）,
    那么对于给每个session用户召回的order物品 item_o，通过查看用户的历史点击列表，可以发现一些历史点击物品 item_c 在 item_trans_scores 中与item_o组成了转换对，
    那么就能进一步验证给用户下一步对item_o进行order操作的可能性。用户历史加购列表也是同理。

    该函数作用是统计对于每个用户-召回order物品对（u, item_o），计算u的历史点击列表和加购列表中物品与item_o的转换分数的统计值。即bigram feature

    
    参数说：
    order_candidate (pd.DataFrame) : order的召回结果样本的 DataFrame，包含以下列：[session, aid, label]

    user_2_test_action_with_type (dict[int, dict[int, list[int]]]) : 
        测试集中的用户行为序列的字典结构。第一层 key 是行为类型（click/cart/order），
        第二层 key 是用户 ID，value 是用户在该行为下的 item 序列。

    item_trans_scores (dict[tuple[int, int], float]) : aid → aid 的 转换分数（已归一化）。

    name (str) : 前缀名称，用于标识生成的特征列名。

    返回值：
    features (pd.DataFrame) :
        每行对应 order_candidate 中的一行样本（session与给session召回的order物品），包含以下列：
            - {name}_click_sum / mean / max / min / last
            - {name}_cart_sum / mean / max / min / last
        表示该候选 item 与用户最近点击 / 加购过的 item 序列之间的 bigram 统计特征。

    特征含义：
    对于候选物品 i ∈ order_candidate 和 session 的点击 / 加购序列 acts：
        - sum：∑ item_trans_scores[(a, i)]，a ∈ acts
        - mean：平均值
        - max：最大值
        - min：最小值
        - last：最后一个 a 到 i 的 bigram 分数
        若 acts 为空，则上述特征值为 -1。
    """
    features = {}

    # 遍历两种行为类型：CLICK 和 CART
    # 以CLICK或CART的物品作为为转换起点
    for src_act in [CLICK, CART]:
        if src_act == CLICK:
            src_act_name = 'click'
        else:
            src_act_name = 'cart'
        print(f'正在生成{src_act_name}的bigram特征...')
        # 初始化存储特征统计量的列表
        sum_sim, mean_sim, max_sim, min_sim, last_sim = [], [], [], [], []

        # u为session id, i为aid
        for idx, u, i in tqdm(order_candidate[['session', 'aid']].itertuples(), total=len(order_candidate)):
            # 根据session id u，在用户行为字典中获取该用户在当前行为类型 (src_act) 下的物品列表；
            # 如果该用户没有记录，则返回空列表 []
            acts = user_2_test_action_with_type[src_act].get(u, [])

            # 初始化当前order候选物品 i 与历史行为物品之间的相似度列表
            sims = []

            if len(acts) > 0:
                # 若用户存在历史行为，则依次计算用户的每个历史物品 a 转换到当前order候选物品 i 的 转换分数的sum、mean、max、min和最后一个转换对的转换分数
                for a in acts:
                    sims.append(item_trans_scores.get((a, i), 0))
                # 计算各种统计特征：总和、均值、最大值、最小值以及最后一个转换分数
                sum_sim.append(np.sum(sims))
                mean_sim.append(np.mean(sims))
                max_sim.append(np.max(sims))
                min_sim.append(np.min(sims))
                last_sim.append(sims[-1])
            else:
                # 如果用户没有相关历史行为，则使用 -1 作为默认值
                sum_sim.append(-1)
                mean_sim.append(-1)
                max_sim.append(-1)
                min_sim.append(-1)
                last_sim.append(-1)

        # 把当前行为类型下计算的各统计特征，加入到 fea 字典中，构造特征列名
        features.update({
            name + f'_{src_act_name}_sum': sum_sim,
            name + f'_{src_act_name}_mean': mean_sim,
            name + f'_{src_act_name}_max': max_sim,
            name + f'_{src_act_name}_min': min_sim,
            name + f'_{src_act_name}_last': last_sim
        })
    return pd.DataFrame(features)


def make_recall_orders_bigram_feature(prefix, item_trans_scores):
    """
    为ORDER召回候选样本生成 Bigram 特征，并保存为 Parquet 文件。

    参数:
    prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。

    item_trans_scores (dict[tuple[int, int], float]) : 归一化后的 bigram 统计（转移分数）字典。
                                                    key 是 (aid1, aid2) 形式的物品转移对，value 是它们的归一化转移分数。
    返回值:
    None. bigram 特征被保存为一个 Parquet 文件.
    """
    print('正在加载数据...')
    # test_actions (df.dataframe): [session, aid, ts, type], 如果prefix=='train_', 为训练验证集中的验证数据，如果prefix=='test_', 为训练测试集中的测试数据。
    test_actions = load_data(prefix, test_only=True)

    # 读取ORDER召回候选样本（用于构造特征）
    # [session, aid, label]
    print('正在读取order召回结果...')
    if prefix == 'train_':
        order_candidate = pd.read_parquet(candidate_path + 'train_orders_candidate.parquet')
    else:
        order_candidate = pd.read_parquet(candidate_path + 'test_orders_candidate.parquet')

    # 构建验证集中 user_id → list[aid] 的行为字典，按行为类型分
    # user_2_test_action_with_type = dict{
    #               0(CLICK):dict{session1:[aid1,aid2...], ...},
    #               1(CART):dict{session1:[aid1,aid2...], ...}
    #               2(ORDER):dict{session1:[aid1,aid2...], ...}
    #               }
    print('正在构建用户行为列表...')
    user_2_test_action_with_type = {
        t: test_actions.loc[test_actions['type'] == t].groupby('session')['aid'].agg(list).to_dict()
        for t in tqdm([CLICK, CART, ORDER])
    }

    del test_actions
    gc.collect()

    # 生成bigram features
    # 包含以下列：
    #         - {name}_click_sum / mean / max / min / last
    #         - {name}_cart_sum / mean / max / min / last
    print('正在生成 bigram 特征...')
    bigram_features = gen_bigram_features(order_candidate, user_2_test_action_with_type, item_trans_scores, 'bigram_normed')

    del item_trans_scores, user_2_test_action_with_type
    gc.collect()

    # 拼接到order_candicate的df中
    # columns= [session, aid, bigram_normed_click_sum, bigram_normed_click_mean,
    #            bigram_normed_click_max, bigram_normed_click_min, bigram_normed_click_last, 
    #            bigram_normed_cart_sum, bigram_normed_cart_mean, bigram_normed_cart_max, 
    #            bigram_normed_cart_min, bigram_normed_cart_last]
    print('正在合并 bigram 特征到 order 召回结果中...')
    order_bigram = pl.concat([pl.DataFrame(order_candidate[['session', 'aid']]), pl.DataFrame(bigram_features)], how="horizontal")
    # 将 session 和 aid 的类型统一转换为 32 位整数（Int32）
    order_bigram = order_bigram.with_columns(pl.col(['session', 'aid']).cast(pl.Int32, strict=False))
    # 将所有 bigram 特征列转换为 32 位浮点数（Float32）
    order_bigram = order_bigram.with_columns(pl.col(['bigram_normed_click_sum', 'bigram_normed_click_mean',
           'bigram_normed_click_max', 'bigram_normed_click_min',
           'bigram_normed_click_last', 'bigram_normed_cart_sum',
           'bigram_normed_cart_mean', 'bigram_normed_cart_max',
           'bigram_normed_cart_min', 'bigram_normed_cart_last']).cast(pl.Float32, strict=False))

    # 保存特征
    print('正在保存 bigram 特征文件...')
    order_bigram.to_pandas.to_parquet(output_path + prefix + 'bigram_feature.parquet')
    print(f'bigram 特征文件已经保存到{output_path + prefix}bigram_feature.parquet中.')


if __name__ == '__main__':
    for prefix in ['test_', 'train_']:
        print(f'正在处理{prefix}...')

        print('----------------------------')
        print('| 正在构建item的转移分数...|')
        print('----------------------------')
        item_trans_scores = gen_item_transition_scores(prefix)

        print('------------------------------')
        print('| 正在构建item的bigram特征...|')
        print('------------------------------')
        make_recall_orders_bigram_feature(prefix, item_trans_scores)