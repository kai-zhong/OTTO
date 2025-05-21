import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import cudf

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
w2v_path = '../../data/preprocess/'
candidate_path = '../../data/candidate/'
output_path = '../../data/feature/'

type2candfile = {'click' : 'clicks_candidate.parquet',
                'cart' : 'carts_candidate.parquet', 
                'order' : 'orders_candidate.parquet'}

def load_data(prefix, type='both'):
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
        train_actions = cudf.read_parquet(raw_opt_path + 'train.parquet')
        test_actions = cudf.read_parquet(raw_opt_path + 'test.parquet')
    else:
        train_actions = cudf.read_parquet(preprocess_path + 'train.parquet')
        test_actions = cudf.read_parquet(preprocess_path + 'test.parquet')
    if type == 'test':
        return test_actions
    elif type == 'train':
        return train_actions
    else:
        return train_actions, test_actions
    

def cos_similarity(X, Y):
    """
    计算余弦相似度
    """
    cos_sim = (X * Y).sum(axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    return cos_sim.reshape(-1)


def get_last_and_hour_actions(action_df):
    """
    构建训练集中 session 的目标商品 aid

    该函数返回两个数据集：
    1. 每个 session 的最后一次行为对应的 aid（作为目标 aid）
    2. 每个 session 最后 1 小时内（但不包含最后一次行为）的所有 aid（作为目标 aid）

    参数:
    action_df (pd.DataFrame) :用户行为数据，包含以下列：
                            - session : int，会话 ID
                            - aid     : int，商品 ID
                            - ts      : int，行为发生的时间戳

    返回值:
    last_actions (pd.DataFrame) :每个 session 的最后一次行为（目标 aid），包含以下列：
                                - session     : int，会话 ID
                                - target_aid  : int，该 session 的 target 商品 ID

    hour_actions (pd.DataFrame) :每个 session 最后 1 小时内但不包含最后一次行为的所有 aid，包含以下列：
                                - session     : int，会话 ID
                                - target_aid  : int，该 session 的 target 商品 ID
    """
    # -------------------------------
    # 1. 获取每个 session 中的最后一次行为（last action）
    # -------------------------------
    print('正在提取每个 session 中的最后一次行为...')
    # 获取每个 session 的最大时间戳（即最后一次行为的时间）
    # [session, ts]
    session_last_ts = action_df.groupby('session')['ts'].max().reset_index()

    # 合并最大时间戳信息到原始数据
    # [session, aid, ts_x, type, ts_y]
    last_actions = action_df.merge(session_last_ts, on='session', how='left')

    # 筛选出行为时间等于最大时间戳的记录（即 session 的最后一次行为）
    # [session, aid]
    last_actions = last_actions[last_actions['ts_x'] == last_actions['ts_y']][['session', 'aid']].drop_duplicates()

    # 按 session 和 aid 排序
    last_actions = last_actions.sort_values(['session', 'aid']).reset_index(drop=True)

    # 重命名 aid 为 'target_aid'，表明是目标行为商品
    last_actions.columns = ['session', 'target_aid']

    # -------------------------------
    # 2. 获取每个 session 最后 1 小时内的其他行为（不包含最后一次行为）
    # -------------------------------
    print('正在提取每个 session 最后 1 小时内的行为（不包含最后一次行为）...')
    
    # [session, ts_max]
    hour_actions = session_last_ts
    hour_actions.columns = ['session', 'ts_max']

    del session_last_ts
    gc.collect()
    
    # 计算 1 小时前的时间
    # [session, ts_max, ts_hour]
    hour_actions['ts_hour'] = hour_actions['ts_max'] - (1 * 60 * 60)        # 一小时前
    # session_last_ts['ts_day'] = session_last_ts['ts'] - (24 * 60 * 60)        # 一天前
    # session_last_ts['ts_week'] = session_last_ts['ts'] - (7 * 24 * 60 * 60)   # 一周前
    
    # 合并时间窗口信息
    # [session, aid, ts, type, ts_max, ts_hour]
    hour_actions = action_df.merge(hour_actions, on='session', how='left')

    # 筛选出：1 小时内的行为，且不是最后一次行为
    hour_actions = hour_actions[
        (hour_actions['ts'] >= hour_actions['ts_hour']) &
        (hour_actions['ts'] != hour_actions['ts_max'])
    ]

    # 保留 session 和 aid 列，并去重
    # [session, aid]
    hour_actions = hour_actions[['session', 'aid']].drop_duplicates()

    # 按 session 和 aid 排序
    hour_actions = hour_actions.sort_values(['session', 'aid']).reset_index(drop=True)

    # 重命名 aid 为 'target_aid'，表示是目标商品
    hour_actions.columns = ['session', 'target_aid']

    return last_actions, hour_actions


def w2v_session_dist_feature(prefix, act_type, candidate_file_name, 
                             test_actions, w2v_dim, chunk_size=20000):
    """
    为每条候选记录计算该商品的向量与其所属 session 的平均向量之间的余弦相似度，生成 session 级别的 w2v 距离特征。

    参数:
        prefix (str) : 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        act_type (str) : 正在处理的行为类型(click, cart, order).
        candidate_file_name (str) : recall的结果文件名
        test_actions (str) : (cudf.DataFrame): 包含 session 和 aid 的原始数据。
                                columns: ['session', 'aid', 'ts', 'type']
        w2v_dim (int) : 使用的w2v生成的item embedding维度.
        chunk_size (int): 每个分块处理的 session 数，默认值为 20000。

    返回:
        None: 该函数无返回值，会将计算的特征保存为 .parquet 文件。

    输出特征文件格式 (DataFrame):
        columns: ['session', 'aid', feature_name]
        feature_name特征表示 aid 向量与其 session 平均embedding向量之间的余弦相似度。
    """
    print('-------------------------------------------------')
    print(f'正在进行{prefix}{act_type} 用户物品w2v相似度计算...')
    print('--------------------------------------------------')
    # 1. 加载 Word2Vec 向量
    print(f'正在读取{w2v_dim}维w2v emb向量文件...')
    # 读取指定的 w2v 向量文件（包含了所有item的w2v嵌入向量）
    # columns = [aid, vec_1, vec_2, ..., vec_{w2v_dim}]
    w2v_df = cudf.read_parquet(w2v_path + prefix + f'w2v_output_{w2v_dim}dims.parquet')
    w2v_df = w2v_df.sort_values('aid').reset_index(drop=True)


    # 2. 计算每个 session 的平均向量（作为当前 session 的用户兴趣表示）
    print('正在计算session embedding表示...')
    # 将 session 中每个 aid 的向量取平均，得到 session 的平均embedding表示
    session_w2v = test_actions[['aid', 'session']]
    # columns = [aid, session, vec_1, vec_2, ..., vec_{w2v_dim}]
    session_w2v = session_w2v.merge(w2v_df, on='aid', how='left')
    session_w2v = session_w2v.fillna(0)
    # columns = [session, vec_1, vec_2, ..., vec_{w2v_dim}]
    session_w2v = session_w2v.iloc[:, 1:]  # 去掉 aid 列，保留 session 和向量维度
    # columns = [session, vec_1, vec_2, ..., vec_{w2v_dim}]
    session_w2v = session_w2v.groupby('session').mean().reset_index()


    # 3. 获取所有待处理的 session，并计算处理批次数
    print('正在提取session list...')
    # 从act_type类型的召回结果样本中获取所有 session id 构建成list，并分块处理
    # 读取候选对
    # columns = [session, aid, label]
    candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
    # 提取去重后的 session 列表
    # session_list(list) = [session1, session2, ...]
    session_list = list(candidate['session'].unique().to_pandas())
    # 分片总数
    chunk_num = int(len(session_list) / chunk_size) + 1
    del candidate
    gc.collect()


    # 4. 开始分批计算余弦相似度特征
    # 保存每一块的余弦相似度结果列表
    cos_sim_list = []
    feature_name = f'session_w2v_dist_{w2v_dim}dim'
    # 按块计算session与候选aid的余弦相似度
    print(f'正在计算session与{act_type}候选物品的余弦相似度...')
    for i in tqdm(range(chunk_num)):
        # 当前批次的 session 范围
        start = i * chunk_size
        end = (i + 1) * chunk_size

        # 读取候选对，并筛选出当前批次的 session
        # [session, aid]
        chunk_candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
        chunk_candidate['session'] = chunk_candidate['session'].astype(np.int32)
        chunk_candidate['aid'] = chunk_candidate['aid'].astype(np.int32)
        # 只保留当前批次的 session-aid 对
        # [session, aid]
        chunk_candidate = chunk_candidate[chunk_candidate['session'].isin(session_list[start:end])][['session', 'aid']]  

        # 获取 aid 对应的向量，作为“候选商品向量”
        # w2v只对click的商品生成了embedding，如果商品只被加购或购买，是不存在embedding的，会出现nan的情况
        # columns = [vec_1, vec_2, ..., vec_{w2v_dim}]
        aid_vec = chunk_candidate[['session', 'aid']].merge(
            w2v_df, on='aid', how='left'
        ).sort_values(['session', 'aid']).iloc[:, 2:]  # 去掉 session 和 aid，仅保留向量列
        # 未出现过的aid默认emb为全零
        aid_vec = aid_vec.fillna(0)


        # 获取 session 的平均向量，作为“目标兴趣向量”
        # columns = [vec_1, vec_2, ..., vec_{w2v_dim}]
        target_aid_vec = chunk_candidate[['session']].merge(
            session_w2v, on='session', how='left'
        ).sort_values(['session']).iloc[:, 1:]  # 只保留向量列，按 session 排序
        # 未出现过的session默认emb为全零
        target_aid_vec = target_aid_vec.fillna(0)

        # 计算候选商品向量 与 session 平均兴趣向量 之间的余弦相似度
        # columns = [session, aid, 'session_w2v_dist_{w2v_dim}dim']
        chunk_candidate[feature_name] = cos_similarity(aid_vec.values, target_aid_vec.values)

        # 转为 pandas（方便后续 concat），并添加到结果列表
        chunk_candidate = chunk_candidate.to_pandas()
        cos_sim_list.append(chunk_candidate)

        del chunk_candidate
        gc.collect()

    # 合并所有批次结果
    candidate = pd.concat(cos_sim_list)
    candidate[feature_name] = candidate[feature_name].astype(np.float32)  # 设置类型为 float32 减小文件体积

    # 保存为 parquet 格式特征文件
    save_name = prefix + act_type + '_' + feature_name
    print(f'正在保存{prefix}-{act_type} w2v_session_dist 特征文件({w2v_dim}维)...')
    candidate.to_parquet(output_path + save_name + '.parquet')
    print(f'特征文件已保存到{output_path}{save_name}.parquet.')
    


def w2v_aid_dist_feature(prefix, act_type, target_type, candidate_file_name, 
                         target_actions, w2v_dim, chunk_size=20000):
    """
    基于 word2vec 向量计算每个候选 aid 与 session 中目标 aid（最后一次交互的aid/最后一小时交互的aid（不包括最后一次））之间的余弦相似度

    参数:
    prefix (str) : 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
    act_type (str) : 正在处理的行为类型(click, cart, order).
    target_type (str) : 正在使用的target aid的来源（最后一次交互的aid【last】/最后一小时交互的aid（不包括最后一次）【hour】）
    candidate_file_name (str) : recall的结果文件名
    target_actions (cudf.DataFrame) : 包含每个 session 的目标商品 aid（['session', 'target_aid']）(last_actions或hour_actions)
    w2v_dim (int) : 使用的w2v生成的item embedding维度.
    chunk_size (int): 每个分块处理的 session 数，默认值为 20000。

    返回:
    无直接返回值。该函数会将生成的特征保存为 parquet 文件，包含以下列:
        - session (int)
        - aid (int)
        - feature_name (float)：候选 aid 和目标 aid 的平均余弦相似度
    """
    print('-------------------------------------------------')
    print(f'正在进行{prefix}{act_type}-{target_type} 用户物品w2v相似度计算...')
    print('--------------------------------------------------')
    # 加载 Word2Vec 向量
    print(f'正在读取{w2v_dim}维w2v emb向量文件...')
    # 读取指定的 w2v 向量文件（包含了所有item的w2v嵌入向量）
    # columns = [aid, vec_1, vec_2, ..., vec_{w2v_dim}]
    w2v_df = cudf.read_parquet(w2v_path + prefix + f'w2v_output_{w2v_dim}dims.parquet')
    w2v_df = w2v_df.sort_values('aid').reset_index(drop=True)

    # 读取候选 aid 数据（只用于提取 session 列表，后面会分块再次加载）
    print('正在提取session list...')
    # [session, aid, label]
    candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
    session_list = list(candidate['session'].unique().to_pandas())  # 转为 pandas 以便切片
    chunk_num = int(len(session_list) / chunk_size) + 1  # 计算分块数
    del candidate
    gc.collect()

    # 开始分批计算余弦相似度特征
    # 保存每一块的余弦相似度结果列表
    cos_sim_list = []  
    feature_name = f'aid_w2v_{target_type}_dist_{w2v_dim}dim'
    print(f'正在计算{target_type}-物品与{act_type}候选物品的余弦相似度...')
    for i in tqdm(range(chunk_num)):
        start = i * chunk_size
        end = (i + 1) * chunk_size

        # 每个 chunk 再次读取候选文件
        # [session, aid, label]
        chunk_candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
        chunk_candidate['session'] = chunk_candidate['session'].astype(np.int32)
        chunk_candidate['aid'] = chunk_candidate['aid'].astype(np.int32)

        # 仅保留当前 chunk 内的 session 数据
        # [session, aid]
        chunk_candidate = chunk_candidate[chunk_candidate['session'].isin(session_list[start:end])][['session', 'aid']]
        gc.collect()

        # 与 target_actions 进行 inner join，将 target_aid 附到 (session, aid)召回样本对上
        # [session, aid, target_aid]
        chunk_candidate = chunk_candidate.merge(target_actions, on='session', how='inner')

        # 排序是为了后面余弦相似度计算时两侧向量对应准确
        # [session, aid, target_aid]
        chunk_candidate = chunk_candidate.sort_values(['session', 'aid']).reset_index(drop=True)

        # 提取候选 aid 对应的 word2vec 向量，去除前两列 ['session', 'aid']，只保留向量维度列
        # w2v只对click的商品生成了embedding，如果商品只被加购或购买，是不存在embedding的，会出现nan的情况
        # columns = [vec_1, vec_2, ..., vec_{w2v_dim}]
        aid_vec = chunk_candidate[['session', 'aid']].merge(
            w2v_df, on='aid', how='left'
        ).sort_values(['session', 'aid']).iloc[:, 2:]
        # 未出现过的aid默认emb为全零
        aid_vec = aid_vec.fillna(0)

        # 提取目标 aid 对应的 word2vec 向量，注意左表是 target_aid，右表是 aid，偏移多一列
        # columns = [vec_1, vec_2, ..., vec_{w2v_dim}]
        target_aid_vec = chunk_candidate[['session', 'target_aid']].merge(
            w2v_df, left_on='target_aid', right_on='aid', how='left'
        ).sort_values(['session', 'aid']).iloc[:, 3:]
        # 未出现过的session默认emb为全零
        target_aid_vec = target_aid_vec.fillna(0)

        # 计算候选 aid 和目标 aid 的余弦相似度，返回一个一维向量
        chunk_candidate[feature_name] = cos_similarity(aid_vec.values, target_aid_vec.values)

        # 有可能多个相同 session, aid 对重复出现（用户点击/加购/购买多次），这里求平均相似度
        chunk_candidate = chunk_candidate.groupby(['session', 'aid'])[feature_name].mean().reset_index()

        # 将结果转为 pandas 后收集
        chunk_candidate = chunk_candidate.to_pandas()
        cos_sim_list.append(chunk_candidate)

        # 清除当前 chunk 的变量释放显存
        del chunk_candidate
        gc.collect()
    
    # 合并所有 chunk 的结果
    candidate = pd.concat(cos_sim_list)

    # 转换为 float32
    candidate[feature_name] = candidate[feature_name].astype(np.float32)

    # 保存为 parquet 格式到指定路径
    save_name = prefix + act_type + '_' + feature_name
    print(f'正在保存{prefix}-{act_type} w2v_aid_dist 特征文件({w2v_dim}维)...')
    candidate.to_parquet(output_path + save_name + '.parquet')
    print(f'特征文件已保存到{output_path}{save_name}.parquet.')
    gc.collect()


def gen_w2v_distance_feature(prefix, act_type, candidate_file_name):
    """
    基于 session 和 aid 的 Word2Vec 表征，为指定数据前缀（如 train_ 或 test_）生成多维度距离特征，并保存到输出路径。
    生成了16维特征和64维特征。

    参数：
        prefix (str) : 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        act_type (str) : 正在处理的行为类型(click, cart, order).
        candidate_file_name (str) : recall的结果文件名
    返回:
        None: 该函数不返回值，功能为保存多个 Word2Vec 距离特征 DataFrame 到磁盘。

    生成的特征文件格式 (DataFrame):
        - columns: ['session', 'aid', 'session_w2v_dist_16dim']
        - 特征维度包括: 16维和64维。
        - 文件将被保存为 .parquet 格式。
    """
    print('正在加载数据...')
    # columns = [session, aid, ts, type]
    test_actions = load_data(prefix, 'test')
    # 生成16维的特征
    w2v_session_dist_feature(prefix, act_type, candidate_file_name, test_actions, w2v_dim=16, chunk_size=10000)
    # 生成64维的特征
    w2v_session_dist_feature(prefix, act_type, candidate_file_name, test_actions, w2v_dim=64, chunk_size=10000)

    gc.collect()

    # last_actions([session, target_aid]) : 每个 session 的最后一次行为对应的 aid
    # hour_actions([session, target_aid]) : 每个 session 最后 1 小时内（但不包含最后一次行为）的所有 aid
    last_actions, hour_actions = get_last_and_hour_actions(test_actions)

    w2v_aid_dist_feature(prefix, act_type, 'last', candidate_file_name, last_actions, w2v_dim=16, chunk_size=10000)
    w2v_aid_dist_feature(prefix, act_type, 'last', candidate_file_name, last_actions, w2v_dim=64, chunk_size=10000)
    w2v_aid_dist_feature(prefix, act_type, 'hour', candidate_file_name, hour_actions, w2v_dim=16, chunk_size=10000)
    w2v_aid_dist_feature(prefix, act_type, 'hour', candidate_file_name, hour_actions, w2v_dim=64, chunk_size=10000)

    gc.collect()


if __name__ == '__main__':
    for prefix in ['train_', 'test_']:
        for act_type, candidate_file_name in type2candfile.items():
            print('--------------------------------------')
            print(f'| 正在进行{prefix}{act_type}类w2v特征生成...|')
            print('--------------------------------------')
            gen_w2v_distance_feature(prefix, act_type, candidate_file_name)