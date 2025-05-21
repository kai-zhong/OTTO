import xgboost as xgb
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import os, sys, pickle, glob, gc
import itertools
from collections import Counter
import cudf, itertools

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
candidate_path = '../../data/candidate/'
cluster_path = '../../data/preprocess/'
temp_path = '../../data/temp/'
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
        train_actions (cudf.dataframe): 如果prefix=='train_'返回训练验证集中的训练数据，如果prefix=='test_'返回训练测试集中的训练数据。
        test_actions (cudf.dataframe): 如果prefix=='train_'返回训练验证集中的验证数据，如果prefix=='test_'返回训练测试集中的测试数据。
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
    

def gen_cluster_transition_matrix_with_gpu(merge_actions):
    """
    计算不同 aid 聚类（cluster_label）之间的转移概率矩阵。

    参数:
        merge_actions (cudf.DataFrame): 包含所有用户行为的合并数据。
            - 主要列: ['session', 'aid', 'ts', 'type', 'cluster_label']

    返回值:
        cluster_trans_prob_matrix (cudf.DataFrame): aid 的 cluster 之间的转移概率矩阵。
            - 列名: ['cluster_x', 'cluster_y', 'cluster_label_trans_prob']
            - 含义:
                - cluster_x: 起始 aid 的 cluster 编号
                - cluster_y: 转移目标 aid 的 cluster 编号
                - cluster_label_trans_prob: 从 cluster_x 到 cluster_y 的归一化转移概率（条件概率）
    """
    # 初始化结果列表，用于存储每个 cluster 的转移统计
    cluster_trans_prob_df_all = []
    # 聚类列名
    cluster_col = 'cluster_label'

    # 遍历每一个 cluster 编号（0 ~ max cluster ID）
    for i in tqdm(range(merge_actions[cluster_col].max() + 1)):

        # 选取当前 cluster 的 aid 且行为类型为点击（type == 0）的数据行
        # ['session', 'ts', 'cluster_label']
        row = merge_actions[(merge_actions[cluster_col] == i) & (merge_actions['type'] == 0)][['session', cluster_col]]

        # # 若该 cluster 中数据过多则下采样，以控制内存和计算时间
        # if len(row) > 3_000_000:
        #     row = row.sample(3_000_000, random_state=1208)

        # 根据 session 合并出在相同 session 中 click 行为 aid 的 cluster
        # ['session', 'ts', 'cluster_label']
        row = row.merge(merge_actions[merge_actions['type'] == 0][['session', cluster_col]], 
                        on='session', how='inner')

        # 统计当前 cluster_x 到 cluster_y 的共现次数，并归一化为转移概率
        # [cluster_x, cluster_y, cluster_label_trans_prob]
        cluster_trans_prob_df = row[[cluster_col + '_x', cluster_col + '_y']].to_pandas().value_counts(normalize=True).reset_index()
        cluster_trans_prob_df = cudf.DataFrame.from_pandas(cluster_trans_prob_df)
        cluster_trans_prob_df.columns = [cluster_col + '_x', cluster_col + '_y', cluster_col + '_trans_prob']

        # 加入总结果中
        cluster_trans_prob_df_all.append(cluster_trans_prob_df)

    # 合并所有 cluster 的转移数据
    # [cluster_x, cluster_y, cluster_label_trans_prob]
    cluster_trans_prob_matrix = cudf.concat(cluster_trans_prob_df_all)

    os.makedirs(temp_path, exist_ok=True)
    cluster_trans_prob_matrix.to_pandas().to_parquet(temp_path + prefix + '_cluster_trans_prob_matrix.parquet')

    return cluster_trans_prob_matrix


def gen_cluster_transition_matrix_with_cpu(merge_actions):
    """
    计算不同 aid 聚类（cluster_label）之间的转移概率矩阵。

    参数:
        merge_actions (cudf.DataFrame): 包含所有用户行为的合并数据。
            - 主要列: ['session', 'aid', 'ts', 'type', 'cluster_label']

    返回值:
        cluster_trans_prob_matrix (cudf.DataFrame): aid 的 cluster 之间的转移概率矩阵。
            - 列名: ['cluster_x', 'cluster_y', 'cluster_label_trans_prob']
            - 含义:
                - cluster_x: 起始 aid 的 cluster 编号
                - cluster_y: 转移目标 aid 的 cluster 编号
                - cluster_label_trans_prob: 从 cluster_x 到 cluster_y 的归一化转移概率（条件概率）
    """
    # 初始化结果列表，用于存储每个 cluster 的转移统计
    cluster_trans_prob_df_all = []
    # 聚类列名
    cluster_col = 'cluster_label'

    merge_actions = merge_actions.to_pandas()

    # 遍历每一个 cluster 编号（0 ~ max cluster ID）
    for i in tqdm(range(merge_actions[cluster_col].max() + 1)):

        # 选取当前 cluster 的 aid 且行为类型为点击（type == 0）的数据行
        # ['session', 'ts', 'cluster_label']
        row = merge_actions[(merge_actions[cluster_col] == i) & (merge_actions['type'] == 0)][['session', cluster_col]]

        # # 若该 cluster 中数据过多则下采样，以控制内存和计算时间
        # if len(row) > 3_000_000:
        #     row = row.sample(3_000_000, random_state=1208)

        # 根据 session 合并出在相同 session 中 click 行为 aid 的 cluster
        # ['session', 'ts', 'cluster_label']
        row = row.merge(merge_actions[merge_actions['type'] == 0][['session', cluster_col]], 
                        on='session', how='inner')

        # 统计当前 cluster_x 到 cluster_y 的共现次数，并归一化为转移概率
        # [cluster_x, cluster_y, cluster_label_trans_prob]
        cluster_trans_prob_df = row[[cluster_col + '_x', cluster_col + '_y']].value_counts(normalize=True).reset_index()
        cluster_trans_prob_df = cudf.DataFrame.from_pandas(cluster_trans_prob_df)
        cluster_trans_prob_df.columns = [cluster_col + '_x', cluster_col + '_y', cluster_col + '_trans_prob']

        # 加入总结果中
        cluster_trans_prob_df_all.append(cluster_trans_prob_df)

    # 合并所有 cluster 的转移数据
    # [cluster_x, cluster_y, cluster_label_trans_prob]
    cluster_trans_prob_matrix = cudf.concat(cluster_trans_prob_df_all)

    os.makedirs(temp_path, exist_ok=True)
    cluster_trans_prob_matrix.to_pandas().to_parquet(temp_path + prefix + '_cluster_trans_prob_matrix.parquet')

    return cluster_trans_prob_matrix


def gen_cluster_features(prefix, cand_type, candidate_file_name, 
                         test_last_aid, cluster_df, cluster_trans_prob_matrix,
                         chunk_size = 500000):
    """
    处理候选商品的 Parquet 文件，添加 cluster 转移概率特征，并保存结果。

    参数：
    - prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
    - cand_type (str) : 行为类型，包括click、cart、order
    - candidate_file_name (str) : 行为召回结果的文件名
    - test_last_aid (cudf.dataframe): 包含每个 session 最近一次交互的商品 aid, [session, aid]
    - cluster_df (cudf.dataframe) : 每个 aid 对应的 cluster, [aid, cluster_label]
    - cluster_co_matrix (cudf.dataframe) : cluster 对之间的转移概率矩阵, # [cluster_x, cluster_y, cluster_label_trans_prob]
    - chunk_size (int) : 每个分块的行数
    """
    # 召回结果路径
    path = candidate_path + prefix + candidate_file_name
    # 召回结果的总行数，用于后面分块读取
    total_rows = pq.ParquetFile(path).metadata.num_rows

    # 用于合并所有分块处理结果
    candidate_results = []

    # 分块计算aid的cluster的转换概率
    print(f'正在分块计算{cand_type}-召回 aid 的 cluster 的转换概率...')
    for i in tqdm(range(0, total_rows, chunk_size)):
        start_idx = i
        end_idx = min(start_idx + chunk_size, total_rows) 
        # 1. 读取召回结果文件当前 chunk
        # [session, aid, label]
        candidate_df = cudf.read_parquet(path)
        candidate_df = candidate_df.iloc[start_idx:end_idx].reset_index(drop=True)

        # [session, aid]
        candidate_df = candidate_df[['session', 'aid']].astype(np.int32)

        # 2. 合并最后一次交互的商品 aid_y
        # [session, aid_x, aid_y]
        candidate_df = candidate_df.merge(test_last_aid, on='session', how='inner')

        # 3. 为 aid_x 和 aid_y 加上聚类标签
        # [session, aid_x, aid_y, aid, cluster_label]
        candidate_df = candidate_df.merge(cluster_df, left_on='aid_x', right_on='aid', how='inner')
        # [session, aid_x, aid_y, cluster_label_x, cluster_label_y]
        # cluster_label_x是aid_x(召回的item)的cluster label
        # cluster_label_y是aid_y(用户最后一次交互的item)的cluster label
        candidate_df = candidate_df.merge(cluster_df, left_on='aid_y', right_on='aid', how='inner')

        # 4. 使用 cluster id 对查询转移概率
        # [session, aid_x, aid_y, cluster_label_x, cluster_label_y, cluster_label_trans_prob]
        candidate_df = candidate_df.merge(cluster_trans_prob_matrix, on=['cluster_label_x', 'cluster_label_y'], how='inner')

        # 5. 删除中间用不到的列，释放内存
        # [session, aid_x, cluster_label_trans_prob]
        del candidate_df['cluster_label_x'], candidate_df['cluster_label_y'], candidate_df['aid_y']
        gc.collect()

        candidate_df['cluster_label_trans_prob'] = candidate_df['cluster_label_trans_prob'].astype(np.float32)
        candidate_df.columns = ['session', 'aid', 'cluster_label_trans_prob']

        candidate_results.append(candidate_df.to_pandas())
        del candidate_df
        gc.collect()

    candidate_results = [cudf.DataFrame.from_pandas(df) for df in candidate_results]
    candidate_results = cudf.concat(candidate_results, ignore_index=True)

    # 6. 保存结果
    print(f'正在保存{prefix}{cand_type}-cluster 特征...')
    candidate_results.to_pandas().to_parquet(output_path + prefix + cand_type + '_cluster_trans_prob.parquet')
    print(f'{cand_type}-cluster 特征已经保存到{output_path + prefix + cand_type}_cluster_trans_prob.parquet 中.')

    del candidate_results
    gc.collect()


if __name__ == '__main__':
    for prefix in ['train_', 'test_']:
        print('--------------------------------------')
        print(f'| 正在进行{prefix}cluster特征生成...|')
        print('--------------------------------------')

        # 读取数据
        print('正在进行数据读取...')
        train_actions, test_actions = load_data(prefix)
        # [session, aid, ts, type]
        merge_actions = cudf.concat([train_actions, test_actions]).reset_index(drop=True)
        del train_actions
        gc.collect()

        # 读取聚类结果：每个 aid 对应的 cluster_16dim
        print('正在读取聚类结果...')
        # [aid, cluster]
        cluster_df = cudf.read_parquet(cluster_path + prefix + 'aid_cluster.parquet')
        # [aid, cluster_label]
        cluster_df.columns = ['aid', 'cluster_label']

        # 加入 aid 对应的 cluster label 信息
        # cluster label是通过w2v emb聚类生成的，所以会存在一些aid没有label，但是数量很小，可以直接用inner join去除
        # [session, aid, ts, type, cluster_label]
        merge_actions = merge_actions.merge(cluster_df, on='aid', how='inner')
        # [session, aid, ts, type, cluster_label]
        test_actions = test_actions.merge(cluster_df, on='aid', how='inner')

        # 取每个 session 中的最后一次交互 aid
        # [session, aid]
        test_last_aid = test_actions.groupby('session')['aid'].last().reset_index()
        del test_actions
        gc.collect()

        # 计算每个 cluster 之间的转移概率矩阵
        # [cluster_x, cluster_y, cluster_label_trans_prob]
        if os.path.exists(f"{temp_path}{prefix}cluster_trans_prob_matrix.parquet"):
            print(f"正在加载缓存文件: {temp_path}{prefix}cluster_trans_prob_matrix.parquet")
            cluster_trans_prob_matrix = cudf.read_parquet(f"{temp_path}{prefix}cluster_trans_prob_matrix.parquet")
        else:
            print('正在构建cluster转移概率矩阵...')
            # cluster_trans_prob_matrix = gen_cluster_transition_matrix_with_gpu(merge_actions)
            cluster_trans_prob_matrix = gen_cluster_transition_matrix_with_cpu(merge_actions)

        del merge_actions
        gc.collect()

        # 遍历三种行为召回结果候选集：点击、加入购物车、下单
        for cand_type, candidate_file_name in type2candfile.items():
            gen_cluster_features(prefix, cand_type, candidate_file_name, 
                                 test_last_aid, cluster_df, cluster_trans_prob_matrix, 
                                 chunk_size = 500000)