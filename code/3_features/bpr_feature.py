import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

candidate_path = '../../data/candidate/'
bpr_path = '../../data/preprocess/'
output_path = '../../data/feature/'


type2candfile = {'click' : 'clicks_candidate.parquet',
                'cart' : 'carts_candidate.parquet', 
                'order' : 'orders_candidate.parquet'}


def user_item_emb_dot(user_ids, item_ids, u2emb, i2emb):
    """
    计算用户向量与物品向量之间的点积，用于生成BPR评分特征。

    参数：
        user_ids（Iterable）：用户ID序列，dataframe中 'session' 那一列。
        item_ids（Iterable）：物品ID序列，dateframe中 'aid' 那一列。
        u2emb（dict）：用户ID到其嵌入向量（ndarray）的映射。
        i2emb（dict）：物品ID到其嵌入向量（ndarray）的映射。

    返回：
        np.ndarray：每对 (user_id, item_id) 对应的点积结果，作为BPR特征分数（一维数组）。
    """

    # 如果嵌入字典中找不到对应的用户/物品，使用默认的0向量（65维）
    # 向量维度是(64+1)维，最后一维是偏置项
    default_emb = np.zeros(64 + 1)

    # 构造用户嵌入矩阵（按顺序将每个用户嵌入堆叠为二维数组）
    # shape = (len(user_ids), embedding_dim)
    u_mat = np.stack([u2emb.get(u, default_emb) for u in user_ids])

    # 构造物品嵌入矩阵
    # shape = (len(item_ids), embedding_dim)
    i_mat = np.stack([i2emb.get(i, default_emb) for i in item_ids])

    # 对应位置相乘后按行求和（即计算点积）
    return np.sum(u_mat * i_mat, axis=1)


def gen_bpr_features_for_recall(prefix, cand_type, candidate_file_name, u2emb, i2emb):
    """
    对点击、加购、下单候选集生成 BPR 特征（用户与物品嵌入向量点积），并保存为 parquet 文件。

    参数：
        prefix (str) : 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        cand_type (str) : 正在处理的行为类型(click, cart, order).
        candidate_file_name (str) : recall的结果文件名
        u2emb（dict）：用户ID到其嵌入向量（ndarray）的映射。
        i2emb（dict）：物品ID到其嵌入向量（ndarray）的映射。

    返回：
        None：结果以 parquet 文件形式保存至指定路径，无函数返回值。
    """
    # columns = [session, aid, label]
    candidate_df = pd.read_parquet(candidate_path + prefix + candidate_file_name)

    # 分块处理
    chunk_size = 500000
    chunk_num = len(candidate_df) // chunk_size

    # 分块计算所有召回物品的 BPR 分数(用户与物品嵌入向量点积)
    pred_chunks = []
    print(f'正在计算{cand_type}召回物品的BPR分数...')
    for chunk_index in tqdm(range(chunk_num + 1)):
        start_idx = chunk_index * chunk_size
        end_idx = (chunk_index + 1) * chunk_size

        # 获取当前块的用户和物品 ID
        # [session]
        user_ids = candidate_df['session'].iloc[start_idx:end_idx]
        # [aid]
        item_ids = candidate_df['aid'].iloc[start_idx:end_idx]

        # 计算当前块的 BPR 分数
        chunk_pred = user_item_emb_dot(user_ids, item_ids, u2emb, i2emb)

        pred_chunks.append(chunk_pred)

    # 将所有块的结果拼接成一个完整数组
    # （len(candidate_df), ）
    pred = np.concatenate(pred_chunks)

    # 将bpr特征加入到recall结果中
    # columns = [session, aid, label, bpr]
    candidate_df['bpr'] = pred

    # 保存为 parquet 文件
    print(f'正在保存{prefix}{cand_type}的BPR特征文件...')
    candidate_df.to_parquet(f'{output_path}{prefix}{cand_type}_bpr_feature.parquet')
    print(f'文件已经保存到{output_path}{prefix}{cand_type}_bpr_feature.parquet中.')


if __name__ == '__main__':
    for prefix in ['train_', 'test_']:
        print(f'正在处理{prefix}...')

        print(f'正在读取用户embedding文件...')
        with open(f'{bpr_path+prefix}u2emb.pkl', 'rb') as fp:
            u2emb = pickle.load(fp)
        print(f'正在读取物品embedding文件...')
        with open(f'{bpr_path+prefix}i2emb.pkl', 'rb') as fp:
            i2emb = pickle.load(fp)

        for cand_type, candidate_file_name in type2candfile.items():
            print(f'正在处理{cand_type}召回结果...')
            gen_bpr_features_for_recall(prefix, cand_type, candidate_file_name, u2emb, i2emb)
            


        
