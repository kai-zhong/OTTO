import scipy.sparse as sparse
import implicit
import pandas as pd
import numpy as np
import pickle
import os

USER_ID = 'session' # 用户 ID 列名
ITEM_ID = 'aid'     # 物品 ID 列名
SEED = 42

full_data_path = '../../data/train_test/'   # 完整数据集路径（用于最终提交）
with_valid_path = '../../data/train_valid/' # 训练 + 验证数据路径（用于调参）
output_path = '../../data/preprocess/'      # 嵌入向量保存路径

train_file = 'train.parquet'
test_file = 'test.parquet'

full_data = False    # 控制使用的数据集范围

# 定义模型超参数
epoch, emb_size = 1000, 64  # epoch: 训练轮数，emb_size: 嵌入向量的维度


if __name__ == '__main__':
    # 确保输出路径存在，否则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)


    # ==== 数据加载 ====
    print('正在载入数据...')
    # 使用的数据是已经经过拆分，包含4列 session、aid、ts、type，分别是Session号，物品id，交互时间戳，交互类型
    # 此外还按照session号从小到大、ts从小到大排序
    if full_data == True:
        df_train = pd.read_parquet(full_data_path + train_file)
        df_test = pd.read_parquet(full_data_path + test_file)
        prefix = 'test_'
    else:
        df_train = pd.read_parquet(with_valid_path + train_file)
        df_test = pd.read_parquet(with_valid_path + test_file)
        prefix = 'train_'

    # ==== 数据合并与 ID 映射 ====
    print('正在进行数据合并与 ID 映射...')
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    # 将原始 session ID 编码为整数标签（user_label）
    # 对用户 ID 列进行编码（例如 session ID 是字符串，如 'A', 'B', 'C'）
    # pd.factorize 会返回两个对象：
    # - 第一个是一个整数数组，用来表示每个用户的唯一编号（从 0 开始）
    # - 第二个是一个唯一值的数组，表示原始的用户 ID 顺序
    #
    # 示例：
    # df[USER_ID] = ['A', 'B', 'A', 'C']
    # factorize(df[USER_ID]) 返回：
    #    - user_label = [0, 1, 0, 2]     （编码后的标签）
    #    - user_idx = ['A', 'B', 'C']    （原始唯一 ID）
    df['user_label'], user_idx = pd.factorize(df[USER_ID])
    # 将原始 aid 编码为整数标签（item_label）
    df['item_label'], item_idx = pd.factorize(df[ITEM_ID])

    # ==== 构建用户-物品交互的稀疏矩阵 ====
    print('正在构建用户-物品交互的稀疏矩阵...')
    # 使用 CSR（Compressed Sparse Row）格式，矩阵值为 1 表示用户与物品有交互
    sparse_item_user = sparse.csr_matrix((np.ones(len(df)), (df['user_label'], df['item_label'])))

    # ==== 训练 BPR 模型 ====
    print('正在训练 BPR 模型...')
    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=emb_size,       # 嵌入维度
        regularization=0.001,   # 正则化系数
        iterations=epoch,       # 训练轮数
        random_state=SEED
    )
    model.fit(sparse_item_user) # 使用稀疏矩阵训练模型

    # 将训练结果转换为字典
    # user_idx 是原始用户 ID，model.user_factors 是用户嵌入向量
    # 向量维度是(64+1)维，最后一维是偏置项
    u2emb = dict(zip(user_idx, model.user_factors))
    # item_idx 是原始物品 ID，model.item_factors 是物品嵌入向量
    # 向量维度是(64+1)维，最后一维是偏置项
    i2emb = dict(zip(item_idx, model.item_factors))
    

    print('正在保存用户embedding...')
    with open(f'{output_path+prefix}u2emb.pkl', 'wb') as fp:
        pickle.dump(u2emb, fp)
    print(f'用户嵌入向量已保存到{output_path+prefix}u2emb.pkl文件中.')

    print('正在保存物品embedding...')
    with open(f'{output_path+prefix}i2emb.pkl', 'wb') as fp:
        pickle.dump(i2emb, fp)
    print(f'用户嵌入向量已保存到{output_path+prefix}i2emb.pkl文件中.')