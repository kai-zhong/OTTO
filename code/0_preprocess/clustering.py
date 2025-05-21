import scanpy as sc
import pandas as pd
import numpy as np

w2v_path = '../../data/preprocess/'     # w2v embedding的数据存储路径
output_path = '../../data/preprocess/'

full_data = True    # 控制使用的数据集范围


def load_data(w2v_path, dims=16, use_full_data=True):
    """
    功能：加载训练集或测试集中的 word2vec 商品嵌入向量数据，并按 aid 排序返回。

    参数：
        w2v_path (str): 嵌入向量文件的目录路径。
        dims (int): 嵌入向量的维度。
        use_full_data (bool): 是否使用完整数据。

    返回：
        w2v (pd.DataFrame): 包含 aid 及其对应嵌入向量的 DataFrame，并按 aid 升序排序。
    """
    print('正在载入w2v数据...')
    if use_full_data == True:
        prefix = 'test_'
    else:
        prefix = 'train_'

    # 读取商品embedding数据文件，并按aid从小到大排序
    w2v = pd.read_parquet(w2v_path + prefix + f'w2v_output_{dims}dims.parquet')
    w2v = w2v.sort_values('aid').reset_index(drop=True)
    return w2v


def clustering(w2v, output_path, use_full_data=True):
    """
    功能：对输入的商品嵌入向量进行聚类（Leiden算法），并保存 aid 与聚类标签的对应关系。

    参数：
        w2v (pd.DataFrame): 包含商品 ID 和嵌入向量的数据表，第一列为 'aid'，其余列为向量维度。
        output_path (str): 聚类结果保存路径。
        use_full_data (bool): 是否使用完整数据。

    返回：
        None。函数会将聚类标签保存为 parquet 文件：`[prefix]aid_cluster.parquet`。
    """
    if use_full_data == True:
        prefix = 'test_'
    else:
        prefix = 'train_'

    # === 聚类构建 ===
    print('正在构建聚类...')
    # === 创建 AnnData 对象，用于 scanpy 的后续聚类处理 ===
    # 从 w2v 中取出嵌入向量（去掉第一列 aid），构建 AnnData 数据结构
    #w2v.iloc[:, 1:].values：从 w2v DataFrame 中提取从第二列开始的数据（即所有的 vec_ 列），用于表示每个 aid 的嵌入向量。iloc[:, 1:] 排除了第一列 aid，取其余列的数据。
    #sc.AnnData(X = ...)：AnnData 是 scanpy 用来存储数据的基本数据结构。在此，它存储了从 w2v 中提取的嵌入数据。X 表示数据矩阵，每行对应一个数据点（在这个上下文中是一个 aid），每列对应一个特征。
    X_all = sc.AnnData(X = w2v.iloc[:, 1:].values)

    # === 构建邻接图（每个点与其他点的邻接关系）===
    # 使用 scanpy 的 neighbors 方法构建 k 近邻图，准备进行图上的聚类
    # use_rep='X' 表示使用 X_all 中的主矩阵作为表示
    # n_neighbors=64 表示每个点寻找 64 个邻居
    # method='umap' 指定使用 UMAP 方法计算邻居距离（虽然 UMAP 常用于降维，但此处只用于图构建）
    sc.pp.neighbors(X_all, use_rep = 'X', n_neighbors=64, method='umap')

    # === 使用 Leiden 算法进行聚类 ===
    # scanpy 的 tl.leiden 会基于邻接图进行社区发现，将数据划分为若干聚类
    # 聚类结果会被保存在 X_all.obs['leiden'] 中（字符串类型）
    sc.tl.leiden(X_all)

    # ===将 w2v 数据框中的 aid 列与对应的聚类标签（leiden）合并===
    print('正在构建聚类标签数据...')
    # 仅保留 aid 列，作为新 DataFrame 的基础
    aid_df = w2v[['aid']].copy()
    # 将 X_all.obs['leiden']（聚类结果）作为一列添加到 aid_df 中
    aid_df['cluster'] = list(X_all.obs['leiden'])  
    # 将 'cluster' 列的数据类型转换为 np.int16，以节省内存
    aid_df['cluster'] = aid_df['cluster'].astype(np.int16)

    print('正在保存聚类标签数据文件...')
    # 将 aid_df 保存为 .parquet 格式的文件，存储路径为 output_path + prefix + 'aid_cluster.parquet'
    aid_df.to_parquet(output_path + prefix + 'aid_cluster.parquet')  # 将 aid_df 数据框保存为 Parquet 格式的文件，便于后续加载使用
    print(f"已保存到{output_path + prefix}aid_cluster.parquet")

if __name__ == '__main__':
    # 载入word2vec的商品emb数据
    w2v = load_data(w2v_path=w2v_path, dims=16, use_full_data=full_data)
    # 执行聚类
    clustering(w2v=w2v, output_path=output_path, use_full_data=full_data)
