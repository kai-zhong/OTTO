# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# 导入 LightGBM 库，用于构建排序模型
import lightgbm as lgb
# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 tqdm 库，用于显示进度条
from tqdm import tqdm
# 导入 os, sys, pickle, glob, gc 库，用于文件操作、系统交互、序列化、文件查找和垃圾回收
import os, sys, pickle, glob, gc
# 导入 itertools 库，用于创建高效迭代器
import itertools
# 导入 collections 库，用于提供额外的数据结构，如 Counter
from collections import Counter
# 导入 random 库，用于生成随机数
import random
# 导入 polars 库，用于高性能的 DataFrame 操作 (原代码使用，保留)
import polars as pl
# 导入 yaml 库，用于读取 YAML 配置文件
import yaml

# 导入 GroupKFold，用于分组交叉验证
from sklearn.model_selection import GroupKFold
# 导入 matplotlib.pyplot，用于绘图 (原代码保留，尽管在此片段中未直接使用)
import matplotlib.pyplot as plt
# 导入 xgboost.plot_importance (原代码保留，尽管在此片段中未直接使用)
from xgboost import plot_importance
# 导入 cudf 和 itertools (原代码使用 RAPIDS cudf，保留)
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)
# 设置 cudf 的默认整数和浮点数位宽
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)


# -

def select_train_sample(candidate_path, type_name):
    """
    从候选数据中选择训练样本，包括所有正样本和部分负样本。

    参数:
        candidate_path (str): 候选数据文件的目录路径。
        type_name (str): 行为类型名称 ('click_all', 'cart', 'order')，用于构建文件名。

    返回：
        pd.DataFrame: 包含选定训练样本的 pandas DataFrame。
    """
    # 构建候选数据文件的完整路径
    if type_name == 'click_all':
        file_path = candidate_path + 'train_' + type_name + '_candidate.parquet'
    else:
        file_path = candidate_path + 'train_' + type_name + 's_candidate.parquet'
    print(f"正在读取候选文件: {file_path}")
    # 使用 pandas 读取 Parquet 文件
    train_all = pd.read_parquet(file_path)
    # 填充 NaN 值为 0
    train_all.fillna(0, inplace = True)
    print(f"读取完成，数据形状: {train_all.shape}")

    # 定义负样本的采样数量
    n = 30000000

    # 将 pandas DataFrame 转换为 Polars DataFrame 进行过滤和采样 (原代码使用 Polars，保留此流程)
    train_all_pl = pl.DataFrame(train_all)
    # 筛选出目标值大于 0 的正样本
    train_true = train_all_pl.filter(pl.col('label') > 0)
    # 筛选出目标值等于 0 的负样本
    train_false = train_all_pl.filter(pl.col('label') == 0)
    # 对负样本进行随机采样
    print(f"负样本数量: {train_false.shape[0]}，采样 {n} 个")
    train_false = train_false.sample(n)

    # 合并正样本和采样后的负样本
    train = pl.concat([train_true, train_false])
    # 按 session 和 aid 排序
    train = train.sort(['session', 'aid'])
    print(f"采样合并后数据形状: {train.shape}")

    # 删除中间变量，释放内存
    del train_true, train_false, train_all, train_all_pl
    gc.collect()

    # 将 Polars DataFrame 转换回 pandas DataFrame 返回
    return train.to_pandas()


def join_features(data_chunk, type_name, datamart_path, oof_path, oof_dict, co_matrix_list, FEATURES):
    """
    将各种预计算的特征合并到给定的数据块中。

    参数:
        data_chunk (pl.DataFrame): 需要合并特征的数据块 (Polars DataFrame)。
        type_name (str): 行为类型名称 ('click_all', 'cart', 'order')。
        datamart_path (str): 特征数据文件的目录路径。
        oof_path (str): OOF (Out-of-Fold) 预测文件的目录路径。
        oof_dict (dict): OOF 文件名到特征列名的映射字典。
        co_matrix_list (list): 协同矩阵特征文件名的列表。
        FEATURES (list): 需要加载和合并的特征列名列表。

    返回：
        pd.DataFrame: 合并了特征的 pandas DataFrame。
    """

    # 根据 type_name 确定特征文件的前缀和类型名称
    prefix = 'train_' # 假设这里总是处理训练阶段的特征数据
    if type_name == 'click_all':
        feature_type_name = 'click'
    else:
        feature_type_name = type_name

    # 将数据块中的 session 和 aid 列转换为 Int32 类型，以确保与特征文件中的数据类型一致
    data_chunk = data_chunk.with_columns([
        pl.col('session').cast(pl.Int32),
        pl.col('aid').cast(pl.Int32)
    ])

    # 提取当前数据块中唯一的 session 和 aid 列表，用于后续过滤特征数据
    data_chunk_session = list(data_chunk['session'].unique().to_pandas())
    data_chunk_aids = list(data_chunk['aid'].unique().to_pandas())

    print('正在合并 OOF 特征...')
    # 合并 OOF (Out-of-Fold) 预测特征
    for oof_file_name in oof_dict.keys():
        print(f"  - {oof_file_name}")
        # 读取 OOF Parquet 文件
        oof = pl.read_parquet(oof_path + oof_file_name)
        # 删除不需要的列
        oof = oof.drop(['label', '__index_level_0__'])
        # 重命名预测列为字典中指定的新列名
        oof = oof.rename({'pred': oof_dict[oof_file_name]})
        # 将 OOF 特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(oof, on=['aid', 'session'], how="left")
        del oof # 删除 OOF DataFrame 释放内存
        gc.collect() # 手动触发垃圾回收

    print('正在合并 BPR 特征...')
    # 合并 BPR (Bayesian Personalized Ranking) 特征
    bpr_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_bpr_feature.parquet')
    # 删除不需要的列
    bpr_df = bpr_df.drop(['label'])
    # 将 Int64 类型的列转换为 Int32 (如果存在)，strict=False 允许转换失败
    bpr_df = bpr_df.with_columns(pl.col(pl.Int64).cast(pl.Int32, strict=False))
    # 将 BPR 特征按 aid 和 session 合并到数据块中
    data_chunk = data_chunk.join(bpr_df, on=['aid', 'session'], how="left")
    del bpr_df # 删除 BPR DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    print('正在合并 Cosine Similarity 特征...')
    # 定义 Cosine Similarity 特征文件名列表
    cos_sim_list = ['aid_w2v_last_dist_64dim.parquet', 'aid_w2v_last_dist_16dim.parquet',
                    'aid_w2v_hour_dist_64dim.parquet', 'aid_w2v_hour_dist_16dim.parquet',
                    'session_w2v_dist_64dim.parquet', 'session_w2v_dist_16dim.parquet']

    # 循环合并 Cosine Similarity 特征
    for mat_name in cos_sim_list:
        print(f"  - {mat_name}")
        # 读取 Cosine Similarity Parquet 文件
        cos_sim_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_' + mat_name)
        # 检查并删除不需要的索引列
        if '__index_level_0__' in list(cos_sim_df.columns):
            print('    - 删除 __index_level_0__ 列')
            cos_sim_df = cos_sim_df.drop(['__index_level_0__'])
        # 过滤只包含当前数据块中存在的 session 和 aid 的特征数据，减少内存占用
        cos_sim_df = cos_sim_df.filter(pl.col("session").is_in(data_chunk_session))
        cos_sim_df = cos_sim_df.filter(pl.col("aid").is_in(data_chunk_aids))
        # 将 Cosine Similarity 特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(cos_sim_df, on=['aid', 'session'], how="left")
        del cos_sim_df # 删除 Cosine Similarity DataFrame 释放内存
        gc.collect() # 手动触发垃圾回收

    print('正在合并协同矩阵特征...')
    # 合并协同矩阵特征
    for co_matrix_name in co_matrix_list:
        print(f"  - {co_matrix_name}")
        # 读取协同矩阵 Parquet 文件
        co_matrix = pl.read_parquet(datamart_path + co_matrix_name)
        # 检查并删除不需要的索引列
        if '__index_level_0__' in list(co_matrix.columns):
            print('    - 删除 __index_level_0__ 列')
            co_matrix = co_matrix.drop(['__index_level_0__'])
        # 检查并删除不需要的 rank 列
        if 'rank' in list(co_matrix.columns):
            print('    - 删除 rank 列')
            co_matrix = co_matrix.drop(['rank'])
        # 过滤只包含当前数据块中存在的 session 和 aid 的特征数据
        co_matrix = co_matrix.filter(pl.col("session").is_in(data_chunk_session))
        co_matrix = co_matrix.filter(pl.col("aid").is_in(data_chunk_aids))
        # 将协同矩阵特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(co_matrix, on=['aid', 'session'], how="left")
        del co_matrix # 删除协同矩阵 DataFrame 释放内存
        gc.collect() # 手动触发垃圾回收

    print('正在合并 Same Vector 特征...')
    # 合并 Same Vector 特征 (基于 aid 的相似性特征)
    # 注意：原代码这里打印了 mat_name，可能是复制错误，应该打印文件名
    print("  - same_aid_df.parquet")
    aid_cvr_features_df = pl.read_parquet(datamart_path + prefix + 'same_aid_df.parquet')
    # 过滤只包含当前数据块中存在的 aid 的特征数据
    aid_cvr_features_df = aid_cvr_features_df.filter(pl.col("aid").is_in(data_chunk_aids))
    # 检查并提取 FEATURES 中包含的列以及 aid 列
    use_cols = [i for i in list(aid_cvr_features_df.columns) if i in FEATURES]
    use_cols.append('aid') # 确保 aid 列被包含
    aid_cvr_features_df = aid_cvr_features_df[use_cols]
    # 将 Same Vector 特征按 aid 合并到数据块中
    data_chunk = data_chunk.join(aid_cvr_features_df, on=['aid'], how="left")
    del aid_cvr_features_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    print('正在合并 Cluster 特征...')
    # 合并 Cluster 特征 (基于用户行为序列聚类得到的转移概率特征)
    cluster_prob_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_cluster_trans_prob.parquet')
    # 过滤只包含当前数据块中存在的 session 的特征数据
    cluster_prob_df = cluster_prob_df.filter(pl.col("session").is_in(data_chunk_session))

    # 检查并提取 FEATURES 中包含的列以及 session 和 aid 列
    use_cols = [i for i in list(cluster_prob_df.columns) if i in FEATURES]
    use_cols.extend(['session', 'aid']) # 确保 session 和 aid 列被包含
    # 注意：原代码这里没有根据 use_cols 筛选 cluster_prob_df，直接 join。
    # 保持原样，但如果 cluster_prob_df 很大且包含很多不需要的列，这里可以优化。
    # cluster_prob_df = cluster_prob_df[use_cols] # 优化选项：先筛选列
    # 将 Cluster 特征按 aid 和 session 合并到数据块中
    data_chunk = data_chunk.join(cluster_prob_df, on=['aid', 'session'], how="left")
    del cluster_prob_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    print('正在合并 Session-Aid 特征...')
    # 合并 Session-Aid 特征 (如 Session 内 Aid 的行为计数、占比等)
    session_aid_df = pl.read_parquet(datamart_path + prefix + 'session_aid_df.parquet')
    # 过滤只包含当前数据块中存在的 session 的特征数据
    session_aid_df = session_aid_df.filter(pl.col("session").is_in(data_chunk_session))

    # 检查并提取 FEATURES 中包含的列以及 session 和 aid 列
    use_cols = [i for i in list(session_aid_df.columns) if i in FEATURES]
    use_cols.extend(['session', 'aid']) # 确保 session 和 aid 列被包含
    session_aid_df = session_aid_df[use_cols]

    # 将 Session-Aid 特征按 session 和 aid 合并到数据块中
    data_chunk = data_chunk.join(session_aid_df, on=['session', 'aid'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del session_aid_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收


    print('正在合并 Last Chunk 特征...')
    # 合并 Last Chunk 特征 (基于 Session 最后一个活动块的 Session-Aid 特征)
    last_chunk_df = pl.read_parquet(datamart_path + prefix + 'last_chunk_session_aid_df.parquet')
    # 过滤只包含当前数据块中存在的 session 的特征数据
    last_chunk_df = last_chunk_df.filter(pl.col("session").is_in(data_chunk_session))

    # 检查并提取 FEATURES 中包含的列以及 session 和 aid 列
    use_cols = [i for i in list(last_chunk_df.columns) if i in FEATURES]
    use_cols.extend(['session', 'aid']) # 确保 session 和 aid 列被包含
    last_chunk_df = last_chunk_df[use_cols]

    # 将 Last Chunk 特征按 session 和 aid 合并到数据块中
    data_chunk = data_chunk.join(last_chunk_df, on=['session', 'aid'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del last_chunk_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收


    print('正在合并 Session 特征...')
    # 合并 Session 特征 (如 Session 的总时长、事件数等)
    session_df = pl.read_parquet(datamart_path + prefix + 'session_df.parquet')
    # 过滤只包含当前数据块中存在的 session 的特征数据
    session_df = session_df.filter(pl.col("session").is_in(data_chunk_session))

    # 检查并提取 FEATURES 中包含的列以及 session 和 day 列
    use_cols = [i for i in list(session_df.columns) if i in FEATURES]
    use_cols.extend(['session', 'day']) # 确保 session 和 day 列被包含
    session_df = session_df[use_cols]

    # 将 Session 特征按 session 合并到数据块中
    data_chunk = data_chunk.join(session_df, on=['session'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del session_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    print('正在合并 Session & Use Aid 特征...')
    # 合并 Session & Use Aid 特征 (Session 中使用的商品的聚合特征)
    session_use_aid_df = pl.read_parquet(datamart_path + prefix + 'session_use_aid_feat_df.parquet')
    # 过滤只包含当前数据块中存在的 session 的特征数据
    session_use_aid_df = session_use_aid_df.filter(pl.col("session").is_in(data_chunk_session))

    # 检查并提取 FEATURES 中包含的列以及 session 列
    use_cols = [i for i in list(session_use_aid_df.columns) if i in FEATURES]
    use_cols.append('session') # 确保 session 列被包含
    session_use_aid_df = session_use_aid_df[use_cols]

    # 将 Session & Use Aid 特征按 session 合并到数据块中
    data_chunk = data_chunk.join(session_use_aid_df, on=['session'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del session_use_aid_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    print('正在合并 Aid & Day 特征...')
    # 合并 Aid & Day 特征 (商品在特定日期上的统计特征)
    session_day_df = pl.read_parquet(datamart_path + prefix + 'session_day_df.parquet')

    # 检查并提取 FEATURES 中包含的列以及 day 和 aid 列
    use_cols = [i for i in list(session_day_df.columns) if i in FEATURES]
    use_cols.extend(['day', 'aid']) # 确保 day 和 aid 列被包含
    session_day_df = session_day_df[use_cols]

    # 将 Aid & Day 特征按 day 和 aid 合并到数据块中
    data_chunk = data_chunk.join(session_day_df, on= ['day', 'aid'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del session_day_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收
    # 合并完成后，day 列不再需要，可以删除
    data_chunk = data_chunk.drop('day')

    print('正在合并 Aid 特征...')
    # 合并 Aid 特征 (商品的全局统计特征)
    aid_features_df = pl.read_parquet(datamart_path + prefix + 'aid_features_df.parquet')
    # 过滤只包含当前数据块中存在的 aid 的特征数据
    aid_features_df = aid_features_df.filter(pl.col("aid").is_in(data_chunk_aids))

    # 检查并提取 FEATURES 中包含的列以及 aid 列
    use_cols = [i for i in list(aid_features_df.columns) if i in FEATURES]
    use_cols.append('aid') # 确保 aid 列被包含
    aid_features_df = aid_features_df[use_cols]
    # 将 Float64 和 Int64 类型的列转换为 Float32 和 Int32，节省内存
    aid_features_df = aid_features_df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32, strict=False),
        pl.col(pl.Int64).cast(pl.Int32, strict=False)
    ])

    # 将 Aid 特征按 aid 合并到数据块中
    data_chunk = data_chunk.join(aid_features_df, on= ['aid'], how="left")
    # 将 Float64 类型的列转换为 Float32，节省内存
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    del aid_features_df # 删除 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

    # 填充部分特定列 (如最后一次行为时间差) 的 NaN 值为 0
    # 注意：原代码只指定了 'last_action_diff_hour'，但 fill_null(0) 应用于所有列
    # 保持原代码逻辑，对除 null_cols 以外的所有列填充 0
    null_cols = ['last_action_diff_hour'] # 需要特殊处理或保留 NaN 的列，但原代码最终还是填充 0
    # 实际上，原代码是先排除 null_cols 进行 fill_null(0)，然后转 pandas，再对所有列 fillna(0)
    # 简化逻辑，直接在 pandas 中对所有列 fillna(0)
    # data_chunk = data_chunk.with_columns(pl.exclude(null_cols).fill_null(0)) # 这行在转 pandas 前执行
    data_chunk = data_chunk.to_pandas() # 转换为 pandas DataFrame
    data_chunk.fillna(0, inplace = True) # 在 pandas 中填充所有 NaN 为 0

    return data_chunk


def main(type_name, candidate_path, datamart_path, oof_path, model_path,
         feature_dict_path, co_matrix_dict_path, oof_dict_path):
    """
    主函数：加载配置、准备数据、训练 LightGBM 排序模型 (GroupKFold 交叉验证) 并保存 OOF 预测。

    参数:
        type_name (str): 行为类型名称 ('click', 'click_all', 'cart', 'order')。
        candidate_path (str): 候选数据文件的目录路径。
        datamart_path (str): 特征数据文件的目录路径。
        oof_path (str): OOF 预测文件的保存目录路径。
        model_path (str): 训练好的模型文件的保存目录路径。
        feature_dict_path (str): 特征配置文件的路径 (.yaml)。
        co_matrix_dict_path (str): 协同矩阵配置文件的路径 (.yaml)。
        oof_dict_path (str): OOF 配置文件的路径 (.yaml)。
    """

    # --- 1. 加载配置文件 ---
    print("正在加载配置文件...")
    with open(feature_dict_path) as yml:
        feature_config = yaml.safe_load(yml)

    with open(co_matrix_dict_path) as yml:
        co_matrix_config = yaml.safe_load(yml)

    with open(oof_dict_path) as yml:
        oof_config = yaml.safe_load(yml)

    # 根据 type_name 获取对应的特征列表、协同矩阵列表和 OOF 字典
    FEATURES = feature_config.get(f'train_{type_name}', []) # 使用 get 避免 key 错误
    co_matrix_list = co_matrix_config.get(f'train_{type_name}', [])
    oof_dict = oof_config.get(f'train_{type_name}', {})
    print(f"加载完成，特征数量: {len(FEATURES)}")

    # --- 2. 准备训练数据 ---
    print("正在选择训练样本并合并特征...")
    # 选择训练样本 (包含所有正样本和部分负样本)
    train = select_train_sample(candidate_path, type_name) # select_train_sample 现在返回 pandas DataFrame

    # 将各种特征合并到训练数据中
    # 注意：join_features 期望 Polars 输入，并返回 pandas。
    # 这里 train 已经是 pandas，需要先转 Polars 再调用 join_features。
    train_pl = pl.DataFrame(train)
    train = join_features(train_pl, type_name, datamart_path, oof_path, oof_dict, co_matrix_list, FEATURES)
    del train_pl # 删除 Polars 副本
    gc.collect()

    # 检查合并后的数据是否包含所有必需的列
    required_cols = ['session', 'aid', 'label'] + FEATURES
    if not all(col in train.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in train.columns]
        print(f"错误：合并特征后缺少以下列: {missing_cols}")
        # 可以选择在这里退出或处理错误
        return

    # 将合并特征后的完整训练数据保存到 datamart，方便后续加载
    datamart_output_path = datamart_path + 'train_' + type_name + '_datamart.parquet'
    print(f"正在保存合并特征后的训练数据到 {datamart_output_path} ...")
    train[['session', 'aid', 'label'] + FEATURES].to_parquet(datamart_output_path, index=False)
    print("保存完成。")

    # 仅保留 session, aid, target 列用于 GroupKFold 分割，节省内存
    train = train[['session', 'aid', 'label']].copy()
    del train # 删除大 DataFrame 释放内存
    gc.collect()

    # --- 3. GroupKFold 交叉验证训练 ---
    print("正在进行 GroupKFold 交叉验证训练...")
    # 用于存储所有折叠的 OOF 预测结果
    result_all = pd.DataFrame()
    # 初始化 GroupKFold，按 session 分组，分成 5 折
    skf = GroupKFold(n_splits=5)

    # 遍历 GroupKFold 生成的训练集和验证集索引
    # train 是进行分割的数据，train['label'] 是目标变量，groups=train['session'] 指定分组依据
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train['label'], groups=train['session'] )):
        print(f"\n--- Fold {fold+1} ---")

        print('正在提取当前折叠的训练和验证数据索引...')
        # 根据索引从 train 中提取当前折叠的训练集和验证集数据
        # 注意：这里只提取了 session, aid, target，特征需要重新加载合并
        X_train = train.loc[train_idx, :].copy()
        X_valid = train.loc[valid_idx, :].copy()

        # 获取当前折叠验证集的所有 session ID，用于后续过滤特征数据
        valid_group_id = list(X_valid['session'].unique())

        print('正在加载并合并当前折叠的特征...')
        # 从之前保存的 datamart 文件中加载完整的特征数据
        features_df = pl.read_parquet(datamart_output_path)

        # 合并训练集特征
        print('  - 合并训练集特征')
        # 将训练集索引数据转换为 Polars DataFrame 进行合并
        X_train_pl = pl.DataFrame(X_train)
        # 将特征数据合并到训练集索引数据中
        X_train_merged = X_train_pl.join(features_df, on = ['session', 'aid'], how = 'left')
        # 按 session 和 aid 排序 (可选，但有助于保持一致性)
        X_train_merged = X_train_merged.sort(['session', 'aid'])
        # 转换回 pandas DataFrame
        X_train_pd = X_train_merged.to_pandas()
        del X_train, X_train_pl, X_train_merged # 删除中间变量释放内存

        # 合并验证集特征
        print('  - 合并验证集特征')
        # 将验证集索引数据转换为 Polars DataFrame 进行合并
        X_valid_pl = pl.DataFrame(X_valid)
        # 将特征数据合并到验证集索引数据中
        X_valid_merged = X_valid_pl.join(features_df, on = ['session', 'aid'], how = 'left')
        # 按 session 和 aid 排序 (可选)
        X_valid_merged = X_valid_merged.sort(['session', 'aid'])
         # 转换回 pandas DataFrame
        X_valid_pd = X_valid_merged.to_pandas()
        del X_valid, X_valid_pl, X_valid_merged # 删除中间变量释放内存
        del features_df # 删除加载的完整特征数据 DataFrame 释放内存
        gc.collect() # 手动触发垃圾回收

        print('正在准备 LightGBM 数据...')
        # LightGBM 排序需要每个 group (session) 的样本数量
        train_group_sizes = X_train_pd.groupby('session').size().values
        valid_group_sizes = X_valid_pd.groupby('session').size().values

        # 准备 LightGBM 的训练数据 (特征矩阵 X, 目标变量 y, 分组信息 group)
        X_train_lgb = X_train_pd[FEATURES] # 训练集特征
        y_train_lgb = X_train_pd['label'] # 训练集目标
        X_valid_lgb = X_valid_pd[FEATURES] # 验证集特征
        y_valid_lgb = X_valid_pd['label'] # 验证集目标

        # 定义 LightGBM 排序模型参数
        # objective: 'lambdarank' 或 'pairwise' 用于排序任务
        # metric: 评估指标，'map@k' 是常用的排序指标
        # device: 'gpu' 使用 GPU 加速
        # verbose: -1 抑制训练过程中的详细输出，除非达到 log_evaluation 周期
        # n_jobs: -1 使用所有可用的 CPU 核心进行并行计算 (GPU 训练时可能主要影响数据准备)
        param = {
            'objective': 'lambdarank', # 排序任务目标函数
            'metric': f'map@{20}', # 评估指标，例如 Mean Average Precision at 20
            'learning_rate': .05, # 学习率
            'n_estimators': 100000, # 树的数量 (迭代次数)
            'max_depth': 7, # 树的最大深度
            'seed': 1208, # 随机种子，用于复现结果
            'n_jobs': -1, # 使用所有可用核心
            'device': 'gpu', # 使用 GPU 训练
            'verbose': -1, # 抑制详细输出
            'boosting_type': 'gbdt', # 梯度提升类型，gbdt 是默认和常用类型
            # 'gpu_platform_id': 0, # 如果有多个 GPU，可以指定平台 ID
            # 'gpu_device_id': 0, # 如果有多个 GPU，可以指定设备 ID
        }

        # 初始化 LightGBM 排序模型
        ranker = lgb.LGBMRanker(**param)

        print('正在训练 LightGBM 模型...')
        # 训练模型
        # X_train_lgb, y_train_lgb 是训练数据和目标
        # group=train_group_sizes 提供训练集的分组信息
        # eval_set=[(X_valid_lgb, y_valid_lgb)] 提供验证数据和目标
        # eval_group=[valid_group_sizes] 提供验证集的分组信息
        # callbacks 用于指定训练过程中的回调函数，如日志记录和早停
        callbacks = [lgb.log_evaluation(period=100), # 每 100 轮打印一次评估指标
                     lgb.early_stopping(stopping_rounds=100, verbose=True)] # 如果验证集指标连续 100 轮没有提升，则提前停止
        ranker.fit(X_train_lgb, y_train_lgb,
                   group=train_group_sizes,
                   eval_set=[(X_valid_lgb, y_valid_lgb)],
                   eval_group=[valid_group_sizes],
                   callbacks=callbacks)

        # 保存训练好的模型
        model_output_path = os.path.join(model_path, f'LGBM_fold{fold}_{type_name}.txt') # LightGBM 模型通常保存为 .txt 文件
        print(f"正在保存模型到 {model_output_path} ...")
        ranker.booster_.save_model(model_output_path) # 使用 booster_.save_model 方法保存模型
        print("模型保存完成。")

        # 删除训练和验证数据 DataFrame，释放内存
        del X_train_pd, y_train_lgb, X_valid_pd, y_valid_lgb, train_group_sizes, valid_group_sizes
        gc.collect()

        # --- 4. 在验证集上进行预测并保存 OOF 结果 ---
        print('正在验证集上进行预测...')
        # 从候选文件中加载当前折叠的验证集数据
        valid_all = pl.read_parquet(candidate_path + 'train_' + type_name + '_candidate.parquet')
        # 转换数据类型并填充 NaN
        valid_all = valid_all.with_columns([
            pl.col(['session','aid']).cast(pl.Int32),
            pl.col(['label']).cast(pl.Float32)
        ])
        valid_all = valid_all.fill_null(0)
        # 过滤只保留当前折叠验证集中的 session
        valid_all = valid_all.filter(pl.col('session').is_in(valid_group_id))

        # 合并特征到验证集数据中
        valid_all_featured = join_features(valid_all, type_name, datamart_path, oof_path, oof_dict, co_matrix_list, FEATURES)
        del valid_all # 删除 Polars 副本
        gc.collect()

        print('正在生成预测结果...')
        # 使用训练好的模型对验证集特征进行预测
        result = ranker.predict(valid_all_featured[FEATURES])

        # 将预测结果添加到验证集 DataFrame 中
        valid_all_featured['pred'] = result

        # 删除预测结果数组，释放内存
        del result
        gc.collect()

        # 将当前折叠的验证集预测结果追加到总结果 DataFrame 中
        result_all = pd.concat([result_all, valid_all_featured[['session', 'aid', 'label', 'pred']]], ignore_index=True)

        # 删除当前折叠的验证集 DataFrame，释放内存
        del valid_all_featured
        gc.collect()

    # --- 5. 保存所有折叠的 OOF 预测结果 ---
    print("\n所有折叠训练和预测完成。")
    print(f"总 OOF 预测结果形状: {result_all.shape}")

    # 构建 OOF 预测文件的保存路径
    if type_name != 'click_all':
        oof_output_path = f'{oof_path}{type_name}_train_lgbm_v3.parquet' # 修改文件名以反映 LightGBM
    else:
        oof_output_path = f'{oof_path}click_train_lgbm_v3_all_target.parquet' # 修改文件名以反映 LightGBM

    print(f"正在保存总 OOF 预测结果到 {oof_output_path} ...")
    result_all.to_parquet(oof_output_path, index=False) # index=False 避免写入索引
    print("总 OOF 预测结果保存完成。")

    del result_all # 删除总结果 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收

# --- 主程序执行部分 ---

# 定义文件路径
candidate_path = '../../data/candidate/'
datamart_path = '../../data/feature/'
oof_path = '../../data/oof/'
model_path = '../../model_v3/'
feature_dict_path = '../../config/feature_config.yaml'
co_matrix_dict_path = '../../config/co_matrix_config.yaml'
oof_dict_path = '../../config/oof_config.yaml'

# 确保模型保存目录存在
os.makedirs(model_path, exist_ok=True)
# 确保 OOF 保存目录存在
os.makedirs(oof_path, exist_ok=True)


# 遍历不同的行为类型，分别进行训练和预测
# 注意：这里假设 'click' 和 'click_all' 是不同的任务或目标
# 'click' 可能只针对点击行为的正样本，'click_all' 可能包含所有行为作为负样本的点击预测任务
for t in ['click', 'click_all', 'cart', 'order']:
    print(f"\n====== 正在处理行为类型: {t} ======")
    # 为每种类型创建一个子目录来保存模型
    current_model_path = os.path.join(model_path, t)
    os.makedirs(current_model_path, exist_ok=True)

    # 调用主函数进行训练和预测
    main(t, candidate_path, datamart_path, oof_path, current_model_path,
         feature_dict_path, co_matrix_dict_path, oof_dict_path)

    print(f"====== 行为类型 {t} 处理完成 ======")

