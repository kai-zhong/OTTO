import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, gc
import polars as pl
import yaml
import warnings

from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm.sklearn')


# 定义文件路径
candidate_path = '../../data/candidate/'
datamart_path = '../../data/feature/'
oof_path = '../../data/oof/'
model_path = '../../model_v4/'
temp_oof_path = '../../data/temp/'
feature_dict_path = '../../config/feature_config.yaml'
co_matrix_dict_path = '../../config/co_matrix_config.yaml'
oof_dict_path = '../../config/oof_config_v4.yaml'


def config_load(type_name):
    with open(feature_dict_path) as yml:
        feature_config = yaml.safe_load(yml)

    with open(co_matrix_dict_path) as yml:
        co_matrix_config = yaml.safe_load(yml)

    with open(oof_dict_path) as yml:
        oof_config = yaml.safe_load(yml)
    
    # 根据 type_name 获取对应的特征列表、协同矩阵列表和 OOF 字典
    FEATURES = feature_config.get(f'train_{type_name}', []) # 使用 get 避免 key 错误
    FEATURES += ['bigram_normed_click_sum', 'bigram_normed_click_mean',
                'bigram_normed_click_max', 'bigram_normed_click_min',
                'bigram_normed_click_last', 'bigram_normed_cart_sum',
                'bigram_normed_cart_mean', 'bigram_normed_cart_max',
                'bigram_normed_cart_min', 'bigram_normed_cart_last']
    co_matrix_list = co_matrix_config.get(f'train_{type_name}', [])
    oof_dict = oof_config.get(f'train_{type_name}', {})
    return FEATURES, co_matrix_list, oof_dict


def select_train_sample(type_name, neg_num = 3000000):
    """
    从候选数据中选择训练样本，包括所有正样本和部分负样本。
    参数:
        type_name (str) : 行为类型名称 ('click_all', 'cart', 'order')，用于构建文件名。
        neg_num (int) : 采样的负样本数量。
    返回：
        train (pl.DataFrame): 包含选定的训练样本。
    """
    if type_name == 'click_all':
        file_path = candidate_path + 'train_' + type_name + '_candidate.parquet'
    else:
        file_path = candidate_path + 'train_' + type_name + 's_candidate.parquet'
    print(f"正在读取候选文件: {file_path}")

    # 读取{type_name}召回候选集文件
    train_all = pd.read_parquet(file_path)
    train_all.fillna(0, inplace = True)
    print(f"读取完成，数据形状: {train_all.shape}")

    # 将 pandas df 转换为 Polars df 进行过滤和采样
    train_all_pl = pl.DataFrame(train_all)
    # 筛选出label值大于 0 的正样本
    train_pos = train_all_pl.filter(pl.col('label') > 0)
    # 筛选出目标值等于 0 的负样本
    train_neg = train_all_pl.filter(pl.col('label') == 0)

    # 对负样本进行随机采样
    print(f"负样本数量: {train_neg.shape[0]}，采样 {neg_num} 个")
    train_neg = train_neg.sample(neg_num)

    # 合并正样本和采样后的负样本
    train = pl.concat([train_pos, train_neg])
    # 按 session 和 aid 排序
    train = train.sort(['session', 'aid'])
    print(f"采样合并后数据形状: {train.shape}")

    del train_pos, train_neg, train_all, train_all_pl
    gc.collect()

    return train


def merge_feature_to_chunk(data_chunk,feature_df, join_on_keys, filter_on_key, 
                           chunk_filter_values, features_list, extra_cols_to_add=None):
    """
    根据主数据块对已加载的特征 DataFrame 进行过滤和列选择，并将其合并到主数据块上。

    Args:
        data_chunk (pl.DataFrame): 要合并特征的主数据块 Polars DataFrame。
        feature_df (pl.DataFrame): 已经加载到内存中的特征 Polars DataFrame。
        join_on_keys (list[str]): 用于执行 join 操作的列名列表。
        filter_on_key (str | None): 特征 DataFrame 中用于根据主数据块值进行过滤的列名。
                                     如果为 None，则不执行过滤。默认为 None。
        chunk_filter_values (pl.Series | None): 来自主数据块中 filter_on_key 列的唯一值序列。
                                                 如果 filter_on_key 不为 None，则需要提供此参数。
                                                 默认为 None。
        features_list (list[str] | None): 指定从特征 DataFrame 中选择要合并的特征列名列表。
                                               如果为 None，则选择除了过滤键之外的所有列（在 join 后保留）。
                                               join_on_keys 中的列总是会被包含。默认为 None。
        extra_cols_to_add (list[str] | None): 指定除了 features_list 和 join_on_keys 之外
                                              额外需要包含的列名列表。函数将选择此列表中且存在于
                                              feature_df 中的列。默认为 None。
    Returns:
        pl.DataFrame: 合并了指定特征后的主数据块 DataFrame。
    """
    # 1. 根据主数据块的值过滤特征数据
    if filter_on_key is not None and chunk_filter_values is not None:
        feature_df = feature_df.filter(pl.col(filter_on_key).is_in(chunk_filter_values))
    
    # 2. 选择要合并的特征列（包括 join_on_keys）
    # 确保 join_on_keys 列被选中
    use_cols = [i for i in list(feature_df.columns) if i in features_list]
    use_cols += join_on_keys

    if extra_cols_to_add is not None:
        for col in extra_cols_to_add:
            if col in feature_df.columns and col not in use_cols:
                use_cols.append(col)
    
    # 实际进行列选择
    feature_df = feature_df[use_cols]

    # 3. 将特征数据合并到主数据块
    data_chunk = data_chunk.join(feature_df, on=join_on_keys, how="left")
    del feature_df
    gc.collect()

    return data_chunk


def join_features(data_chunk, type_name, oof_dict, co_matrix_list, FEATURES):
    """
    将各种预计算的特征合并到给定的数据块中。

    参数:
        data_chunk (pl.DataFrame): 需要合并特征的数据块 (Polars DataFrame)。
        type_name (str): 行为类型名称 ('click_all', 'cart', 'order')。
        oof_dict (dict): OOF 文件名到特征列名的映射字典。
        co_matrix_list (list): 协同矩阵特征文件名的列表。
        FEATURES (list): 需要加载和合并的特征列名列表。

    返回：
        pd.DataFrame: 合并了特征的 pandas DataFrame。
    """
    prefix = 'train_'
    if type_name == 'click_all':
        feature_type_name = 'click'
    else:
        feature_type_name = type_name

    data_chunk = data_chunk.with_columns([
        pl.col('session').cast(pl.Int32),
        pl.col('aid').cast(pl.Int32)])

    # 提取当前数据块中唯一的 session 和 aid 列表，用于后续过滤特征数据
    data_chunk_session = list(data_chunk['session'].unique().to_pandas())
    data_chunk_aids = list(data_chunk['aid'].unique().to_pandas())

    print('正在合并bigram特征...')
    bigram_df = pl.read_parquet(datamart_path + prefix + 'bigram_feature.parquet')
    if '__index_level_0__' in list(bigram_df.columns):
        bigram_df = bigram_df.drop(['__index_level_0__'])
    data_chunk = data_chunk.join(bigram_df, on=['aid', 'session'], how='left')
    del bigram_df
    gc.collect()


    print('正在合并 OOF 特征...')
    # 合并 OOF (Out-of-Fold) 预测特征
    for oof_file_name in tqdm(oof_dict.keys()):
        # 读取 OOF Parquet 文件
        oof = pl.read_parquet(oof_path + oof_file_name)
        oof = oof.drop(['label'])
        # 删除不需要的列
        if '__index_level_0__' in oof.columns:
            oof = oof.drop(['__index_level_0__'])
        # 重命名预测列为字典中指定的新列名
        oof = oof.rename({'pred': oof_dict[oof_file_name]})
        # 将 OOF 特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(oof, on=['aid', 'session'], how="left")
        del oof
        gc.collect()



    print('正在合并 BPR 特征...')
    # 合并 BPR (Bayesian Personalized Ranking) 特征
    # [session, aid, label, bpr]，bpr列是session和商品的bpr embedding的相似度
    bpr_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_bpr_feature.parquet')
    # 删除不需要的列
    bpr_df = bpr_df.drop(['label'])
    # 将 Int64 类型的列转换为 Int32 (如果存在)，strict=False 允许转换失败
    bpr_df = bpr_df.with_columns(pl.col(pl.Int64).cast(pl.Int32, strict=False))
    # 将 BPR 特征按 aid 和 session 合并到数据块中
    data_chunk = data_chunk.join(bpr_df, on=['aid', 'session'], how="left")
    del bpr_df
    gc.collect()



    print('正在合并 Cosine Similarity 特征...')
    # 定义 Cosine Similarity 特征文件名列表
    cos_sim_list = ['aid_w2v_last_dist_64dim.parquet', 'aid_w2v_last_dist_16dim.parquet', # 每个{act_type}类型召回 aid 与每个session最后一个交互的商品的余弦相似度
                    'aid_w2v_hour_dist_64dim.parquet', 'aid_w2v_hour_dist_16dim.parquet', # 每个{act_type}类型召回 aid 与每个 session 最后一小时内交互的商品（无特定类型，不含最后一次） 的平均余弦相似度
                    'session_w2v_dist_64dim.parquet', 'session_w2v_dist_16dim.parquet']   # 每个{act_type}类型召回 aid 与每个 session 所有交互过的商品 embedding 的平均值 的余弦相似度

    # 循环合并 Cosine Similarity 特征
    # 合并完成后columns = ['session', 'aid', 'label', 'bpr', 
    #                   'aid_w2v_last_dist_64dim', 'aid_w2v_last_dist_16dim', 
    #                   'aid_w2v_hour_dist_64dim', 'aid_w2v_hour_dist_16dim', 
    #                   'session_w2v_dist_64dim', 'session_w2v_dist_16dim']
    for mat_name in tqdm(cos_sim_list):
        # print(f"  - {mat_name}")
        # 读取 Cosine Similarity Parquet 文件
        cos_sim_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_' + mat_name)
        # 检查并删除不需要的索引列
        if '__index_level_0__' in list(cos_sim_df.columns):
            # print('    - 删除 __index_level_0__ 列')
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
    for co_matrix_name in tqdm(co_matrix_list):
        # print(f"  - {co_matrix_name}")
        # 读取协同矩阵 Parquet 文件
        co_matrix = pl.read_parquet(datamart_path + co_matrix_name)
        # 检查并删除不需要的索引列
        if '__index_level_0__' in list(co_matrix.columns):
            co_matrix = co_matrix.drop(['__index_level_0__'])
        # 检查并删除不需要的 rank 列
        if 'rank' in list(co_matrix.columns):
            co_matrix = co_matrix.drop(['rank'])
        # 过滤只包含当前数据块中存在的 session 和 aid 的特征数据
        co_matrix = co_matrix.filter(pl.col("session").is_in(data_chunk_session))
        co_matrix = co_matrix.filter(pl.col("aid").is_in(data_chunk_aids))
        # 将协同矩阵特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(co_matrix, on=['aid', 'session'], how="left")
        del co_matrix
        gc.collect()
    


    print('正在合并 Same Vector 特征...')
    # 合并 Same Vector 特征 (基于 aid 的相似性特征)
    same_aid_df = pl.read_parquet(datamart_path + prefix + 'same_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=same_aid_df, 
                                        filter_on_key='aid', chunk_filter_values=data_chunk_aids, 
                                        join_on_keys=['aid'],
                                        features_list=FEATURES)

    print('正在合并 Cluster 特征...')
    # 合并 Cluster 特征 (基于用户行为序列聚类得到的转移概率特征)
    # cluster_label_trans_prob表示从该 session 的最后一次交互的商品所属的聚类 (cluster) 转移到该 aid（候选商品）所属的聚类 (cluster) 的概率。
    # [session, aid, cluster_label_trans_prob]
    cluster_prob_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_cluster_trans_prob.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=cluster_prob_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['aid', 'session'],
                                        features_list=FEATURES)

    print('正在合并 Session-Aid 特征...')
    # 合并 Session-Aid 特征 (如 Session 内 Aid 的行为计数、占比等)
    session_aid_df = pl.read_parquet(datamart_path + prefix + 'session_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_aid_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    print('正在合并 Last Chunk 特征...')
    # 合并 Last Chunk 特征 (基于 Session 最后一个活动块的 Session-Aid 特征)
    last_chunk_df = pl.read_parquet(datamart_path + prefix + 'last_chunk_session_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=last_chunk_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    print('正在合并 Session 特征...')
    session_df = pl.read_parquet(datamart_path + prefix + 'session_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session'],
                                        features_list=FEATURES,
                                        extra_cols_to_add=['day'])
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    print('正在合并 Session & Use Aid 特征...')
    session_use_aid_df = pl.read_parquet(datamart_path + prefix + 'session_use_aid_feat_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_use_aid_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False))

    print('正在合并 Aid & Day 特征...')
    aid_day_df = pl.read_parquet(datamart_path + prefix + 'aid_day_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=aid_day_df, 
                                        filter_on_key='aid', chunk_filter_values=data_chunk_aids, 
                                        join_on_keys=['day', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    data_chunk = data_chunk.drop('day')


    print('正在合并 Aid 特征...')
    # 合并 Aid 特征
    aid_features_df = pl.read_parquet(datamart_path + prefix + 'aid_features_df.parquet')
    aid_features_df = aid_features_df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32, strict=False),
        pl.col(pl.Int64).cast(pl.Int32, strict=False)])
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=aid_features_df, 
                                        filter_on_key='aid', chunk_filter_values=data_chunk_aids, 
                                        join_on_keys=['aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False))
    
    # 使用 pl.exclude(null_cols) 排除指定列，然后对剩余列填充 null 值为 0
    null_cols = ['last_action_diff_hour']
    data_chunk = data_chunk.with_columns(pl.exclude(null_cols).fill_null(0))
    # data_chunk.write_parquet("../../data/temp/temp.parquet")
    # del data_chunk
    # gc.collect()
    # data_chunk_pd = pd.read_parquet("../../data/temp/temp.parquet")
    chunk_size = 200_000
    chunks = []
    print('正在转换为pandas...')
    for i in tqdm(range(0, data_chunk.shape[0], chunk_size)):
        chunk = data_chunk.slice(i, chunk_size).to_pandas()
        chunks.append(chunk)

    del data_chunk
    gc.collect()

    data_chunk = pd.concat(chunks, ignore_index=True)

    return data_chunk


def feature_prepare(type_name):
    """
    加载配置、准备数据

    参数:
        type_name (str): 行为类型名称 ('click', 'click_all', 'cart', 'order')。

    返回：
        保存处理好的特征文件到 datamart_path + 'train_' + type_name + '_datamart.parquet'中
    """
    # --- 1. 加载配置文件 ---
    print("正在加载配置文件...")
    FEATURES, co_matrix_list, oof_dict = config_load(type_name)
    print(f"加载完成，特征数量: {len(FEATURES)}")

    # --- 2. 准备训练数据 ---
    join_feature_path = datamart_path + 'train_' + type_name + '_datamart.parquet'
    if not os.path.exists(join_feature_path):
        print("正在选择训练样本...")
        train = select_train_sample(type_name, neg_num=8000000)
        print('正在合并特征...')
        train = join_features(train, type_name, oof_dict, co_matrix_list, FEATURES)

        # 保存计算结果到 Parquet 文件
        print('正在保存合并特征文件...')
        train.to_parquet(join_feature_path)
        print(f"文件 {join_feature_path} 已成功生成并保存。")

    else:
        print(f"文件 {join_feature_path} 已存在，正在加载合并特征...")
        train = pl.read_parquet(join_feature_path).to_pandas()
        print(f"已从现有文件加载合并特征数据到 'train' 变量。")

    train = train[['session', 'aid', 'label'] + FEATURES]
    gc.collect()

    return train
    

def model_train(type_name, train):
    """
    模型训练

    参数:
        type_name (str) : 行为类型名称 ('click', 'click_all', 'cart', 'order')。
        train (pandas.df) : 训练数据集。 

    返回：
        
    """
    FEATURES, co_matrix_list, oof_dict = config_load(type_name)
    
    # # 用于存储所有折叠的 OOF 预测结果
    # result_all = pd.DataFrame()

    os.makedirs(temp_oof_path, exist_ok=True)
    # List to store the paths of temporary OOF files saved for each fold
    oof_fold_files = []

    # 初始化 GroupKFold，按 session 分组，分成 5 折
    skf = GroupKFold(n_splits=5)

    # 遍历 GroupKFold 生成的训练集和验证集索引
    # train 是进行分割的数据，train['label'] 是label，groups=train['session'] 指定分组依据
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train['label'], groups=train['session'] )):
        print(f"\n--- Fold {fold+1} ---")
        print('正在提取当前Fold的训练和验证数据索引...')
        # 根据索引从 train 中提取当前Fold的训练集和验证集数据
        # [session, aid, label]
        X_train = train.loc[train_idx, :].copy()
        X_valid = train.loc[valid_idx, :].copy()

        # 获取当前折叠验证集的所有 session ID，用于后续过滤特征数据
        valid_group_id = list(X_valid['session'].unique())

        print('正在加载并合并当前折叠的特征...')
        # 从之前保存的 datamart 文件中加载完整的特征数据
        features_df = pl.read_parquet(datamart_path + 'train_' + type_name + '_datamart.parquet')

        # 合并训练集特征
        print('  - 合并训练集特征')
        # 将训练集数据转换为 Polars DataFrame 进行合并
        X_train_pl = pl.DataFrame(X_train)
        # 将特征数据合并到训练集数据中
        X_train_merged = X_train_pl.join(features_df, on = ['session', 'aid'], how = 'left')
        # 按 session 和 aid 排序 
        X_train_merged = X_train_merged.sort(['session', 'aid'])
        X_train_pd = X_train_merged.to_pandas()
        del X_train, X_train_pl, X_train_merged
        gc.collect()

        print('  - 合并验证集特征')
        # 将验证集数据转换为 Polars DataFrame 进行合并
        X_valid_pl = pl.DataFrame(X_valid)
        # 将特征数据合并到验证集数据中
        X_valid_merged = X_valid_pl.join(features_df, on = ['session', 'aid'], how = 'left')
        # 按 session 和 aid 排序
        X_valid_merged = X_valid_merged.sort(['session', 'aid'])
        X_valid_pd = X_valid_merged.to_pandas()
        del X_valid, X_valid_pl, X_valid_merged 
        del features_df
        gc.collect()

        print('正在准备 LightGBM 数据...')
        # LightGBM 排序需要每个 group (session) 的样本数量
        train_group_sizes = X_train_pd.groupby('session').size().values
        valid_group_sizes = X_valid_pd.groupby('session').size().values

        # 准备 LightGBM 的训练数据 (特征矩阵 X, 标签 y, 分组信息 group)
        X_train_lgb = X_train_pd[FEATURES] # 训练集特征
        y_train_lgb = X_train_pd['label'] # 训练集label
        X_valid_lgb = X_valid_pd[FEATURES] # 验证集特征
        y_valid_lgb = X_valid_pd['label'] # 验证集label

        # 定义 LightGBM 排序模型参数
        # objective: 'lambdarank' 或 'pairwise' 用于排序任务
        # metric: 评估指标，'map@k' 是常用的排序指标
        # device: 'gpu' 使用 GPU 加速
        # verbose: -1 抑制训练过程中的详细输出，除非达到 log_evaluation 周期
        # n_jobs: -1 使用所有可用的 CPU 核心进行并行计算 (GPU 训练时可能主要影响数据准备)
        param = {
            'objective': 'lambdarank', # 排序任务目标函数
            # 'metric': f'map@{20}', # 评估指标，例如 Mean Average Precision at 20
            'metric': 'map',  # 核心指标名称
            'eval_at': [20],  # 指定评估位置@20
            'learning_rate': .05, # 学习率
            'n_estimators': 10000, # 树的数量 (迭代次数)
            'max_depth': 7, # 树的最大深度
            'seed': 42, # 随机种子，用于复现结果
            'n_jobs': -1, # 使用所有可用核心
            'device': 'cpu', # 使用 CPU 训练
            'verbose': -1,
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
        model_output_path = os.path.join(model_path, type_name, f'LGBM_fold{fold}_{type_name}.txt') # LightGBM 模型通常保存为 .txt 文件
        print(f"正在保存模型到 {model_output_path} ...")
        ranker.booster_.save_model(model_output_path) # 使用 booster_.save_model 方法保存模型
        print("模型保存完成。")

        del X_train_pd, y_train_lgb, X_valid_pd, y_valid_lgb, train_group_sizes, valid_group_sizes
        gc.collect()


        # --- 4. 在验证集上进行预测并保存 OOF 结果 ---
        print('正在验证集上进行预测...')
        # 从候选文件中加载当前折叠的验证集数据
        if type_name == 'click_all':
            valid_candidate_path = candidate_path + 'train_' + type_name + '_candidate.parquet'
        else:
            valid_candidate_path = candidate_path + 'train_' + type_name + 's_candidate.parquet'
        valid_all = pl.read_parquet(valid_candidate_path)
        # 转换数据类型并填充 NaN
        valid_all = valid_all.with_columns([
            pl.col(['session','aid']).cast(pl.Int32),
            pl.col(['label']).cast(pl.Float32)
        ])
        valid_all = valid_all.fill_null(0)
        # 过滤只保留当前折叠验证集中的 session
        valid_all = valid_all.filter(pl.col('session').is_in(valid_group_id))

        # 合并特征到验证集数据中
        print('合并特征到验证集数据中...')
        valid_all_featured = join_features(valid_all, type_name, oof_dict, co_matrix_list, FEATURES)
        del valid_all # 删除 Polars 副本
        gc.collect()

        print('正在生成预测结果...')
        # # 使用训练好的模型对验证集特征进行预测
        # result = ranker.predict(valid_all_featured[FEATURES])

        # # 将预测结果添加到验证集 DataFrame 中
        # valid_all_featured['pred'] = result

        chunk_size = 100000  # 根据你的内存大小调整这个批处理大小
        predictions = []     # 用于存储每个批次的预测结果

        num_samples = valid_all_featured.shape[0]
        print(f"总样本数: {num_samples}，批处理大小: {chunk_size}")

        for i in tqdm(range(0, num_samples, chunk_size), desc="正在进行分块预测"):
            # 获取当前批次的索引范围
            start_idx = i
            end_idx = min(i + chunk_size, num_samples)

            # 提取当前批次的数据（只选择特征列）
            # 使用 iloc 进行基于位置的索引，通常更稳定且可能对内存更友好
            X_batch = valid_all_featured.iloc[start_idx:end_idx][FEATURES]

            # 对当前批次进行预测
            batch_predictions = ranker.predict(X_batch)

            # 将当前批次的预测结果添加到列表中
            predictions.append(batch_predictions)

            # 可选：删除批次数据和预测结果，帮助内存回收
            del X_batch, batch_predictions
            gc.collect()

        valid_all_featured = valid_all_featured[['session', 'aid', 'label']]
        gc.collect()

        # 将所有批次的预测结果拼接起来
        # 如果预测结果是 NumPy 数组，可以使用 np.concatenate
        # 如果是 Series 或 DataFrame，使用 pd.concat
        # LightGBM 的 predict 通常返回 NumPy 数组
        result = np.concatenate(predictions, axis=0)

        print("分块预测完成。")

        # --- 保存当前折叠的 OOF 预测结果到临时文件 ---
        # 临时文件名为 type_name_foldN_temp.parquet
        temp_oof_file = os.path.join(temp_oof_path, f'{type_name}_fold{fold}_temp.parquet')
        print(f"正在保存 Fold {fold+1} 的 OOF 预测结果到临时文件: {temp_oof_file} ...")

        valid_all_featured = valid_all_featured[['session', 'aid', 'label']]
        gc.collect()

        # 将拼接后的预测结果添加到验证集 DataFrame 中
        # 确保 result 的顺序与 valid_all_featured 的行顺序一致
        valid_all_featured['pred'] = result

        # 删除预测结果数组，释放内存
        del result
        gc.collect()

        # 将当前折叠的验证集预测结果追加到总结果 DataFrame 中
        # result_all = pd.concat([result_all, valid_all_featured[['session', 'aid', 'label', 'pred']]], ignore_index=True)
        
        # 保存为 Parquet 文件
        valid_all_featured.to_parquet(temp_oof_file, index=False)
        print(f"Fold {fold+1} OOF 预测结果保存完成。")
        oof_fold_files.append(temp_oof_file)

        # 删除当前折叠的验证集 DataFrame，释放内存
        del valid_all_featured
        gc.collect()

    # --- 5. 保存所有折叠的 OOF 预测结果 ---
    print("\n所有折叠训练和预测完成。")
    print("正在加载并拼接所有折叠的临时 OOF 预测结果文件...")
    # 用于存储从文件加载的每个折叠的 DataFrame
    result_all = []
    for fpath in tqdm(oof_fold_files):
        fold_df = pd.read_parquet(fpath)
        result_all.append(fold_df)

    print("正在拼接所有折叠的数据...")
    # 使用 pd.concat 拼接所有加载的 DataFrame
    result_all = pd.concat(result_all, ignore_index=True)
    gc.collect()

    print(f"总 OOF 预测结果形状: {result_all.shape}")

    # 构建 OOF 预测文件的保存路径
    if type_name != 'click_all':
        oof_output_path = f'{oof_path}{type_name}_train_lgbm_v4.parquet' # 修改文件名以反映 LightGBM
    else:
        oof_output_path = f'{oof_path}click_train_lgbm_v4_all_target.parquet' # 修改文件名以反映 LightGBM

    print(f"正在保存总 OOF 预测结果到 {oof_output_path} ...")
    result_all.to_parquet(oof_output_path, index=False) # index=False 避免写入索引
    print("总 OOF 预测结果保存完成。")

    del result_all # 删除总结果 DataFrame 释放内存
    gc.collect() # 手动触发垃圾回收




if __name__ == '__main__':
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)

    # 遍历不同的行为类型，分别进行训练和预测
    for type_name in ['click', 'click_all', 'cart', 'order']:
        print(f"\n====== 正在处理行为类型: {type_name} ======")
        # 构建 OOF 预测文件的保存路径
        if type_name != 'click_all':
            oof_output_path = f'{oof_path}{type_name}_train_lgbm_v4.parquet' # 修改文件名以反映 LightGBM

        else:
            oof_output_path = f'{oof_path}click_train_lgbm_v4_all_target.parquet' # 修改文件名以反映 LightGBM

        if os.path.exists(oof_output_path):
            print(f'{oof_output_path} 已存在，跳过处理。')
            continue

        # 为每种类型创建一个子目录来保存模型
        current_model_path = os.path.join(model_path, type_name)
        os.makedirs(current_model_path, exist_ok=True)

        # 合并特征文件，准备训练数据
        print(f"\n====== 正在准备{type_name}模型的训练数据 ======")
        train = feature_prepare(type_name)

        # GroupKFold 交叉验证训练
        print("正在进行 GroupKFold 交叉验证训练...")
        model_train(type_name, train)


        print(f"====== 行为类型 {type_name} 处理完成 ======")