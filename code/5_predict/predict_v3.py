import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, gc
import polars as pl
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm.sklearn')

candidate_path = '../../data/candidate/'
datamart_path ='../../data/feature/'
oof_path = '../../output/'
output_path = '../../output/'
model_path = '../../model_v3/'
feature_dict_path = '../../config/feature_config.yaml'
co_matrix_dict_path = '../../config/co_matrix_config.yaml'
oof_dict_path = '../../config/oof_config_v3.yaml'

def config_load(type_name):
    with open(feature_dict_path) as yml:
        feature_config = yaml.safe_load(yml)

    with open(co_matrix_dict_path) as yml:
        co_matrix_config = yaml.safe_load(yml)

    with open(oof_dict_path) as yml:
        oof_config = yaml.safe_load(yml)
    
    # 根据 type_name 获取对应的特征列表、协同矩阵列表和 OOF 字典
    FEATURES = feature_config.get(f'test_{type_name}', []) # 使用 get 避免 key 错误
    co_matrix_list = co_matrix_config.get(f'test_{type_name}', [])
    oof_dict = oof_config.get(f'test_{type_name}', {})
    return FEATURES, co_matrix_list, oof_dict


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
    prefix = 'test_'
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


    # print('正在合并 OOF 特征...')
    # 合并 OOF (Out-of-Fold) 预测特征
    for oof_file_name in oof_dict.keys():
        # 读取 OOF Parquet 文件
        oof = pl.read_parquet(oof_path + oof_file_name)
        # 删除不需要的列
        if '__index_level_0__' in oof.columns:
            oof = oof.drop(['__index_level_0__'])
        # 重命名预测列为字典中指定的新列名
        oof = oof.rename({'pred': oof_dict[oof_file_name]})
        # 将 OOF 特征按 aid 和 session 合并到数据块中
        data_chunk = data_chunk.join(oof, on=['aid', 'session'], how="left")
        del oof
        gc.collect()



    # print('正在合并 BPR 特征...')
    # 合并 BPR (Bayesian Personalized Ranking) 特征
    # [session, aid, label, bpr]，bpr列是session和商品的bpr embedding的相似度
    bpr_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_bpr_feature.parquet')
    # 将 Int64 类型的列转换为 Int32 (如果存在)，strict=False 允许转换失败
    bpr_df = bpr_df.with_columns(pl.col(pl.Int64).cast(pl.Int32, strict=False))
    # 将 BPR 特征按 aid 和 session 合并到数据块中
    data_chunk = data_chunk.join(bpr_df, on=['aid', 'session'], how="left")
    del bpr_df
    gc.collect()



    # print('正在合并 Cosine Similarity 特征...')
    # 定义 Cosine Similarity 特征文件名列表
    cos_sim_list = ['aid_w2v_last_dist_64dim.parquet', 'aid_w2v_last_dist_16dim.parquet', # 每个{act_type}类型召回 aid 与每个session最后一个交互的商品的余弦相似度
                    'aid_w2v_hour_dist_64dim.parquet', 'aid_w2v_hour_dist_16dim.parquet', # 每个{act_type}类型召回 aid 与每个 session 最后一小时内交互的商品（无特定类型，不含最后一次） 的平均余弦相似度
                    'session_w2v_dist_64dim.parquet', 'session_w2v_dist_16dim.parquet']   # 每个{act_type}类型召回 aid 与每个 session 所有交互过的商品 embedding 的平均值 的余弦相似度

    # 循环合并 Cosine Similarity 特征
    # 合并完成后columns = ['session', 'aid', 'label', 'bpr', 
    #                   'aid_w2v_last_dist_64dim', 'aid_w2v_last_dist_16dim', 
    #                   'aid_w2v_hour_dist_64dim', 'aid_w2v_hour_dist_16dim', 
    #                   'session_w2v_dist_64dim', 'session_w2v_dist_16dim']
    for mat_name in cos_sim_list:
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


    # print('正在合并协同矩阵特征...')
    # 合并协同矩阵特征
    for co_matrix_name in co_matrix_list:
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
    


    # print('正在合并 Same Vector 特征...')
    # 合并 Same Vector 特征 (基于 aid 的相似性特征)
    same_aid_df = pl.read_parquet(datamart_path + prefix + 'same_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=same_aid_df, 
                                        filter_on_key='aid', chunk_filter_values=data_chunk_aids, 
                                        join_on_keys=['aid'],
                                        features_list=FEATURES)

    # print('正在合并 Cluster 特征...')
    # 合并 Cluster 特征 (基于用户行为序列聚类得到的转移概率特征)
    # cluster_label_trans_prob表示从该 session 的最后一次交互的商品所属的聚类 (cluster) 转移到该 aid（候选商品）所属的聚类 (cluster) 的概率。
    # [session, aid, cluster_label_trans_prob]
    cluster_prob_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_cluster_trans_prob.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=cluster_prob_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['aid', 'session'],
                                        features_list=FEATURES)

    # print('正在合并 Session-Aid 特征...')
    # 合并 Session-Aid 特征 (如 Session 内 Aid 的行为计数、占比等)
    session_aid_df = pl.read_parquet(datamart_path + prefix + 'session_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_aid_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    # print('正在合并 Last Chunk 特征...')
    # 合并 Last Chunk 特征 (基于 Session 最后一个活动块的 Session-Aid 特征)
    last_chunk_df = pl.read_parquet(datamart_path + prefix + 'last_chunk_session_aid_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=last_chunk_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    # print('正在合并 Session 特征...')
    session_df = pl.read_parquet(datamart_path + prefix + 'session_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session'],
                                        features_list=FEATURES,
                                        extra_cols_to_add=['day'])
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast

    # print('正在合并 Session & Use Aid 特征...')
    session_use_aid_df = pl.read_parquet(datamart_path + prefix + 'session_use_aid_feat_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=session_use_aid_df, 
                                        filter_on_key='session', chunk_filter_values=data_chunk_session, 
                                        join_on_keys=['session'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False))

    # print('正在合并 Aid & Day 特征...')
    aid_day_df = pl.read_parquet(datamart_path + prefix + 'aid_day_df.parquet')
    data_chunk = merge_feature_to_chunk(data_chunk=data_chunk, feature_df=aid_day_df, 
                                        filter_on_key='aid', chunk_filter_values=data_chunk_aids, 
                                        join_on_keys=['day', 'aid'],
                                        features_list=FEATURES)
    data_chunk = data_chunk.with_columns(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
    data_chunk = data_chunk.drop('day')


    # print('正在合并 Aid 特征...')
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


    data_chunk = data_chunk.to_pandas()
    # chunk_size = 200_000
    # chunks = []
    # print('正在转换为pandas...')
    # for i in tqdm(range(0, data_chunk.shape[0], chunk_size)):
    #     chunk = data_chunk.slice(i, chunk_size).to_pandas()
    #     chunks.append(chunk)

    # del data_chunk
    # gc.collect()

    # data_chunk = pd.concat(chunks, ignore_index=True)

    return data_chunk


def predict(type_name, test_features, FEATURES, co_matrix_list, oof_dict):
    model_type_path = model_path + type_name + '/'
    model_paths = [model_type_path + i for i in os.listdir(model_type_path) if '.txt' in i]
    model_paths = sorted(model_paths)

    # pred = test_features[['session', 'aid']].copy()
    result = np.zeros(len(test_features))
    for i, lgb_model in enumerate(model_paths):
        # print(f'正在使用第{i}/{len(model_paths)}个模型进行预测...')
        ranker = lgb.Booster(model_file=lgb_model)
        result += ranker.predict(test_features[FEATURES]) / 5

    return result



if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)
    
    for type_name in ['click', 'click_all', 'cart', 'order']:
        print(f"\n====== 正在处理行为类型: {type_name} ======")

        if type_name == 'click_all':
            output_file_path = f'{output_path}click_test_v3_all_target.parquet'
        else:
            output_file_path = f'{output_path}{type_name}_test_v3.parquet'
        if os.path.exists(output_file_path) : 
            print(f'{type_name}类型已预测完成.')
            continue
        
        print(f'正在构建测试数据特征...')
        # --- 1. 加载配置文件 ---
        print("正在加载配置文件...")
        FEATURES, co_matrix_list, oof_dict = config_load(type_name)
        print(f"加载完成，特征数量: {len(FEATURES)}")

        # --- 2. 准备特征数据 ---
        join_feature_path = datamart_path + 'test_' + type_name + '_datamart.parquet'
        print("正在加载待预测样本...")
        if type_name == 'click_all':
            candidate_file_path = candidate_path + 'test_' + type_name + '_candidate.parquet'
        else:
            candidate_file_path = candidate_path + 'test_' + type_name + 's_candidate.parquet'
        test_features = pl.read_parquet(candidate_file_path)
        test_features = test_features.fill_null(0)
        test_features = test_features.fill_nan(0)

        chunk_size = 20000000
        pred_result = []
        num_samples = test_features.shape[0]
        for i in tqdm(range(0, num_samples, chunk_size), desc='正在进行模型预测'):
            start_idx = i
            end_idx = min(i + chunk_size, num_samples)
            test_chunk = test_features[start_idx:end_idx]
            # print('正在合并特征...')
            test_chunk = join_features(test_chunk, type_name, oof_dict, co_matrix_list, FEATURES)
            test_chunk = test_chunk[['session', 'aid'] + FEATURES]
        
            # print(f'正在进行预测...')
            chunk_pred = predict(type_name, test_chunk, FEATURES, co_matrix_list, oof_dict)

            # 构建 DataFrame 包含预测分数
            chunk_result_df = test_chunk[['session', 'aid']].copy()
            chunk_result_df['pred'] = chunk_pred

            pred_result.append(chunk_result_df)

            del test_chunk, chunk_pred, chunk_result_df
            gc.collect()
        
        test_pred = pd.concat(pred_result, ignore_index=True)

        print(f'正在保存{type_name}预测结果...')
        if type_name != 'click_all':
            test_pred.to_parquet(f'{output_path}{type_name}_test_v3.parquet')
            print(f'结果已保存到{output_path}{type_name}_test_v3.parquet.')
        else:
            test_pred.to_parquet(f'{output_path}click_test_v3_all_target.parquet')
            print(f'结果已保存到{output_path}click_test_v3_all_target.parquet.')