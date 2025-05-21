import pandas as pd
import numpy as np
import os, gc
import cudf
from tqdm import tqdm

# # 定义文件路径
# evaluate_path = '../../data/evaluate_format/'
# oof_path = '../../data/oof/'

evaluate_path = '../../output/'
oof_path = '../../output/'

type2resultfile_v3 = {
    'click': 'click_test_v3.parquet',
    'cart': 'cart_test_v3.parquet',
    'order': 'order_test_v3.parquet'
}

type2resultfile_v4 = {
    'click': 'click_test_v4.parquet',
    'cart': 'cart_test_v4.parquet',
    'order': 'order_test_v4.parquet'
}


for type_name in ['click']:
    print(f'正在处理{type_name}')
    # 设置不同 type 的权重
    if type_name == 'click':
        w1 = 1
        w2 = 0
    elif type_name == 'cart':
        w1 = 0.4
        w2 = 0.6
    else:  # orders
        w1 = 0.3
        w2 = 0.7

    # 加载两个模型结果并标记来源
    result_v3_df = pd.read_parquet(os.path.join(oof_path, type2resultfile_v3[type_name]))[['session', 'aid', 'pred']]
    result_v3_df['model'] = 'v3'

    result_v4_df = pd.read_parquet(os.path.join(oof_path, type2resultfile_v4[type_name]))[['session', 'aid', 'pred']]
    result_v4_df['model'] = 'v4'

    # 合并
    combined_df = pd.concat([result_v3_df, result_v4_df], ignore_index=True)

    del result_v3_df, result_v4_df
    gc.collect()

    # 分块处理
    unique_sessions = combined_df['session'].unique()
    chunk_size = 100000
    session_chunks = [unique_sessions[i:i + chunk_size] for i in range(0, len(unique_sessions), chunk_size)]

    chunk_results = []

    for chunk in tqdm(session_chunks):
        # 提取子集并转换为 cudf
        chunk_df_pd = combined_df[combined_df['session'].isin(chunk)]
        chunk_df = cudf.DataFrame.from_pandas(chunk_df_pd)

        # 分别提取两个模型的数据
        chunk_v3 = chunk_df[chunk_df['model'] == 'v3'].rename(columns={'pred': 'pred_v3'})
        chunk_v4 = chunk_df[chunk_df['model'] == 'v4'].rename(columns={'pred': 'pred_v4'})

        # 合并两个模型的预测
        merged = chunk_v3.merge(chunk_v4, on=['session', 'aid'], how='inner')

        # 计算加权预测
        merged['pred'] = merged['pred_v3'] * w1 + merged['pred_v4'] * w2

        # 只保留必要列
        merged = merged[['session', 'aid', 'pred']]

        # 转回 pandas
        chunk_results.append(merged.to_pandas())

    del combined_df, chunk_df_pd, chunk_df, chunk_v3, chunk_v4, merged
    gc.collect()

    # 合并最终结果
    final_result_df = pd.concat(chunk_results, ignore_index=True)
    final_result_df.to_parquet(os.path.join(oof_path, f'{type_name}_test_combined-weight.parquet'))
    print(f'文件已经保存至{oof_path+type_name}_test_combined.parquet')
    del chunk_results, final_result_df
    gc.collect()
