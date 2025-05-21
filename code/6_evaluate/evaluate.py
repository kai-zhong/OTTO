import pandas as pd
import numpy as np
import argparse
import os, gc
import cudf

from utils import Logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description='评估')
parser.add_argument('--logfile', default='evaluate1.log')
parser.add_argument('--recomp', default='False')
parser.add_argument('--mversion', default='v3')
parser.add_argument('--k', default=20)

args = parser.parse_args()
logfile = args.logfile
recomp = args.recomp.lower() == 'true'
mVersion = args.mversion
K = args.k

# 初始化日志
os.makedirs('../../log', exist_ok=True)
log = Logger(f'../../log/{logfile}').logger

# 定义文件路径
# evaluate_path = '../../data/evaluate_format/'
# oof_path = '../../data/oof/'

# type2resultfile = {'clicks' : f'click_train_lgbm_{mVersion}.parquet',
#                 'carts' : f'cart_train_lgbm_{mVersion}.parquet', 
#                 'orders' : f'order_train_lgbm_{mVersion}.parquet'}

evaluate_path = '../../output/'
oof_path = '../../output/'

type2resultfile = {'clicks' : f'click_test_{mVersion}.parquet',
                'carts' : f'cart_test_{mVersion}.parquet', 
                'orders' : f'order_test_{mVersion}.parquet'}


def construct_eval_df(type_name, df, eval_format_file):
    unique_sessions = df['session'].unique()
    log.info(f'正在将{type_name}预测结果构建为用于评估的格式...')
    chunk_size = 100000
    eval_df = []
    for i in tqdm(range(0, len(unique_sessions), chunk_size)):
        chunk_sessions = unique_sessions[i:i+chunk_size]

        # 选出当前块中所有 session 对应的记录
        chunk_df = df[df["session"].isin(chunk_sessions)]
        chunk_df = cudf.DataFrame(chunk_df)
        chunk_df = (
            chunk_df.sort_values(by=["session", "pred"], ascending=[True, False])
            .groupby("session")
            .head(K)
        )
        # 再 groupby 一次，把结果变成 list
        chunk_df = (
            chunk_df.groupby("session")["aid"]
            .agg(list)
            .reset_index(name="pred_list")
        )

        eval_df.append(chunk_df)
    
    del df, unique_sessions
    gc.collect

    eval_df = cudf.concat(eval_df, ignore_index=True)
    eval_df = eval_df.to_pandas()

    log.info(f'正在保存{type_name}的评估格式预测结果...')
    eval_df.to_parquet(os.path.join(evaluate_path, eval_format_file))
    log.info(f'评估格式预测结果已保存至{os.path.join(evaluate_path, eval_format_file)}')
    return eval_df


def apk(actual, predicted):
    """
    计算 average precision at K
    """
    actual = list(actual)
    predicted = list(predicted)

    if len(actual) == 0:
        return 0.0
    if len(predicted) == 0:
        return 0.0

    predicted = predicted[:K]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), K)


def eval(type_name, eval_df):
    # 计算 MAP@20 和 R_type 所需的变量
    map_scores = []
    numerator_rtype = 0
    denominator_rtype = 0

    for _, row in tqdm(eval_df.iterrows(), total = len(eval_df)):
        gt = row["ground_truth"]
        pred = row["pred_list"]

        map_scores.append(apk(gt, pred))

        # R_type 分子和分母
        intersect = set(gt) & set(pred[:20])
        numerator_rtype += len(intersect)
        denominator_rtype += min(20, len(gt))

    # 计算最终指标
    map20 = np.mean(map_scores)
    rtype = numerator_rtype / denominator_rtype if denominator_rtype > 0 else 0.0

    log.info(f"{type_name} MAP@20: {map20:.6f}")
    log.info(f"{type_name} R_type: {rtype:.6f}")

    return rtype


if __name__ == '__main__':
    log.info(f'----正在评估{mVersion}版本模型...----')
    label_df = pd.read_parquet('../../data/train_valid/test_labels.parquet')
    rtype_list = []
    for type_name in ['clicks', 'carts', 'orders']:
        log.info(f"====== 正在评估行为类型: {type_name} ======")
        type_label_df = label_df[label_df['type'] == type_name]
        # eval_format_file = f'eval_{type_name}_{mVersion}_df.parquet'
        eval_format_file = f'output_{type_name}_{mVersion}_df.parquet'
        if recomp == False and os.path.exists(os.path.join(evaluate_path, eval_format_file)):
            result_eval_df = pd.read_parquet(os.path.join(evaluate_path, eval_format_file))
        else:
            result_eval_df = pd.read_parquet(os.path.join(oof_path, type2resultfile[type_name]))
            result_eval_df = construct_eval_df(type_name, result_eval_df, eval_format_file)
        
    #     log.info('正在 merge ground_truth 与预测结果...')
    #     log.info(f'拼接前预测结果session数 : {len(result_eval_df)}')
    #     result_eval_df = pd.merge(result_eval_df, type_label_df, on='session', how="inner")
    #     log.info(f'拼接后预测结果session数 : {len(result_eval_df)}')

    #     log.info(f'正在评估{type_name}预测结果的MAP@{K}与Rtype...')
    #     rtype = eval(type_name, result_eval_df)
    #     rtype_list.append(rtype)
    
    # sum_rtype_score = rtype_list[0] * 0.1 + rtype_list[1] * 0.3 + rtype_list[2] * 0.6
    # log.info(f'模型{mVersion} Rtype综合分数为 {sum_rtype_score}.')
