import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys, pickle, glob, gc
import cudf, itertools

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
output_path = '../../data/feature/'

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


def reload_to_gpu(df):
    df = cudf.DataFrame(df)
    return df


def compute_aid_day_session_ratio(test_actions, act_type, feature_name):
    """
    为每个商品（aid）计算其在每天（day）发生某种交互类型的session数占总发生交互的session数的比例.

    参数：
    test_actions (cudf.dataframe) : 包含用户行为数据的 cudf dataframe 测试集数据，
                                    [session, aid, ts, type, date, dow, day, hour]
    act_type (int) : 0表示click，1表示cart，2表示order，如果为None则无筛选条件，不区分交互类型进行占比计算
    feature_name (str) : 最后构造的ratio的列名

    返回:
    aid_day_session_ratio (cudf.dataframe) : 包含每个商品在每天【发生act_type类型交互的session数占总发生交互的session数的比例】的特征的dataframe.
                                            columns = [aid, day, feature_name]
    """
    # 仅计算act_type交互类型的占比, 否则不区分交互类型
    if act_type != None:
        test_actions = test_actions[test_actions['type'] == act_type]
    
    # 计算每一天有多少session发生过交互
    # [day, day_session_num]
    day_session_num = test_actions.groupby('day')['session'].nunique().reset_index() 
    day_session_num.columns = ['day', 'day_session_num']
    # 每个商品在每一天发生过交互的session数量
    # [aid, day, aid_day_session_num]
    aid_day_session_num = test_actions.groupby(['aid', 'day'])['session'].nunique().reset_index() 
    aid_day_session_num.columns = ['aid', 'day', 'aid_day_session_num']
    
    # [aid, day, aid_day_session_num, day_session_num]
    aid_day_session_ratio = aid_day_session_num.merge(day_session_num, on='day', how='left')
    # 计算每个商品在每一天发生交互的session占总发生交互的session的比例
    # [aid, day, aid_day_session_num, day_session_num, aid_day_session_ratio]
    aid_day_session_ratio[feature_name] = aid_day_session_ratio['aid_day_session_num']/aid_day_session_ratio['day_session_num']
    aid_day_session_ratio[feature_name] *= 10    # 放大权重，便于后续建模
    # [aid, day, aid_day_session_ratio]
    aid_day_session_ratio = aid_day_session_ratio[['aid', 'day', feature_name]]
    return aid_day_session_ratio


def gen_aid_day_features(prefix, test_actions):
    """
    为每个商品（aid）生成其在每天（day）维度上的交互特征，包括行为占比和唯一 session 比例。

    该函数的主要功能是统计每个商品在每天的交互情况，并生成四种特征：
        1. 商品在每一天发生交互的次数占所有交互行为数的比例（daily_aid_share）
        2. 商品在该天被 session 交互的比例（aid_day_inter_session_ratio）
        3. 商品在该天被 session 加入购物车的比例（aid_day_cart_session_ratio）
        4. 商品在该天被唯一 session 下单的比例（aid_day_order_session_ratio）

    所有特征的统计单位均为 aid（商品）在某一具体 day（天）下的行为表现。

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，要求至少包含以下列：
                                        [session, aid, ts, type]

    返回值:
        None
        文件输出: 一个 parquet 文件，文件名为 {prefix}aid_day_df.parquet，包含以下列:
            - 'day' (int): 一年中的第几天
            - 'aid' (int): 商品 ID
            - 'daily_aid_share' (float32): 商品在每一天所有行为中出现的占比（当天商品被交互次数/当天总共发生的所有商品交互次数）
            - 'aid_day_inter_session_ratio' (float32): 商品每日交互的session数占总发生交互的session数的比例
            - 'aid_day_cart_session_ratio' (float32): 商品每日cart的session数占总发生cart的session数的比例
            - 'aid_day_order_session_ratio' (float32): 商品每日order的session数占总发生order的session数的比例
    """
    # 将时间戳转换为 datetime 格式（单位为纳秒，ts 为秒，需要乘 1e9，且加上2小时偏移）
    # 加2小时偏移，是为了调整时区，不加两小时就是从22：00开始的数据了
    test_actions['date'] = cudf.to_datetime((test_actions['ts'] + 2 * 60 * 60) * 1e9)
    # 提取日期相关字段
    # columns = [session, aid, ts, type, date, dow, day, hour]
    test_actions['dow'] = test_actions['date'].dt.dayofweek    # 提取星期几（0=周一, ..., 6=周日）
    test_actions['day'] = test_actions['date'].dt.dayofyear    # 提取一年中的第几天
    test_actions['hour'] = test_actions['date'].dt.hour        # 提取小时（24小时制）

    # ========== 计算商品在每一天发生交互的次数占所有交互行为数的比例（当天商品被交互次数/当天总共发生的所有商品交互次数） ==========
    # 计算每天每个商品被交互的次数
    # [aid, day, day_n]
    aid_day_features = test_actions[['aid', 'day']].value_counts().reset_index()
    aid_day_features.columns = ['aid', 'day', 'day_n']
    # 计算每天发生的交互行为数量（所有商品合计）
    # [day, total_action]
    day_actions_num = test_actions[['day']].value_counts().reset_index()
    day_actions_num.columns = ['day', 'total_action'] 

    # 计算商品在某一天的行为占比（点击数 / 总行为数）
    # [aid, day, day_n, total_action]
    aid_day_features = aid_day_features.merge(day_actions_num, on='day', how='left')
    # [aid, day, day_n, total_action, daily_aid_share]
    aid_day_features['daily_aid_share'] = aid_day_features['day_n'] / aid_day_features['total_action']
    aid_day_features['daily_aid_share'] = aid_day_features['daily_aid_share'].astype(np.float32)
    # [day, aid, daily_aid_share]
    aid_day_features = aid_day_features[['day', 'aid', 'daily_aid_share']]


    # ========== 计算商品每日交互的 session 占总发生交互的session 的比例 ==========
    # [aid, day, aid_day_inter_session_ratio]
    aid_day_inter_session_ratio = compute_aid_day_session_ratio(test_actions, None, 'aid_day_inter_session_ratio')

    # ========== 计算商品每日cart的 session 占总发生cart的session 的比例 ==========
    # [aid, day, aid_day_cart_session_ratio]
    aid_day_cart_session_ratio = compute_aid_day_session_ratio(test_actions, 1, 'aid_day_cart_session_ratio')

    # ========== 计算商品每日order的 session 占总发生order的session 的比例 ==========
    # [aid, day, aid_day_order_session_ratio]
    aid_day_order_session_ratio = compute_aid_day_session_ratio(test_actions, 2, 'aid_day_order_session_ratio')

    # 合并所有特征
    # [day, aid, daily_aid_share, aid_day_inter_session_ratio, aid_day_cart_session_ratio, aid_day_order_session_ratio]
    aid_day_features = aid_day_features.merge(aid_day_inter_session_ratio, on=['aid', 'day'], how='left')
    aid_day_features = aid_day_features.merge(aid_day_cart_session_ratio, on=['aid', 'day'], how='left')
    aid_day_features = aid_day_features.merge(aid_day_order_session_ratio, on=['aid', 'day'], how='left')

    # 将 cudf 转为 pandas，并填补缺失值为 0
    aid_day_features = aid_day_features.to_pandas().fillna(0)

    # 强制类型转换为 float32，节省内存
    aid_day_features[['aid_day_inter_session_ratio', 'aid_day_cart_session_ratio', 'aid_day_order_session_ratio']] = aid_day_features[['aid_day_inter_session_ratio', 'aid_day_cart_session_ratio', 'aid_day_order_session_ratio']].astype(np.float32)
    
    # 保存为 parquet 文件
    print('正在保存商品daily特征文件...')
    aid_day_features.to_parquet(output_path + prefix + 'aid_day_df.parquet')
    print(f'特征文件已保存至{output_path + prefix}aid_day_df.parquet中。')

    del aid_day_features
    gc.collect()


def gen_session_feature(prefix, test_actions):
    """
    生成用户特征，可以分为三类：
        - 整个会话统计特征（交互量、各类行为率、转化率）
        - 最后1天内的局部统计特征（24小时内的交互量、行为率、转化率）
        - 其他
    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，要求至少包含以下列：
                                        [session, aid, ts, type]
    返回：
        None.保存文件为{prefix}_session_df.parquet',包含以下列：
                            ['session', 'aid', 'day', 'last_type', 'session_hour_last', 'session_dow_last', 
                            'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                            'lastday_all_counts', 'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 
                            'lastday_session_cart_cvr', 'lastday_session_order_cvr',
                            'nunique_aids', 'count_per_ts', 'count_per_aids', 'ts_per_length', 
                            ]
    """
    # 将时间戳转换为 datetime 格式（单位为纳秒，ts 为秒，需要乘 1e9，且加上2小时偏移）
    # 加2小时偏移，是为了调整时区，不加两小时就是从22：00开始的数据了
    test_actions['date'] = cudf.to_datetime((test_actions['ts'] + 2 * 60 * 60) * 1e9)
    # 提取日期相关字段
    # columns = [session, aid, ts, type, date, dow, day, hour]
    test_actions['dow'] = test_actions['date'].dt.dayofweek    # 提取星期几（0=周一, ..., 6=周日）
    test_actions['day'] = test_actions['date'].dt.dayofyear    # 提取一年中的第几天
    test_actions['hour'] = test_actions['date'].dt.hour        # 提取小时（24小时制）


    # ========== 获取每个会话最后一次交互的时间（小时和星期几） ==========
    # 获取每个session的最后一条交互样本（因为test_actions是按时间排序过的，所以最后一条样本就是session的最后一次交互）
    # ['session', 'aid', 'day', 'session_hour_last', 'session_dow_last']
    session_last_time = test_actions.groupby('session')[['aid', 'day', 'hour', 'dow']].last().reset_index()
    session_last_time.columns = ['session', 'aid', 'day', 'session_hour_last', 'session_dow_last']


    # ========== 获取每个会话的最后一次行为类型 ==========
    # ['session', 'last_type']
    session_last_type = test_actions.groupby('session').last().reset_index()[['session', 'type']]
    session_last_type.columns = ['session', 'last_type']


    # ========== 计算每个session的总交互次数、点击率、加入购物车率、下单率、点击加购转化率、加购购买转化率 ==========
    # 计算每个会话的总交互次数
    # ['session', 'all_counts']
    session_df = test_actions['session'].value_counts().reset_index()
    session_df.columns = ['session', 'all_counts']
    # 计算每个会话的点击次数
    # [session, click_counts]
    click_num =  test_actions[test_actions['type'] == 0][['session']].value_counts().reset_index().rename(columns={'count': 'click_counts'})
    # 计算每个会话的加购次数
    # [session, cart_counts]
    cart_num =  test_actions[test_actions['type'] == 1][['session']].value_counts().reset_index().rename(columns={'count': 'cart_counts'})
    # 计算每个会话的下单次数
    # [session, order_counts]
    order_num =  test_actions[test_actions['type'] == 2][['session']].value_counts().reset_index().rename(columns={'count': 'order_counts'})

    # 合并会话的总交互次数和每种行为的交互次数
    # [session, all_counts, click_counts, cart_counts, order_counts]
    session_df = session_df.merge(click_num, on = 'session', how = 'left')
    session_df = session_df.merge(cart_num, on = 'session', how = 'left')
    session_df = session_df.merge(order_num, on = 'session', how = 'left')
    session_df = session_df.fillna(0)

    # 计算点击率、加入购物车率、下单率、点击加购转化率、加购购买转化率
    # [session, all_counts, click_counts, cart_counts, order_counts, 
    #   click_ratio, cart_ratio, order_ratio, session_cart_cvr, session_order_cvr]
    session_df['click_ratio'] = session_df['click_counts'] / session_df['all_counts']
    session_df['cart_ratio'] = session_df['cart_counts'] / session_df['all_counts']
    session_df['order_ratio'] = session_df['order_counts'] / session_df['all_counts']
    session_df['session_cart_cvr'] = session_df['cart_counts'] / session_df['click_counts']
    session_df['session_order_cvr'] = session_df['order_counts'] / session_df['cart_counts']
    session_df = session_df.fillna(0)

    # 只保留相关列
    # [session, all_counts, click_ratio, cart_ratio, order_ratio, session_cart_cvr, session_order_cvr]
    session_df = session_df[['session', 'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 
                             'session_cart_cvr', 'session_order_cvr']]
    del click_num, cart_num, order_num
    gc.collect()


    # ========== 计算每个session最后一次交互前1天内的总交互次数、点击率、加入购物车率、下单率、点击加购转化率、加购购买转化率 ==========
    # 获取每个session最后一次交互的时间戳
    # [session, max_ts]
    session_max_ts = test_actions.groupby('session')['ts'].max().reset_index()
    session_max_ts.columns = ['session', 'max_ts']
    # 获取每个session最后一次交互往前一天（24小时前）的时间戳
    # [session, max_ts, day_ts]
    session_max_ts['day_ts'] = session_max_ts['max_ts'] - (24 * 60 * 60)

    # 计算每个session最后一次交互前1天内的交互记录
    # [session, aid, ts, type, date, dow, day, hour, max_ts, day_ts]
    session_lastday_inter = test_actions.merge(session_max_ts, on = 'session', how = 'left')
    session_lastday_inter = session_lastday_inter[session_lastday_inter['ts'] >= session_lastday_inter['day_ts']]

    # 计算每个session最后一次交互前1天内的总交互数量
    # ['session', 'lastday_all_counts']
    session_lastday_df = session_lastday_inter['session'].value_counts().reset_index()
    session_lastday_df.columns = ['session', 'lastday_all_counts']

    # 计算每个session最后一次交互前1天内的点击总数
    # [session, lastday_click_counts]
    click_num =  session_lastday_inter[session_lastday_inter['type'] == 0][['session']].value_counts().reset_index().rename(columns={'count': 'lastday_click_counts'})
    # 计算每个session最后一次交互前1天内的cart总数
    # [session, lastday_cart_counts]
    cart_num =  session_lastday_inter[session_lastday_inter['type'] == 1][['session']].value_counts().reset_index().rename(columns={'count': 'lastday_cart_counts'})
    # 计算每个session最后一次交互前1天内的order总数
    # [session, lastday_order_counts]
    order_num =  session_lastday_inter[session_lastday_inter['type'] == 2][['session']].value_counts().reset_index().rename(columns={'count': 'lastday_order_counts'})

    # 合并过去1天内的交互次数
    session_lastday_df = session_lastday_df.merge(click_num, on = 'session', how = 'left')
    session_lastday_df = session_lastday_df.merge(cart_num, on = 'session', how = 'left')
    session_lastday_df = session_lastday_df.merge(order_num, on = 'session', how = 'left')
    session_lastday_df = session_lastday_df.fillna(0)

    # 计算过去1天的点击率、加入购物车率、下单率以及转化率
    # ['session', 'lastday_all_counts', 'lastday_click_ratio', 
    #   'lastday_cart_ratio', 'lastday_order_ratio', 
    #   'lastday_session_cart_cvr', 'lastday_session_order_cvr']
    session_lastday_df['lastday_click_ratio'] = session_lastday_df['lastday_click_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_cart_ratio'] = session_lastday_df['lastday_cart_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_order_ratio'] = session_lastday_df['lastday_order_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_session_cart_cvr'] = session_lastday_df['lastday_cart_counts'] / session_lastday_df['lastday_click_counts']
    session_lastday_df['lastday_session_order_cvr'] = session_lastday_df['lastday_order_counts'] / session_lastday_df['lastday_cart_counts']
    session_lastday_df = session_lastday_df[['session', 'lastday_all_counts', 'lastday_click_ratio', 
                                             'lastday_cart_ratio', 'lastday_order_ratio', 
                                             'lastday_session_cart_cvr', 'lastday_session_order_cvr']]
    session_lastday_df = session_lastday_df.fillna(0)
    del session_lastday_inter, click_num, cart_num, order_num
    gc.collect()


    # ========== 计算每个session交互过的不同的商品数量、session的时间跨度（单位：小时）、时间戳的唯一值数（类似于交互次数）、最后两次交互的时间间隔（单位：小时）、相邻交互之间的时间间隔的均值，取log（单位：log(秒)）==========
    # 计算每个会话交互过的不同的商品数量
    # ['session', 'nunique_aids']
    session_nunique_df = test_actions.groupby(['session'])['aid'].nunique().reset_index()
    session_nunique_df.columns = ['session', 'nunique_aids']

    # 计算每个会话从交互第一个商品到交互最后一个商品之间的时间长度（单位：小时）
    # ['session', 'ts_length']
    session_ts_length = test_actions.groupby('session').agg({'ts': ['min', 'max']}).reset_index()
    session_ts_length.columns = ['session', 'ts_min', 'ts_max']
    session_ts_length['ts_length'] = (session_ts_length['ts_max'] - session_ts_length['ts_min']) / 3600
    session_ts_length = session_ts_length[['session', 'ts_length']]

    # 计算每个会话内时间戳的唯一值数(我感觉类似于计算每个session的总交互次数)
    # ['session', 'ts_nunique']
    session_ts_nunique = test_actions.groupby('session')['ts'].nunique().reset_index()
    session_ts_nunique.columns = ['session', 'ts_nunique']

    # 计算每个会话最后两次交互的时间间隔（单位：小时）
    # ['session', 'session_last_diff']
    session_ts_unique = test_actions[['session', 'ts']].drop_duplicates()
    session_ts_unique['ts_diff'] = session_ts_unique.groupby('session')['ts'].diff()
    session_last_diff = (session_ts_unique.groupby('session')['ts_diff'].last() / 3600).reset_index()
    session_last_diff.columns = ['session', 'session_last_diff']
    session_last_diff = session_last_diff.fillna(0)

    # 计算每个会话相邻交互之间的时间间隔的均值，取log（单位：log(秒)）
    # 交互时间间隔这种数据：有些用户可能几秒点一次；有些用户可能几小时点一次；
    # 直接拿来用的话，极端大的值会拉偏模型。 所以一般会对这种跨度很大的特征取 log，让它分布更集中、更稳定。
    # ['session', 'ts_diff_mean']
    session_ts_unique = session_ts_unique.groupby('session')['ts_diff'].mean().reset_index()
    session_ts_unique['ts_diff_mean'] = cudf.DataFrame(np.log1p(session_ts_unique.to_pandas().fillna(0)['ts_diff']))
    session_ts_unique = session_ts_unique[['session', 'ts_diff_mean']]

    # 合并所有计算得到的特征
    # ['session', 'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr', 
    # 'lastday_all_counts', 'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 'lastday_session_cart_cvr', 'lastday_session_order_cvr', 
    # 'nunique_aids', 'ts_length', 'ts_nunique', 'session_last_diff', 
    # 'aid', 'day', 'session_hour_last', 'session_dow_last', 'last_type']
    session_df = session_df.merge(session_lastday_df, on = 'session', how = 'left')
    session_df = session_df.merge(session_nunique_df, on = 'session', how = 'left')
    session_df = session_df.merge(session_ts_length, on = 'session', how = 'left')
    session_df = session_df.merge(session_ts_nunique, on = 'session', how = 'left')
    session_df = session_df.merge(session_last_diff, on = 'session', how = 'left')
    session_df = session_df.merge(session_last_time, on = 'session', how = 'left')
    session_df = session_df.merge(session_last_type, on = 'session', how = 'left')

    # ['session', 'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr', 
    # 'lastday_all_counts', 'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 'lastday_session_cart_cvr', 'lastday_session_order_cvr', 
    # 'nunique_aids', 'ts_length', 'ts_nunique', 'session_last_diff', 
    # 'aid', 'day', 'session_hour_last', 'session_dow_last', 'last_type',
    # 'count_per_ts', 'count_per_aids', 'ts_per_length']
    # 计算每个 session 中，平均每次交互之间的间隔时间
    session_df['count_per_ts'] = session_df['ts_length'] / session_df['all_counts']
    # 计算每个 session 中，不同 aid 的占比（交互的商品种类数/总交互数）（越大说明交互商品越丰富）
    session_df['count_per_aids'] = session_df['nunique_aids'] / session_df['all_counts']
    # 计算每个 session 中，相邻两次不同时间戳平均间隔（单位：小时）
    session_df['ts_per_length'] = session_df['ts_length'] / session_df['ts_nunique']
    
    session_df = session_df.to_pandas().fillna(0)

    # 转换为float32类型以节省内存
    float32_cols = ['click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                    'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 
                    'lastday_session_cart_cvr', 'lastday_session_order_cvr',
                    'count_per_ts', 'count_per_aids', 'ts_per_length']
    session_df[float32_cols] = session_df[float32_cols].astype(np.float32)

    # 保留最终特征列并保存为parquet文件
    # session_df = session_df[['session', 'aid', 'day', 'last_type', 'session_hour_last', 'session_dow_last', 
                            # 'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                            # 'lastday_all_counts', 'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 
                            # 'lastday_session_cart_cvr', 'lastday_session_order_cvr',
                            # 'nunique_aids', 'count_per_ts', 'count_per_aids', 'ts_per_length', 
                            # ]]
    session_df = session_df[['session', 'aid', 'day', 'last_type', 'session_hour_last', 'session_dow_last', 
                            'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                            'lastday_all_counts', 'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 
                            'lastday_session_cart_cvr', 'lastday_session_order_cvr',
                            'nunique_aids', 'count_per_ts', 'count_per_aids', 'ts_per_length', 
                            ]]
    
    print('正在保存session特征文件...')
    session_df.to_parquet(output_path + prefix + 'session_df.parquet')
    print(f'{prefix}session特征文件已经保存到{output_path+prefix}session_df.parquet中.')


def gen_aid_feature(prefix, merge_actions, test_actions):
    """
    生成与aid相关的特征数据，并保存为parquet文件。
    该函数基于用户行为数据（点击、加购、下单）生成与aid相关的多种特征，包括转化率、跳跃比率、时间差、重复行为比率等。
    特征生成后以parquet格式保存到指定路径。
    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，包含以下列：
                                        [session, aid, ts, type]
        merge_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集+训练集数据，包含以下列：
                                        [session, aid, ts, type]

    返回值:
        None: 函数无返回值，直接将生成的特征数据保存到指定路径。
    """
    # ========== 1. 初始化以aid为主键的特征表 ==========
    # 初始化以aid为主键的特征表，包含所有出现过的aid，作为aid list
    # columns = [aid]
    aid_features = merge_actions[['aid']].drop_duplicates().reset_index(drop=True)


    # ========== 2. 计算每个被点击商品的历史click-cart转化率和历史click-order转化率 ==========
    # 包括：
    # 1. cart_cvr_df['cart_hist_cvr'] : 点击商品A之后进行加购的session占点击商品A的session的比例
    # 2. cart_cvr_df['order_hist_cvr'] : 点击商品A之后进行下单的session占点击商品A的session的比例
    # 分块处理
    chunk = 7000    # 每块chunk的行数
    cart_cvr_df = []
    chunk_num = int(len(merge_actions['aid'].drop_duplicates()) / chunk) + 1
    print(f"正在计算历史会话转化率...")
    # 按aid分块计算每个aid的点击、加购、下单的转化率
    for i in tqdm(range(chunk_num)):
        start = i * chunk
        end = (i + 1) * chunk

        # 筛选aid号在start和end之间的被点击商品
        # [session, aid, ts, type]
        row_click = merge_actions[(merge_actions['aid'] >= start) & (merge_actions['aid'] < end) & (merge_actions['type'] == 0)]
        # 将存在于相同session的click商品和cart商品对应起来进行关联
        # [session, aid_x(click商品), ts_x, type_x, aid_y(cart商品), ts_y, type_y]
        row_cart = row_click.merge(merge_actions[merge_actions['type'] == 1], on='session', how='inner')
        # 将存在于相同session的click商品和order商品对应起来进行关联
        # [session, aid_x(click商品), ts_x, type_x, aid_y(order商品), ts_y, type_y]
        row_order = row_click.merge(merge_actions[merge_actions['type'] == 2], on='session', how='inner')

        # 统计每个 aid(click商品) 被多少个session点击过
        # ['aid', 'click_n']
        click_all = row_click[['aid', 'session']].drop_duplicates()['aid'].value_counts().reset_index()
        click_all.columns = ['aid', 'click_n']
        # 统计aid_x（cart商品）被多少个存在点击行为的session加购过
        # ['aid', 'cart_n']
        click_cart = row_cart[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()
        click_cart.columns = ['aid', 'cart_n']
        # 统计aid_x（order商品）被多少个存在点击行为的session下单过
        # ['aid', 'order_n']
        click_order = row_order[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()
        click_order.columns = ['aid', 'order_n']

        # 合并点击、加购、下单次数
        # [aid, click_n, cart_n, order_n]
        click_all = click_all.merge(click_cart, on='aid', how='left')
        click_all = click_all.merge(click_order, on='aid', how='left')
        click_all = click_all.fillna(0)

        # 计算点击到加购的转化率
        # ['aid', 'click_n', 'cart_n', 'order_n', 'cart_hist_cvr']
        click_all['cart_hist_cvr'] = click_all['cart_n'] / click_all['click_n']
        # 计算点击到下单的转化率
        # ['aid', 'click_n', 'cart_n', 'order_n', 'cart_hist_cvr', 'order_hist_cvr']
        click_all['order_hist_cvr'] = click_all['order_n'] / click_all['click_n']

        # 保存本块结果
        cart_cvr_df.append(click_all)

    # 合并所有块的转化率特征
    # ['aid', 'cart_hist_cvr', 'order_hist_cvr']
    cart_cvr_df = cudf.concat(cart_cvr_df)
    cart_cvr_df = cart_cvr_df[['aid', 'cart_hist_cvr', 'order_hist_cvr']].reset_index(drop=True)
    cart_cvr_df = cart_cvr_df.to_pandas()

    del click_all, click_cart, click_order, row_click, row_cart, row_order
    gc.collect()



    # ========== 3. 计算交互转换比例和时间差 ==========
    # 与商品A交互后，session紧接着与商品B进行交互，则我们称（商品A->商品B）是一组转换，记为商品B的一次转换
    # 包括：
    # 1. type_counts['aid_{action_type_src}_{action_type_dst}_ratio'](action_type∈{click, cart, order}) :
    #             转换action_type_src->action_type_dst占总转换数的比例.(共9个)
    # 2. type_counts['cvr_sum'] : 商品A的总转换数
    # 3. type_counts['cvr_sum_share'] : 商品A的转换数占所有商品的转换总数的比例
    # 4. type_ts_diff['aid_click_click_diffts'] : 两个商品的交互类型分别为click->click的转换的转换间隔的平均值
    # 5. type_ts_diff['aid_click_cart_diffts'] : 两个商品的交互类型分别为click->cart的转换的转换间隔的平均值
    # 6. type_ts_diff['aid_cart_click_diffts'] : 两个商品的交互类型分别为cart->click的转换的转换间隔的平均值
    # 7. type_ts_diff['aid_cart_order_diffts'] : 两个商品的交互类型分别为cart->order的转换的转换间隔的平均值
    print("计算事件转换比例与时间差...")
    # --- 3.1 准备数据 ---
    # 按 session 和 aid 分组，获取同组内上一次交互的时间戳和类型
    # [session, aid, ts, type, before_ts, before_type]
    merge_actions['before_ts'] = merge_actions.groupby(['session', 'aid'])['ts'].shift()
    merge_actions['before_type'] = merge_actions.groupby(['session', 'aid'])['type'].shift()
    
    # 筛选掉每个 session-aid 组的第一个事件 (因为没有 'before_type'/'before_ts')
    # [session, aid, ts, type, before_ts, before_type]
    merge_ts = merge_actions[merge_actions['before_type'] >= 0].copy()

    # 创建表示交互类型转换的字符串 (例如 '1_0' 代表从 click(0) 到 cart(1))
    # [session, aid, ts, type, before_ts, before_type, type_cvr]
    merge_ts['type_cvr'] = merge_ts['type'].astype('str') + '_' + merge_ts['before_type'].astype('str')
    
    # 计算当前交互与上一次交互之间的时间差（单位：小时）
    # [session, aid, ts, type, before_ts, before_type, type_cvr, diff_ts]
    merge_ts['diff_ts'] = (merge_ts['ts'] - merge_ts['before_ts']) / 3600

    # --- 3.2 计算事件转换比例 ---
    # 按 aid 统计每种交互类型转换 ('type_cvr') 的发生次数
    # [aid, type_cvr, n]
    type_counts = merge_ts[['aid', 'type_cvr']].value_counts().reset_index()
    type_counts.columns = ['aid', 'type_cvr', 'n'] # 重命名列

    # 将转换类型计数进行透视，使每种转换类型成为单独的列
    # [aid, 0_0, 0_1, 0_2, 1_0, 1_1, 1_2, 2_0, 2_1, 2_2]
    type_counts = type_counts.pivot_table(index = ['aid'], columns = ['type_cvr'], values = ['n']).reset_index()

    # 重命名透视后的列名，使其更具描述性
    # 列名格式：aid_{当前事件类型}_{前一事件类型}_ratio
    # ['aid', 'aid_click_click_ratio', 'aid_cart_click_ratio', 'aid_order_click_ratio', 
    #           'aid_click_cart_ratio', 'aid_cart_cart_ratio', 'aid_order_cart_ratio', 
    #           'aid_click_order_ratio', 'aid_cart_order_ratio', 'aid_order_order_ratio']
    type_counts.columns = ['aid',
                           'aid_click_click_ratio', # 0_0: click -> click 比例
                           'aid_cart_click_ratio',  # 1_0: click -> cart 比例
                           'aid_order_click_ratio', # 2_0: click -> order 比例
                           'aid_click_cart_ratio',  # 0_1: cart -> click 比例
                           'aid_cart_cart_ratio',   # 1_1: cart -> cart 比例
                           'aid_order_cart_ratio',  # 2_1: cart -> order 比例
                           'aid_click_order_ratio', # 0_2: order -> click 比例
                           'aid_cart_order_ratio',  # 1_2: order -> cart 比例
                           'aid_order_order_ratio']# 2_2: order -> order 比例
    
    # 计算每个 aid 的总转换次数 (所有转换类型的次数之和)
    type_counts['cvr_sum'] = type_counts.iloc[:,1:].sum(1)
    # 将各种转换类型的计数归一化，计算其占总转换次数的比例
    # ['aid', 'aid_click_click_ratio', 'aid_cart_click_ratio', 'aid_order_click_ratio', 
    #           'aid_click_cart_ratio', 'aid_cart_cart_ratio', 'aid_order_cart_ratio', 
    #           'aid_click_order_ratio', 'aid_cart_order_ratio', 'aid_order_order_ratio',
    #           'cvr_sum']
    for i in list(type_counts.columns)[1:10]: # 遍历 9 种转换比例列
        type_counts[i] = type_counts[i] / type_counts['cvr_sum']
    
    # 转换为 cuDF DataFrame 并填充 NaN 为 0 (如果某个 aid 没有某种转换，透视后会是 NaN)
    type_counts = type_counts.fillna(0)

    # 计算每个 aid 的转换次数占所有 aid 总转换次数的比例 (衡量该 aid 的活跃度)
    # ['aid', 'aid_click_click_ratio', 'aid_cart_click_ratio', 'aid_order_click_ratio', 
    #           'aid_click_cart_ratio', 'aid_cart_cart_ratio', 'aid_order_cart_ratio', 
    #           'aid_click_order_ratio', 'aid_cart_order_ratio', 'aid_order_order_ratio'
    #           'cvr_sum', 'cvr_sum_share']
    type_counts['cvr_sum_share'] = type_counts['cvr_sum'] / type_counts['cvr_sum'].sum()

    # --- 3.3 计算事件转换的平均时间差 ---
    # 按 aid 和转换类型 ('type_cvr') 分组，计算每组转换间隔的平均时间差 ('diff_ts')。
    # [aid, type_cvr, diff_ts]
    type_ts_diff = merge_ts.groupby(['aid', 'type_cvr'])['diff_ts'].mean().reset_index()
    # 同样使用 pivot_table 将转换类型变为列。
    # [aid, 0_0, 0_1, 0_2, 1_0, 1_1, 1_2, 2_0, 2_1, 2_2]
    # 列名格式：aid_{当前事件类型}_{前一事件类型}_diffts
    # ['aid', 'aid_click_click_diffts', 'aid_cart_click_diffts', 'aid_order_click_diffts', 
    #           'aid_click_cart_diffts', 'aid_cart_cart_diffts', 'aid_order_cart_diffts', 
    #           'aid_click_order_diffts', 'aid_cart_order_diffts', 'aid_order_order_diffts']
    type_ts_diff = type_ts_diff.pivot_table(index = ['aid'], columns = ['type_cvr'], values = ['diff_ts']).reset_index()
    type_ts_diff.columns = ['aid',
                           'aid_click_click_diffts',    # 0_0: click -> click 平均时间差
                           'aid_cart_click_diffts',     # 1_0: click -> cart 平均时间差
                           'aid_order_click_diffts',    # 2_0: click -> order 平均时间差
                           'aid_click_cart_diffts',     # 0_1: cart -> click 平均时间差
                           'aid_cart_cart_diffts',      # 1_1: cart -> cart 平均时间差
                           'aid_order_cart_diffts',     # 2_1: cart -> order 平均时间差
                           'aid_click_order_diffts',    # 0_2: order -> click 平均时间差
                           'aid_cart_order_diffts',     # 1_2: order -> cart 平均时间差
                           'aid_order_order_diffts']    # 2_2: order -> order 平均时间差

    # 仅选取部分可能比较重要的时间差特征进行保留。
    # 包括 click-click、click-cart、cart-click、cart-order的时间差。
    # ['aid', 'aid_click_click_diffts', 'aid_click_cart_diffts', 'aid_cart_click_diffts', 'aid_cart_order_diffts']
    type_ts_diff = type_ts_diff[['aid', 'aid_click_click_diffts', 'aid_click_cart_diffts',
                                 'aid_cart_click_diffts', 'aid_cart_order_diffts']]

    type_counts = type_counts.to_pandas()
    type_ts_diff = type_ts_diff.to_pandas()

    del merge_actions['before_ts'], merge_actions['before_type']
    del merge_ts
    gc.collect()



    # ========== 4. 计算商品跳过（Skip）行为的比例 ==========
    # 1. cart_df['cart_click_skip_ratio'] : 计算商品没有经过点击直接被加购占该商品总共被加购次数的比例
    # 2. order_df['order_click_skip_ratio'] : 计算商品没有经过点击直接被购买占该商品总共被购买次数的比例
    # 3. order_df['order_cart_skip_ratio'] : 计算商品没有经过加购直接被购买占该商品总共被购买次数的比例

    # --- 4.1 加购跳过点击 (Cart skip Click) ---
    print("正在计算cart skip click比例...")

    # 计算每个 aid 发生加购事件 (type == 1) 的唯一会话数。这是计算比例的分母。
    # [aid, cart_unique_num]
    cart_df = merge_actions[merge_actions['type'] == 1][['session', 'aid']].drop_duplicates()['aid'].value_counts().reset_index()
    cart_df.columns = ['aid', 'cart_unique_num'] # 重命名列

    # 获取所有包含加购事件的会话 ID 列表。
    cart_session = list(merge_actions[merge_actions['type'] == 1]['session'].unique().to_pandas())
    # 筛选出这些会话的所有记录。
    # [session, aid, ts, type]
    merge_cart = merge_actions[merge_actions['session'].isin(cart_session)]
    # 再次进行透视操作，统计每个 session-aid 对下各类型事件的次数。
    # ['session', 'aid', 'count_click', 'count_cart', 'count_order']
    merge_pivot = merge_cart.pivot_table(index=['session', 'aid'], columns=['type'], aggfunc='count').reset_index()
    merge_pivot.columns = ['session', 'aid', 'count_click', 'count_cart', 'count_order'] # 重命名列
    merge_pivot = merge_pivot.fillna(0)

    # 筛选出满足条件 `count_cart - count_click > 0` 的记录。
    # 比如session1中加购商品A的次数大于点击商品A的次数
    # 这意味着在同一个 session-aid 对中，加购次数比点击次数多。
    # 然后统计满足此条件的 aid 及其出现的次数 (value_counts)。
    # ['aid', 'cart_click_skip_num']
    skip_aid_df = merge_pivot[(merge_pivot['count_cart'] -  merge_pivot['count_click']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df.columns = ['aid', 'cart_click_skip_num'] # 重命名列：aid, "cart-skip-click"的会话数

    # 将 "跳过" 计数左连接到包含唯一加购会话数的 cart_df。
    # [aid, cart_unique_num, cart_click_skip_num]
    cart_df = cart_df.merge(skip_aid_df, on = ['aid'], how = 'left')
    # 转换回 cuDF 并填充 NaN 为 0 (如果某 aid 从未出现跳过情况，skip_num 会是 NaN)。
    cart_df = cart_df.fillna(0)
    # 计算“加购跳过点击”的比例。(跳过会话数) / (总加购会话数)。
    # ['aid', 'cart_click_skip_ratio']
    cart_df['cart_click_skip_ratio'] = cart_df['cart_click_skip_num'] / cart_df['cart_unique_num']
    cart_df = cart_df[['aid', 'cart_click_skip_ratio']]
    del merge_pivot, merge_cart, cart_session, skip_aid_df
    gc.collect()

    # --- 4.2 下单跳过点击 / 跳过加购 (Order skip Click / Cart) ---
    print("计算order skip Click/Cart 的比例...")
    # 计算每个 aid 发生下单事件 (type == 2) 的唯一会话数。
    # ['aid', 'order_unique_num']
    order_df = merge_actions[merge_actions['type'] == 2][['session', 'aid']].drop_duplicates()['aid'].value_counts().reset_index()
    order_df.columns = ['aid', 'order_unique_num']

    # 获取所有包含下单事件的会话 ID 列表。
    order_session = list(merge_actions[merge_actions['type'] == 2]['session'].unique().to_pandas())
    # 筛选出这些会话的所有记录。
    # [session, aid, ts, type]
    merge_order = merge_actions[merge_actions['session'].isin(order_session)]
    # 进行透视操作。
    # ['session', 'aid', 'count_click', 'count_cart', 'count_order']
    merge_pivot = merge_order.pivot_table(index=['session', 'aid'], columns=['type'], aggfunc='count').reset_index()
    merge_pivot.columns = ['session', 'aid', 'count_click', 'count_cart', 'count_order'] # 重命名列
    merge_pivot = merge_pivot.fillna(0)

    # 筛选出满足条件 `count_order - count_click > 0` 的记录 (下单次数 > 点击次数)。
    # 统计满足此条件的 aid 及其出现次数。
    # ['aid', 'order_click_skip_num']
    skip_aid_df_1 = merge_pivot[(merge_pivot['count_order'] -  merge_pivot['count_click']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df_1.columns = ['aid', 'order_click_skip_num'] # 重命名列：aid, "下单跳过点击"的会话数

    # 筛选出满足条件 `count_order - count_cart > 0` 的记录 (下单次数 > 加购次数)。
    # 统计满足此条件的 aid 及其出现次数。
    skip_aid_df_2 = merge_pivot[(merge_pivot['count_order'] -  merge_pivot['count_cart']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df_2.columns = ['aid', 'order_cart_skip_num'] # 重命名列：aid, "下单跳过加购"的会话数

    # 将两种 "跳过" 计数左连接到包含唯一下单会话数的 order_df。
    # [aid, order_unique_num, order_click_skip_num, order_cart_skip_num]
    order_df = order_df.merge(skip_aid_df_1, on = ['aid'], how = 'left')
    order_df = order_df.merge(skip_aid_df_2, on = ['aid'], how = 'left')
    order_df = order_df.fillna(0)
    # 计算“下单跳过点击”的比例。
    # [aid, order_unique_num, order_click_skip_num, order_cart_skip_num, order_click_skip_ratio]
    order_df['order_click_skip_ratio'] = order_df['order_click_skip_num'] / order_df['order_unique_num']
    # 计算“下单跳过加购”的比例。
    # [aid, order_unique_num, order_click_skip_num, order_cart_skip_num, order_click_skip_ratio, order_cart_skip_ratio]
    order_df['order_cart_skip_ratio'] = order_df['order_cart_skip_num'] / order_df['order_unique_num']
    # 选取最终需要的列: aid, 下单跳过加购比例, 下单跳过点击比例。
    # ['aid', 'order_cart_skip_ratio', 'order_click_skip_ratio']
    order_df = order_df[['aid', 'order_cart_skip_ratio', 'order_click_skip_ratio']]

    cart_df = cart_df.to_pandas()
    order_df = order_df.to_pandas()
    del merge_pivot, merge_order, order_session, skip_aid_df_1, skip_aid_df_2
    gc.collect()



    # ========== 5. 计算基于唯一会话数的特征 ==========
    # 这部分计算了：
    # 1. 'click_cvr_unique' : 点击商品A的session数占总session数的比例
    # 2. 'cart_cvr_unique'  : 对商品A进行加购的session占点击商品A的session的比例
    # 3. 'order_cvr_unique' : 对商品A进行下单的session占加购商品A的session的比例
    # 4. 'click_order_cvr_unique' : 对商品A进行下单的session占点击商品A的session的比例
    print("计算基于唯一会话数的特征...")

    # 按 aid 分组，计算点击事件 (type == 0) 的唯一会话数 (nunique)。(aid被多少session点击过)
    # ['aid', 'click_session']
    unique_click_session = merge_actions[merge_actions['type'] == 0].groupby(['aid'])['session'].nunique().reset_index()
    unique_click_session.columns = ['aid', 'click_session']
    # 按 aid 分组，计算加购事件 (type == 1) 的唯一会话数。(aid被多少session加购过)
    # ['aid', 'cart_session']
    unique_cart_session = merge_actions[merge_actions['type'] == 1].groupby(['aid'])['session'].nunique().reset_index()
    unique_cart_session.columns = ['aid', 'cart_session']
    # 按 aid 分组，计算下单事件 (type == 2) 的唯一会话数。(aid被多少session下单过)
    # ['aid', 'order_session']
    unique_order_session = merge_actions[merge_actions['type'] == 2].groupby(['aid'])['session'].nunique().reset_index()
    unique_order_session.columns = ['aid', 'order_session']

    # 将三种事件类型的唯一会话数统计结果左连接合并。
    # ['aid', 'click_session', 'cart_session', 'order_session']
    unique_session = unique_click_session.merge(unique_cart_session, on = 'aid', how = 'left')
    unique_session = unique_session.merge(unique_order_session, on = 'aid', how = 'left')

    # 计算该 aid 的点击会话数占总会话数的比例。
    # *注意*: 原本这个分母 `len(merge_actions['aid'].unique())` 是总 aid 数，而不是总 session 数。
    #         所以 `click_cvr_unique` 的含义更接近于“该 aid 被多少比例的独立会话触达过”，
    #         而不是传统意义上的转化率。它的值可能大于 1。
    #         如果意图是计算占总会话数的比例，分母应为 `len(merge['session'].unique())`。
    # total_unique_aids = len(merge_actions['aid'].unique()) # 获取总的唯一 aid 数量
    # unique_session['click_cvr_unique'] = unique_session['click_session'] / len(merge_actions['aid'].unique())
    unique_session['click_cvr_unique'] = unique_session['click_session'] / len(merge_actions['session'].unique())

    # 计算基于唯一会话数的点击到加购转化率 (Unique Cart CVR)。
    # 有多少比例的session对这个商品在点击后加购(发生加购的独立会话数) / (发生点击的独立会话数)。
    unique_session['cart_cvr_unique'] = unique_session['cart_session'] / unique_session['click_session']
    # 计算基于唯一会话数的加购到下单转化率 (Unique Order CVR)。
    # 有多少比例的session对这个商品在加购后下单(发生下单的独立会话数) / (发生加购的独立会话数)。
    unique_session['order_cvr_unique'] = unique_session['order_session'] / unique_session['cart_session']
    # 计算基于唯一会话数的点击到下单转化率 (Unique Click-Order CVR)。
    # 有多少比例的session对这个商品在点击后下单(发生下单的独立会话数) / (发生点击的独立会话数)。
    unique_session['click_order_cvr_unique'] = unique_session['order_session'] / unique_session['click_session']

    # 对 'click_cvr_unique' 特征应用自然对数 (log) 变换。
    # *注意*:
    # 1. 使用 `.to_pandas()` 可能会引入性能开销，尤其是在 cuDF 环境下。
    # 2. 直接对比例值取 `np.log`，如果值为 0 或负数（理论上比例不应为负），会产生 -Inf 或 NaN。
    #    使用 `np.log1p` (计算 log(1+x)) 通常更安全，因为 log1p(0) = 0。
    # 3. 如果 `click_cvr_unique` 可能大于 1 (如前所述)，取对数后可能为正。
    # 这里保留原始码的写法，但在实际应用中建议谨慎处理。
    # ['aid', 'click_session', 'cart_session', 'order_session', 'click_cvr_unique', 'cart_cvr_unique', 'order_cvr_unique', 'click_order_cvr_unique']
    unique_session['click_cvr_unique'] = np.log(unique_session['click_cvr_unique']) # 原始码
    # 建议的更安全写法 (如果需要对数变换):
    # temp_series = unique_session['click_cvr_unique'].to_pandas()
    # unique_session['click_cvr_unique'] = np.log1p(unique_session['click_cvr_unique'].clip(lower=0)) # clip 确保非负

    unique_session = unique_session.to_pandas()

    del unique_order_session, unique_cart_session, unique_click_session
    gc.collect()



    # ========== 6. 计算基于总事件数的特征 ==========
    # 这部分计算基于所有事件记录总数的转化率和计数。
    # 包括：
    # 1. cvr_df['cart_cvr'] : (商品A总加购次数) / (商品A总点击次数)。
    # 2. cvr_df['order_cvr'] : (商品A总下单次数) / (商品A总加购次数)。
    # 3. cvr_df['click_order_cvr'] : (商品A总下单次数) / (商品A总点击次数)。
    print("计算基于总事件数的特征...")
    # 计算每个 aid 的总点击次数 (value_counts 默认统计次数)。
    click_counts = merge_actions[merge_actions['type'] == 0]['aid'].value_counts().reset_index()
    click_counts.columns = ['aid', 'click_n']
    # 计算每个 aid 的总加购次数。
    cart_counts = merge_actions[merge_actions['type'] == 1]['aid'].value_counts().reset_index()
    cart_counts.columns = ['aid', 'cart_n']
    # 计算每个 aid 的总下单次数。
    order_counts = merge_actions[merge_actions['type'] == 2]['aid'].value_counts().reset_index()
    order_counts.columns = ['aid', 'order_n']

    # 将三种事件类型的总次数统计结果左连接合并。
    # ['aid', 'click_n', 'cart_n', 'order_n']
    cvr_df = click_counts.merge(cart_counts, on = 'aid', how = 'left')
    cvr_df = cvr_df.merge(order_counts, on = 'aid', how = 'left')

    # 计算基于事件总数的点击到加购转化率 (Cart CVR)。
    # ['aid', 'click_n', 'cart_n', 'order_n', 'cart_cvr', 'order_cvr', 'click_order_cvr']
    # (总加购次数) / (总点击次数)。
    cvr_df['cart_cvr'] = cvr_df['cart_n'] / cvr_df['click_n']
    # 计算基于事件总数的加购到下单转化率 (Order CVR)。
    # (总下单次数) / (总加购次数)。
    cvr_df['order_cvr'] = cvr_df['order_n'] / cvr_df['cart_n']
    # 计算基于事件总数的点击到下单转化率 (Click-Order CVR)。
    # (总下单次数) / (总点击次数)。
    cvr_df['click_order_cvr'] = cvr_df['order_n'] / cvr_df['click_n']

    # 将计算出的转化率转换为 float32 数据类型，以节省内存。
    cvr_df['cart_cvr'] = cvr_df['cart_cvr'].astype(np.float32)
    cvr_df['order_cvr'] = cvr_df['order_cvr'].astype(np.float32)
    cvr_df['click_order_cvr'] = cvr_df['click_order_cvr'].astype(np.float32)

    cvr_df = cvr_df.to_pandas()

    del click_counts, cart_counts, order_counts
    gc.collect()

    # ========== 7. 计算重复交互比例 ==========
    # 这部分计算用户对同一个 aid 进行重复交互的程度。
    # 包括：
    # 1. repeat_df['click_repeat_ratio'] : 商品A被重复点击的比例（点击商品A的session数/商品A被点击的总次数)
    # 2. repeat_df['cart_repeat_ratio'] : 商品A被重复加购的比例（加购商品A的session数/商品A被加购的总次数)
    # 3. repeat_df['order_repeat_ratio'] : 商品A被重复下单的比例（下单商品A的session数/商品A被下单的总次数)
    print("计算重复交互比例...")
    # 合并唯一会话计数 (来自 unique_session) 和总事件计数 (来自 cvr_df)。
    # 选取需要的列进行合并。
    # ['aid', 'click_session', 'cart_session', 'order_session', 'click_n', 'cart_n', 'order_n']
    repeat_df = unique_session[['aid', 'click_session', 'cart_session', 'order_session']].merge(
        cvr_df[['aid', 'click_n', 'cart_n', 'order_n']], on = 'aid', how = 'left')

    # 计算点击事件的重复率。
    # (唯一点击会话数) / (总点击次数)。
    # 值越接近 1，表示商品在每个会话中平均只被点击一次。
    # 值越小，表示商品在单个会话被重复点击的次数越多。
    repeat_df['click_repeat_ratio'] = repeat_df['click_session'] / repeat_df['click_n']
    # 计算加购事件的重复率。
    repeat_df['cart_repeat_ratio'] = repeat_df['cart_session'] / repeat_df['cart_n']
    # 计算下单事件的重复率。
    repeat_df['order_repeat_ratio'] = repeat_df['order_session'] / repeat_df['order_n']
    # 选取最终需要的列: aid, 三种事件的重复率。
    # ['aid', 'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio']
    repeat_df = repeat_df[['aid', 'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio']]


    # ========== 8. 计算基于测试/验证集(test)的转化率 ==========
    # 包括：
    # 1. cvr_df_test['cart_cvr_test'] : 测试集中加购的转化率（测试集中商品A被加购的次数/测试集中商品A被点击的次数）
    # 2. cvr_df_test['order_cvr_test'] : 测试集中下单的转化率（测试集中商品A被下单的次数/测试集中商品A被加购的次数）
    print("计算基于测试集(train)的转化率...")
    # 计算测试集中每个 aid 的总点击次数。
    click_counts_test = test_actions[test_actions['type'] == 0]['aid'].value_counts().reset_index()
    click_counts_test.columns = ['aid', 'click_n']
    # 计算测试集中每个 aid 的总加购次数。
    cart_counts_test = test_actions[test_actions['type'] == 1]['aid'].value_counts().reset_index()
    cart_counts_test.columns = ['aid', 'cart_n']
    # 计算测试集中每个 aid 的总下单次数。
    order_counts_test = test_actions[test_actions['type'] == 2]['aid'].value_counts().reset_index()
    order_counts_test.columns = ['aid', 'order_n']

    # 合并测试集中的事件计数。
    # ['aid', 'click_n', 'cart_n', 'order_n']
    cvr_df_test = click_counts_test.merge(cart_counts_test, on = 'aid', how = 'left')
    cvr_df_test = cvr_df_test.merge(order_counts_test, on = 'aid', how = 'left')

    # 计算基于测试集数据的点击到加购转化率。
    cvr_df_test['cart_cvr_test'] = cvr_df_test['cart_n'] / cvr_df_test['click_n']
    # 计算基于测试集数据的加购到下单转化率。
    cvr_df_test['order_cvr_test'] = cvr_df_test['order_n'] / cvr_df_test['cart_n']
    # 选取需要的列: aid, 测试集加购转化率, 测试集下单转化率。
    # ['aid', 'cart_cvr_test', 'order_cvr_test']
    cvr_df_test = cvr_df_test[['aid', 'cart_cvr_test', 'order_cvr_test']]

    # 转换数据类型为 float32。
    cvr_df_test['cart_cvr_test'] = cvr_df_test['cart_cvr_test'].astype(np.float32)
    cvr_df_test['order_cvr_test'] = cvr_df_test['order_cvr_test'].astype(np.float32)

    cvr_df_test = cvr_df_test.to_pandas()

    del click_counts_test, cart_counts_test, order_counts_test
    gc.collect()



    # ========== 9. 计算merge集和test集的事件份额 (Share / Popularity) ==========
    # 计算每个 aid 的各类事件在其对应事件总量中所占的比例，反映其流行度。
    # 包括：
    # merge集：
    # 1. click_share_all['click_share_all'] : 商品A被点击的次数占所有商品被点击的总数的比例
    # 2. cart_share_all['cart_share_all'] : 商品A被加购的次数占所有商品被加购的总数的比例
    # 3. order_share_all['order_share_all'] : 商品A被下单的次数占所有商品被下单的总数的比例
    # test集：
    # 1. click_share_test['click_share_test'] : 商品A被点击的次数占所有商品被点击的总数的比例
    # 2. cart_share_test['cart_share_test'] : 商品A被加购的次数占所有商品被加购的总数的比例
    # 3. order_share_test['order_share_test'] : 商品A被下单的次数占所有商品被下单的总数的比例

    print("计算事件份额 (流行度)...")

    # --- 9.1 基于 merge 数据集 (全时期) ---
    # 计算在 merge 数据集中，每个 aid 的点击次数占总点击次数的比例 (全局点击份额)。
    # value_counts(normalize=True) 直接计算比例。
    click_share_all = merge_actions[merge_actions['type'] == 0]['aid'].value_counts(normalize = True).reset_index()
    click_share_all.columns = ['aid', 'click_share_all']
    # 计算全局加购份额。
    cart_share_all = merge_actions[merge_actions['type'] == 1]['aid'].value_counts(normalize = True).reset_index()
    cart_share_all.columns = ['aid', 'cart_share_all']
    # 计算全局下单份额。
    order_share_all = merge_actions[merge_actions['type'] == 2]['aid'].value_counts(normalize = True).reset_index()
    order_share_all.columns = ['aid', 'order_share_all']

    # --- 9.2 基于 test 数据集 (测试时期) ---
    # 计算在 test 数据集中，每个 aid 的点击份额。
    click_share_test = test_actions[test_actions['type'] == 0]['aid'].value_counts(normalize = True).reset_index()
    click_share_test.columns = ['aid', 'click_share_test']
    # 计算训练集加购份额。
    cart_share_test = test_actions[test_actions['type'] == 1]['aid'].value_counts(normalize = True).reset_index()
    cart_share_test.columns = ['aid', 'cart_share_test']
    # 计算训练集下单份额。
    order_share_test = test_actions[test_actions['type'] == 2]['aid'].value_counts(normalize = True).reset_index()
    order_share_test.columns = ['aid', 'order_share_test']

    click_share_all = click_share_all.to_pandas()
    cart_share_all = cart_share_all.to_pandas()
    order_share_all = order_share_all.to_pandas()
    click_share_test = click_share_test.to_pandas()
    cart_share_test = cart_share_test.to_pandas()
    order_share_test = order_share_test.to_pandas()



    # ========== 10. 计算其他时间相关特征 ==========
    # 包括：
    # 1. session_diff_action['aid_next_diff_mean'] : 在同一session中，商品A出现的时间间隔（2次以上）的平均数（对所有session中商品A出现的时间间隔取平均）
    # 2. merge_ts_rank['aid_rank_mean'] : 商品A在每个session中出现的平均顺序（第几个被交互）
    # 3. aid_max_df['aid_last_action_diff'] : 商品A最后一次被交互的时间与全局最后一次交互发生的时间的差值（单位：小时）
    print("计算其他时间相关特征...")
    # --- 10.1 aid 同一会话内相邻动作的平均时间差 ---
    # 选取 session, aid, ts 列，并按 session 和时间戳排序。
    # ['session', 'aid', 'ts']
    session_aid_ts = merge_actions[['session', 'aid', 'ts']].sort_values(['session', 'ts'])
    session_aid_ts = session_aid_ts.to_pandas()
    # 使用 groupby(['session', 'aid'])['ts'].diff() 计算同一 session 内，同一 aid 相邻两次出现的时间戳差值 (单位：秒)。
    # ['session', 'aid', 'ts', 'aid_next_diff_mean']
    session_aid_ts['aid_next_diff'] = session_aid_ts.groupby(['session', 'aid'])['ts'].diff()
    # 按 aid 分组，计算该 aid 在所有会话中，相邻两次出现的平均时间差 (单位：秒)。
    # 注意：列名仍然是 'aid_next_diff_mean'，但现在是 aid 级别的平均值。
    #       如果一个 aid 在某个 session 只出现一次，diff() 结果是 NaN，mean() 会忽略 NaN。
    # [aid, aid_next_diff_mean]
    session_diff_action = session_aid_ts.groupby(['aid'])['aid_next_diff'].mean().reset_index()
    session_diff_action.columns = ['aid', 'aid_next_diff_mean']

    # --- 10.2 aid 在会话中的平均排名 ---
    # 衡量一个 aid 通常在会话的早期还是晚期被交互。
    # 按 session 和 aid 分组，获取每个 aid 在该 session 中首次出现的时间戳。
    merge_ts_rank = merge_actions.groupby(['session', 'aid'])['ts'].first().reset_index()
    # 按 session 和首次出现时间戳排序。
    merge_ts_rank = merge_ts_rank.sort_values(['session', 'ts']).reset_index(drop=True)
    # 初始化 rank 列为 1。
    merge_ts_rank['rank'] = 1
    # 按 session 分组，使用 cumsum() 计算每个 aid 在该 session 中按时间顺序的排名。
    merge_ts_rank['rank'] = merge_ts_rank.groupby('session')['rank'].cumsum()
    # 按 aid 分组，计算该 aid 在所有出现过的 session 中的平均排名。
    merge_ts_rank = merge_ts_rank.groupby('aid')['rank'].mean().reset_index()
    merge_ts_rank.columns = ['aid', 'aid_rank_mean'] # 重命名列

    # --- 10.3 aid 最后活跃距今时间 ---
    # 衡量一个 aid 最近一次被交互的时间。
    # 按 aid 分组，获取该 aid 最后一次出现（最大）的时间戳。
    aid_max_df = merge_actions.groupby('aid')['ts'].max().reset_index()
    # 计算全局最晚时间戳 (`merge['ts'].max()`) 与该 aid 最后一次出现时间戳的时间差(单位：小时)
    aid_max_df['aid_last_action_diff'] = (merge_actions['ts'].max() - aid_max_df['ts'] ) / 3600
    # 选取需要的列: aid, 最后活跃时间差。
    aid_max_df = aid_max_df[['aid', 'aid_last_action_diff']]

    # session_diff_action = session_diff_action.to_pandas()
    merge_ts_rank = merge_ts_rank.to_pandas()
    aid_max_df = aid_max_df.to_pandas()

    del session_aid_ts
    gc.collect()

    # ========== 11. 合并所有计算得到的特征 ==========
    print("合并所有计算得到的特征...")
    cart_cvr_df = reload_to_gpu(cart_cvr_df)
    aid_features = aid_features.merge(cart_cvr_df, on = 'aid', how = 'left')        # 历史会话转化率
    del cart_cvr_df
    gc.collect()
    
    cvr_df = reload_to_gpu(cvr_df)
    aid_features = aid_features.merge(cvr_df, on = 'aid', how = 'left')             # 全局事件数转化率 & 计数
    del cvr_df
    gc.collect()

    cvr_df_test = reload_to_gpu(cvr_df_test)
    aid_features = aid_features.merge(cvr_df_test, on = 'aid', how = 'left')        # 训练集事件数转化率
    del cvr_df_test
    gc.collect()

    click_share_all = reload_to_gpu(click_share_all)
    aid_features = aid_features.merge(click_share_all, on = 'aid', how = 'left')    # 全局点击份额
    del click_share_all
    gc.collect()

    click_share_test = reload_to_gpu(click_share_test)
    aid_features = aid_features.merge(click_share_test, on = 'aid', how = 'left')   # 训练集点击份额
    del click_share_test
    gc.collect()

    cart_share_all = reload_to_gpu(cart_share_all)
    aid_features = aid_features.merge(cart_share_all, on = 'aid', how = 'left')     # 全局加购份额
    del cart_share_all
    gc.collect()

    cart_share_test = reload_to_gpu(cart_share_test)
    aid_features = aid_features.merge(cart_share_test, on = 'aid', how = 'left')    # 训练集加购份额
    del cart_share_test
    gc.collect()

    order_share_all = reload_to_gpu(order_share_all)
    aid_features = aid_features.merge(order_share_all, on = 'aid', how = 'left')    # 全局下单份额
    del order_share_all
    gc.collect()

    order_share_test = reload_to_gpu(order_share_test)
    aid_features = aid_features.merge(order_share_test, on = 'aid', how = 'left')   # 训练集下单份额
    del order_share_test
    gc.collect()

    session_diff_action = reload_to_gpu(session_diff_action)
    aid_features = aid_features.merge(session_diff_action, on = 'aid', how = 'left')# aid 同会话平均时间间隔
    del session_diff_action
    gc.collect()

    unique_session = reload_to_gpu(unique_session)
    aid_features = aid_features.merge(unique_session, on = 'aid', how = 'left')     # 唯一会话数 & 转化率
    del unique_session
    gc.collect()

    repeat_df = reload_to_gpu(repeat_df)
    aid_features = aid_features.merge(repeat_df, on = 'aid', how = 'left')          # 重复交互比例
    del repeat_df
    gc.collect()

    merge_ts_rank = reload_to_gpu(merge_ts_rank)
    aid_features = aid_features.merge(merge_ts_rank, on = 'aid', how = 'left')      # aid 平均会话排名 (修正变量名)
    del merge_ts_rank
    gc.collect()

    aid_max_df = reload_to_gpu(aid_max_df)
    aid_features = aid_features.merge(aid_max_df, on = 'aid', how = 'left')         # aid 最后活跃时间差
    del aid_max_df
    gc.collect()

    cart_df = reload_to_gpu(cart_df)
    aid_features = aid_features.merge(cart_df, on = 'aid', how = 'left')            # 加购跳过点击比例
    del cart_df
    gc.collect()

    order_df = reload_to_gpu(order_df)
    aid_features = aid_features.merge(order_df, on = 'aid', how = 'left')           # 下单跳过点击/加购比例
    del order_df
    gc.collect()

    type_counts = reload_to_gpu(type_counts)
    aid_features = aid_features.merge(type_counts, on = 'aid', how = 'left')        # 事件转换比例 & 份额
    del type_counts
    gc.collect()

    type_ts_diff = reload_to_gpu(type_ts_diff)
    aid_features = aid_features.merge(type_ts_diff, on = 'aid', how = 'left')       # 事件转换平均时间差
    del type_ts_diff
    gc.collect()

    float32_cols = ['cart_hist_cvr', 'order_hist_cvr', 'aid_next_diff_mean', 'click_cvr_unique', 
                    'cart_cvr_unique', 'order_cvr_unique', 'click_order_cvr_unique', 
                    'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio', 'aid_rank_mean',
                   'cart_click_skip_ratio', 'order_cart_skip_ratio', 'order_click_skip_ratio',
                   'aid_click_click_ratio', 'aid_cart_click_ratio',
                   'aid_order_click_ratio', 'aid_click_cart_ratio', 'aid_cart_cart_ratio',
                   'aid_order_cart_ratio', 'aid_click_order_ratio', 'aid_cart_order_ratio',
                   'aid_order_order_ratio', 'cvr_sum', 'cvr_sum_share',
                   'aid_click_click_diffts', 'aid_click_cart_diffts',
                   'aid_cart_click_diffts', 'aid_cart_order_diffts']

    aid_features[float32_cols] = aid_features[float32_cols].astype(np.float32)
        
    # *注意*: 填充 0 可能会改变某些特征的统计分布，需要根据具体业务理解决定是否合适。
    #         例如，对于转化率，填充 0 可能表示该路径未发生，是合理的；
    #         但对于时间差，填充 0 可能引入误导信息，也许填充平均值或中位数更好，或者保持 NaN。
    aid_features = aid_features.to_pandas().fillna(0)
    aid_features['aid_next_diff_mean'] = np.log1p(aid_features['aid_next_diff_mean'])

    aid_features = aid_features[['aid', 'cart_hist_cvr', 'order_hist_cvr', 'cart_cvr', 'order_cvr', 
                                 'click_order_cvr', 'cart_cvr_test', 'order_cvr_test',
                                 'click_cvr_unique', 'cart_cvr_unique', 'order_cvr_unique', 'click_order_cvr_unique',
                                 'click_share_all', 'click_share_test', 'cart_share_all',
                                 'cart_share_test', 'order_share_all', 'order_share_test', 'aid_next_diff_mean',
                                'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio', 'aid_rank_mean',
                                'aid_last_action_diff', 'cart_click_skip_ratio', 'order_cart_skip_ratio', 
                                 'order_click_skip_ratio', 'aid_click_click_ratio', 'aid_cart_click_ratio',
                               'aid_order_click_ratio', 'aid_click_cart_ratio', 'aid_cart_cart_ratio',
                               'aid_order_cart_ratio', 'aid_click_order_ratio', 'aid_cart_order_ratio',
                               'aid_order_order_ratio', 'cvr_sum', 'cvr_sum_share',
                               'aid_click_click_diffts', 'aid_click_cart_diffts',
                               'aid_cart_click_diffts', 'aid_cart_order_diffts']]
    
    print('正在保存aid特征文件...')
    aid_features.to_parquet(output_path + prefix + 'aid_features_df.parquet')
    print(f"aid 特征已成功保存到: {output_path + prefix}aid_features_df.parquet")
    del aid_features
    
    
def gen_last_chunk_session_aid(prefix, test_actions):
    """
    为每个会话(session)和商品(aid)提取其在“最后一个活动块”中的交互特征。

    这个函数首先根据事件间的时间差将每个会话划分为不同的“活动块”(chunk)，
    然后重点分析每个会话的最后一个活动块，统计其中每个商品被点击、加购、下单的次数，
    以及该商品交互次数占该块总交互次数的比例等特征。

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，要求至少包含以下列：
                                        [session, aid, ts, type]

    返回：
        None: 函数执行完毕后，会将生成的包含最后一个活动块特征的 DataFrame
              保存到指定的 output_path 目录下，
              文件名格式为 {output_path}{prefix}last_chunk_session_aid_df.parquet。
    """
    # --- 1. 准备数据和计算时间差 ---
    # 复制输入 DataFrame，避免修改原始数据
    test_chunk = test_actions.copy()
    # 按 session 分组，计算同会话内相邻事件的时间戳差值 (单位：秒)
    # diff() 计算当前行与上一行的差值
    test_chunk['ts_diff'] = test_chunk.groupby('session')['ts'].diff()
    # ['session', 'aid', 'ts', 'type', 'ts_diff']
    test_chunk['ts_diff'] = test_chunk['ts_diff'].fillna(0)

    # --- 2. 划分活动块 (Chunk) ---
    # 定义活动块：如果两个相邻事件的时间差小于 1 小时 (3600 秒)，则认为它们属于同一个活动块。
    # 创建一个标志列 'chunk_flg'：时间差小于 3600 秒为 0，否则为 1 (表示新块的开始)
    # np.where 用于条件判断赋值
    # ['session', 'aid', 'ts', 'type', 'ts_diff', 'chunk_flg']
    test_chunk['chunk_flg'] = np.where(test_chunk.to_pandas()['ts_diff'] < 3600, 0, 1)
    # 按 session 分组，对 'chunk_flg' 进行累加求和 (cumsum)
    # 这样，每个 session 内的事件就被分配到了不同的 chunk 编号 (从 0 或 1 开始)
    # ['session', 'aid', 'ts', 'type', 'ts_diff', 'chunk_flg', 'chunk']
    test_chunk['chunk'] = test_chunk.groupby('session')['chunk_flg'].cumsum()

    # --- 3. 确定最后一个活动块 ---
    # 按 session 分组，找到每个 session 的最大 chunk 编号，即最后一个活动块的编号
    test_max_chunk = test_chunk.groupby('session')['chunk'].max().reset_index()
    test_max_chunk.columns = ['session', 'max_chunk'] # 重命名列
    # 将最大 chunk 编号信息合并回 test_chunk
    # ['session', 'aid', 'ts', 'type', 'ts_diff', 'chunk_flg', 'chunk', 'max_chunk']
    test_chunk = test_chunk.merge(test_max_chunk, on = 'session', how = 'left')

    # --- 4. 计算事件在块内的逆序排名 ---
    # 按 session 和时间戳排序，确保后续 rank 计算基于时间顺序
    test_chunk = test_chunk.sort_values(['session', 'ts']).reset_index(drop=True)
    # 按 session 和 chunk 分组，计算每个事件在块内的逆序排名 (按 ts 降序)
    # rank(ascending=False) 使得时间戳最大的事件排名为 1
    # method='max' 处理时间戳相同时的排名方式 (都取最大排名)
    # 'last_chunk_num' 表示事件距离块内最后一个事件的位置 (1 表示最后那个事件)
    test_chunk['last_chunk_num'] = test_chunk.groupby(['session', 'chunk'])['ts'].rank(ascending = False, method = 'max')

    # --- 5. 计算会话平均块内事件数 ---
    # 统计每个 session 内每个 chunk 的事件数量
    chunk_count = test_chunk[['session', 'chunk']].value_counts().reset_index()
    chunk_count.columns = ['session', 'chunk', 'chunk_counts'] # 重命名列
    # 按 session 分组，计算每个 session 内平均每个 chunk 包含多少事件
    # ['session', 'session_counts_mean']
    chunk_count = chunk_count.groupby('session')['chunk_counts'].mean().reset_index()
    chunk_count.columns = ['session', 'session_counts_mean'] # 重命名列

    # --- 6. 提取最后一个活动块的数据并聚合 ---
    # 筛选出只属于最后一个活动块的事件记录
    test_last_chunk = test_chunk[test_chunk['chunk'] == test_chunk['max_chunk']].copy()
    # 对最后一个活动块的数据进行透视操作 (pivot_table)：
    # - aggfunc: 'count' - 统计每个 session-aid 对下，各类事件发生的次数
    # [session, aid, 0(被click的次数), 1(被cart的次数), 2(被order的次数)]
    test_last_chunk = test_last_chunk.pivot_table(index = ['session', 'aid'],
                                                     columns = ['type'],
                                                     values = ['ts'],
                                                     aggfunc='count').reset_index()
    # 重命名透视后的多级列名
    # 列名格式：last_chunk_{事件类型} (click/cart/order)
    test_last_chunk.columns = ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order']
    test_last_chunk = test_last_chunk.fillna(0)
    # 计算每个 session-aid 对在最后一个块中的总交互次数
    # iloc[:, 2:] 选取点击、加购、下单次数列
    # ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total']
    test_last_chunk['last_chunk_aid_total'] = test_last_chunk.iloc[:,2:].sum(axis=1) # axis=1 按行求和

    # --- 7. 合并辅助特征 ---
    # 将之前计算的最大块编号 (max_chunk) 和会话平均块内事件数 (session_counts_mean) 合并进来
    # ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total', 
    #   'max_chunk', 'session_counts_mean']
    test_last_chunk = test_last_chunk.merge(test_max_chunk, on = 'session', how = 'left')
    test_last_chunk = test_last_chunk.merge(chunk_count, on = 'session', how = 'left')

    # --- 8. 计算最后一个块的总交互数和商品交互比例 ---
    # 按 session 分组，计算每个 session 在最后一个块中的总交互次数 (所有 aid 的交互次数之和)
    test_last_chunk_total = test_last_chunk.groupby('session')['last_chunk_aid_total'].sum().reset_index()
    test_last_chunk_total.columns = ['session', 'last_chunk_total'] # 重命名列
    # 将会话在最后一个块的总交互次数合并回来
    # ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total', 
    # 'max_chunk', 'session_counts_mean', 'last_chunk_total']
    test_last_chunk = test_last_chunk.merge(test_last_chunk_total, on = 'session', how = 'left')
    # 计算每个 session-aid 对的交互次数占该 session 最后一个块总交互次数的比例
    # ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total', 
    # 'max_chunk', 'session_counts_mean', 'last_chunk_total', 'last_chunk_aid_ratio']
    test_last_chunk['last_chunk_aid_ratio'] = test_last_chunk['last_chunk_aid_total'] / test_last_chunk['last_chunk_total']

    # --- 9. 合并商品在块内最后交互的排名 ---
    # 从 test_chunk (包含所有块的数据) 中提取每个 session-aid 对的最小 'last_chunk_num'
    # 因为 rank 是降序排的，最小值代表该 aid 在其所在块中 *最后一次* 出现的位置距离块结尾有多近
    # (例如，最小值为 1 表示该 aid 是块内交互的最后一个商品之一)
    #  [session, aid, last_chunk_num]
    test_chunk_last_num = test_chunk.groupby(['session', 'aid'])['last_chunk_num'].min().reset_index()
    # 将这个最小排名（最后交互位置）合并到 test_last_chunk DataFrame
    # 注意：这里合并的是该 aid 在 *所有* 块中最后一次出现的位置的最小排名，
    #       而 test_last_chunk 只包含最后一个块的数据。如果某 aid 只在非最后块出现，这里会是 NaN。
    # ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total', 
    # 'max_chunk', 'session_counts_mean', 'last_chunk_total', 'last_chunk_aid_ratio', 'last_chunk_num']
    test_last_chunk = test_last_chunk.merge(test_chunk_last_num, on = ['session', 'aid'], how = 'left')

    del test_chunk, test_max_chunk, chunk_count, test_last_chunk_total, test_chunk_last_num
    gc.collect()

    # --- 10. 转换数据类型 ---
    # 定义需要转换为 float32 和 int32 的列名列表，以节省内存
    float32_cols = ['last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total',
                    'session_counts_mean', 'last_chunk_total', 'last_chunk_aid_ratio']
    int32_cols = ['max_chunk', 'last_chunk_num']
    test_last_chunk = test_last_chunk.fillna(0)
    test_last_chunk[float32_cols] = test_last_chunk[float32_cols].astype(np.float32)
    test_last_chunk[int32_cols] = test_last_chunk[int32_cols].astype(np.int32)

    # --- 11. 保存结果 ---
    # 构建完整的输出文件路径
    output_file = output_path + prefix + 'last_chunk_session_aid_df.parquet'
    # 将最终处理好的特征 DataFrame 保存为 Parquet 文件
    print("正在保存last_chunk_session 特征文件...")
    test_last_chunk.to_parquet(output_file, index=False) # index=False 避免写入索引
    print(f"last_chunk_session 特征已保存到 {output_file}。")

    del test_last_chunk
    gc.collect()


def gen_user_aid_feature(prefix, test_actions):
    """
    为训练/测试数据集生成用户会话(session)和商品(aid)相关的交互特征。

    该函数计算了每个会话中每个商品的各种交互特征，包括总行为数、不同行为类型（点击、加入购物车、下单）的占比，
    以及在会话结束前不同时间窗口（最近1小时、最近1天、最近1周）内该商品的特定行为数。

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，要求至少包含以下列：
                                        [session, aid, ts, type]

    返回：
        None: 函数执行完毕后，会将生成的包含 session-aid 特征的 DataFrame
              保存到指定的 output_path 目录下，文件名为 {output_path}{prefix}session_aid_df.parquet。
    """
    # --- 1. 计算session总行为数、session-aid总行为数及占比 ---
    # 计算每个 session 的总行为数
    # ['session', 'session_total_action']
    session_aid_total_df = test_actions[['session']].value_counts().reset_index()
    session_aid_total_df.columns = ['session', 'session_total_action']

    # 计算每个 session 中每个 aid 的总行为数
    # ['session', 'aid', 'session_aid_total_action']
    session_aid_df = test_actions[['session', 'aid']].value_counts().reset_index()
    session_aid_df.columns = ['session', 'aid', 'session_aid_total_action']
    # 将 session 的总行为数合并到 session_aid_df 中，以便计算占比
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action']
    session_aid_df = session_aid_df.merge(session_aid_total_df, on = 'session', how = 'left')
    # 计算每个 session 中每个 aid 的行为数占该 session 总行为数的比例
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action', 'session_aid_share']
    session_aid_df['session_aid_share'] = session_aid_df['session_aid_total_action'] / session_aid_df['session_total_action']

    # --- 2. 计算不同行为类型 (点击、加购、下单) 的 session-aid 行为数及占比 ---
    # 筛选出点击行为数据，计算每个 session-aid 的总点击数
    session_aid_click_df = test_actions[test_actions['type'] == 0][['session', 'aid']].value_counts().reset_index()
    session_aid_click_df.columns = ['session', 'aid', 'session_aid_total_click']
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action', 'session_aid_share', 'session_aid_total_click']
    session_aid_df = session_aid_df.merge(session_aid_click_df, on = ['session', 'aid'], how = 'left')
    # 计算每个 session-aid 对的点击数占该 aid 在该 session 中总行为数的比例（点击率）
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action', 'session_aid_share', 
    # 'session_aid_total_click', 'session_aid_click_share']
    session_aid_df['session_aid_click_share'] = session_aid_df['session_aid_total_click'] / session_aid_df['session_aid_total_action']

    # 筛选出加入购物车行为数据，计算每个 session-aid 的总加购数
    session_aid_cart_df = test_actions[test_actions['type'] == 1][['session', 'aid']].value_counts().reset_index()
    session_aid_cart_df.columns = ['session', 'aid', 'session_aid_total_cart']
    # 将加购数合并到 session_aid_df 中
    session_aid_df = session_aid_df.merge(session_aid_cart_df, on = ['session', 'aid'], how = 'left')
    # 计算每个 session-aid 对的加购数占该 aid 在该 session 中总行为数的比例
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action', 'session_aid_share', 
    # 'session_aid_total_click', 'session_aid_click_share', 'session_aid_total_cart', 'session_aid_cart_share']
    session_aid_df['session_aid_cart_share'] = session_aid_df['session_aid_total_cart'] / session_aid_df['session_aid_total_action']

    # 筛选出下单行为数据，计算每个 session-aid 的总下单数
    session_aid_order_df = test_actions[test_actions['type'] == 2][['session', 'aid']].value_counts().reset_index()
    session_aid_order_df.columns = ['session', 'aid', 'session_aid_total_order']
    # 将下单数合并到 session_aid_df 中
    session_aid_df = session_aid_df.merge(session_aid_order_df, on = ['session', 'aid'], how = 'left')
    # 计算每个 session-aid 对的下单数占该 aid 在该 session 中总行为数的比例
    # ['session', 'aid', 'session_aid_total_action', 'session_total_action', 'session_aid_share', 
    # 'session_aid_total_click', 'session_aid_click_share', 'session_aid_total_cart', 'session_aid_cart_share', 
    # 'session_aid_total_order', 'session_aid_order_share']
    session_aid_df['session_aid_order_share'] = session_aid_df['session_aid_total_order'] / session_aid_df['session_aid_total_action']
    # 选取最终需要的列
    session_aid_df = session_aid_df[['session', 'aid', 'session_aid_total_action', 'session_aid_share',
                                     'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share']]
    session_aid_df = session_aid_df.fillna(0)

    del session_aid_click_df, session_aid_cart_df, session_aid_order_df
    del session_aid_total_df 
    gc.collect()

    float32_cols = ['session_aid_total_action', 'session_aid_share',
                    'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share']

    # ['session', 'aid', 'session_aid_total_action', 'session_aid_share',
    # 'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share'] 
    session_aid_df[float32_cols] = session_aid_df[float32_cols].astype(np.float32)


    # --- 3. 计算会话结束前不同时间窗口 (最近1小时、1天、1周) 内的 session-aid 行为数 ---
    # 计算每个 session 的最大时间戳 (即 session 的结束时间)
    # ['session', 'ts_max']
    session_end_ts = test_actions.groupby(['session'])['ts'].max().reset_index()
    session_end_ts.columns = ['session', 'ts_max']

    # --- 3.1 计算最近1小时内的行为数 ---
    # 计算最大时间戳减去1小时的时间点 (单位：秒)
    # ['session', 'ts_max', 'diff_1hour']
    session_end_ts['diff_1hour']  = session_end_ts['ts_max'] - (60 * 60) # 1小时 = 3600 秒
    # 将计算出的1小时前时间点合并回原始测试数据
    test_with_end_ts = test_actions.merge(session_end_ts[['session', 'diff_1hour']], on ='session', how = 'left')
    # 筛选出时间戳在最近1小时内的行为数据
    # ['session', 'aid', 'ts', 'type', 'diff_1hour']
    test_last_1hour = test_with_end_ts[test_with_end_ts['ts'] >= test_with_end_ts['diff_1hour']].copy() # 复制以避免 SettingWithCopyWarning
    del test_with_end_ts
    gc.collect()

    # 在最近1小时的行为数据中，计算每个 session-aid 的总点击数
    test_last_1hour_click = test_last_1hour[test_last_1hour['type'] == 0][['session', 'aid']].value_counts().reset_index()
    test_last_1hour_click.columns = ['session', 'aid', 'last_1hour_clicks']

    # 在最近1小时的行为数据中，计算每个 session-aid 的总加入购物车数
    test_last_1hour_cart = test_last_1hour[test_last_1hour['type'] == 1][['session', 'aid']].value_counts().reset_index()
    test_last_1hour_cart.columns = ['session', 'aid', 'last_1hour_carts']

    # 在最近1小时的行为数据中，计算每个 session-aid 的总下单数
    test_last_1hour_order = test_last_1hour[test_last_1hour['type'] == 2][['session', 'aid']].value_counts().reset_index()
    test_last_1hour_order.columns = ['session', 'aid', 'last_1hour_orders']
    del test_last_1hour
    gc.collect()

    # --- 3.2 计算最近1天的行为数 (逻辑同上，时间窗口变为24小时) ---
    # ['session', 'ts_max', 'diff_1hour', 'diff_1day']
    session_end_ts['diff_1day']  = session_end_ts['ts_max'] - (24 * 60 * 60) # 1天 = 86400 秒
    test_with_end_ts = test_actions.merge(session_end_ts[['session', 'diff_1day']], on ='session', how = 'left')
    test_last_1day = test_with_end_ts[test_with_end_ts['ts'] >= test_with_end_ts['diff_1day']].copy() 
    del test_with_end_ts 
    gc.collect()

    # 在最近1天的行为数据中，计算每个 session-aid 的总点击数
    test_last_1day_click = test_last_1day[test_last_1day['type'] == 0][['session', 'aid']].value_counts().reset_index()
    test_last_1day_click.columns = ['session', 'aid', 'last_1day_clicks']

    # 在最近1天的行为数据中，计算每个 session-aid 的总加入购物车数
    test_last_1day_cart = test_last_1day[test_last_1day['type'] == 1][['session', 'aid']].value_counts().reset_index()
    test_last_1day_cart.columns = ['session', 'aid', 'last_1day_carts']

    # 在最近1天的行为数据中，计算每个 session-aid 的总下单数
    test_last_1day_order = test_last_1day[test_last_1day['type'] == 2][['session', 'aid']].value_counts().reset_index()
    test_last_1day_order.columns = ['session', 'aid', 'last_1day_orders']
    del test_last_1day
    gc.collect()

    # --- 3.3 计算最近1周的行为数 (逻辑同上，时间窗口变为7天) ---
    # ['session', 'ts_max', 'diff_1hour', 'diff_1day', 'diff_1week']
    session_end_ts['diff_1week']  = session_end_ts['ts_max'] - (7 * 24 * 60 * 60) # 1周 = 604800 秒
    test_with_end_ts = test_actions.merge(session_end_ts[['session', 'diff_1week']], on ='session', how = 'left')
    test_last_1week = test_with_end_ts[test_with_end_ts['ts'] >= test_with_end_ts['diff_1week']].copy() # 复制以避免 SettingWithCopyWarning
    del test_with_end_ts
    del session_end_ts
    gc.collect()

    # 在最近1周的行为数据中，计算每个 session-aid 的总点击数
    test_last_1week_click = test_last_1week[test_last_1week['type'] == 0][['session', 'aid']].value_counts().reset_index()
    test_last_1week_click.columns = ['session', 'aid', 'last_1week_clicks']

    # 在最近1周的行为数据中，计算每个 session-aid 的总加入购物车数
    test_last_1week_cart = test_last_1week[test_last_1week['type'] == 1][['session', 'aid']].value_counts().reset_index()
    test_last_1week_cart.columns = ['session', 'aid', 'last_1week_carts']

    # 在最近1周的行为数据中，计算每个 session-aid 的总下单数
    test_last_1week_order = test_last_1week[test_last_1week['type'] == 2][['session', 'aid']].value_counts().reset_index()
    test_last_1week_order.columns = ['session', 'aid', 'last_1week_orders']

    del test_last_1week
    gc.collect()

    # --- 4. 计算每个 session-aid 的最后一次行为与 session 结束时间的差值 ---
    # 计算每个 session 中每个 aid 的最后一次行为时间
    # [session, aid, ts]
    test_last_action = test_actions.groupby(['session', 'aid'])['ts'].max().reset_index()

    # 计算每个 session 的最后一次行为时间 (session 的结束时间)，并合并
    # [session, ts_max]
    session_end_ts_for_diff = test_actions.groupby(['session'])['ts'].max().reset_index()
    session_end_ts_for_diff.columns = ['session', 'ts_max'] # 重命名列名

    # 将 session 的结束时间合并到 test_last_action 中
    # [session, aid, ts, ts_max]
    test_last_action = test_last_action.merge(session_end_ts_for_diff, on = 'session', how = 'left')

    # 计算时间差 (session 结束时间 - aid 最后行为时间)，并转换为小时
    # [session, aid, ts, ts_max, last_action_diff_hour]
    test_last_action['last_action_diff_hour'] = (test_last_action['ts_max'] - test_last_action['ts']) / 3600 # 转换为小时
    
    # ['session', 'aid', 'last_action_diff_hour']
    test_last_action = test_last_action[['session', 'aid', 'last_action_diff_hour']]

    del session_end_ts_for_diff
    gc.collect()

    # --- 5. 合并所有计算出的特征 ---
    # 将 最近1小时、1天、1周内的行为数特征 和 最后一次行为时间差特征 合并到主 DataFrame (session_aid_df) 中
    # ['session', 'aid', 'session_aid_total_action', 'session_aid_share', 
    #   'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share', 
    #   'last_1hour_clicks', 'last_1hour_carts', 'last_1hour_orders', 
    #   'last_1day_clicks', 'last_1day_carts', 'last_1day_orders', 
    #   'last_1week_clicks', 'last_1week_carts', 'last_1week_orders', 
    #   'last_action_diff_hour']
    session_aid_df = session_aid_df.merge(test_last_1hour_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1hour_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1hour_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1day_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1day_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1day_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1week_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1week_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_1week_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(test_last_action, on = ['session', 'aid'], how = 'left')

    del test_last_1hour_click, test_last_1hour_cart, test_last_1hour_order
    del test_last_1day_click, test_last_1day_cart, test_last_1day_order
    del test_last_1week_click, test_last_1week_cart, test_last_1week_order
    del test_last_action
    gc.collect()

    session_aid_df = session_aid_df.to_pandas()
    # session_aid_df.fillna(0, inplace = True)
    
    print('正在保存user_aid特征文件...')
    session_aid_df.to_parquet(output_path + prefix + 'session_aid_df.parquet')
    print(f'user_aid特征文件已保存到{output_path + prefix}session_aid_df.parquet中.')
    del session_aid_df
    gc.collect()


def gen_session_action_feature(prefix, test_actions):
    """
    为每个会话(session)生成基于其包含的商品(aid)特征的聚合特征。

    这个函数读取预先计算好的商品(aid)特征(来自gen_aid_feature()函数)，将这些特征与用户行为数据合并，
    然后按会话(session)分组，计算每个会话中所有商品特征的平均值，
    从而得到代表该会话整体特征的聚合值。

    参数:
        prefix (str): 所有特征文件路径的前缀字符串，'train_'表示使用训练验证集，'test_'表示使用训练测试集。
        test_actions (cudf.DataFrame): 包含用户行为数据的 cudf dataframe 测试集数据，要求至少包含以下列：
                                        [session, aid, ts, type]

    返回：
        None: 函数执行完毕后，会将生成的包含会话聚合特征的 DataFrame
              保存到指定的 output_path 目录下，
              文件名为 {output_path}{prefix}session_use_aid_feat_df.parquet。
    """
    # --- 1. 读取预计算的商品特征并与用户行为数据合并 ---
    # 读取预计算好的 aid 特征文件
    aid_features = cudf.DataFrame(pd.read_parquet(output_path + prefix + 'aid_features_df.parquet'))

    # 用户行为数据去除重复行
    # ['session', 'aid', 'ts', 'type']
    test_dup = test_actions[['session', 'aid', 'ts', 'type']].drop_duplicates().reset_index(drop=True)

    # 将读取的 aid 特征合并到用户行为数据中
    # ['session', 'aid', 'ts', 'type', 
    # 'cart_cvr_unique', 'click_cvr_unique', 'click_share_all', 'aid_rank_mean']
    test_dup = test_dup.merge(aid_features[['aid', 'cart_cvr_unique', 'click_cvr_unique',
                                     'click_share_all', 'aid_rank_mean']], on = 'aid', how = 'left')

    del aid_features
    gc.collect()


    # --- 2. 按Session聚合商品特征 ---
    # 按 'session' 列进行分组，对合并后的商品特征列计算平均值
    # 对 'click_cvr_unique', 'cart_cvr_unique', 'click_share_all', 'aid_rank_mean' 列分别计算 'mean'
    # ['session', 'session_click_cvr_unique', 'session_cart_cvr_unique',
    #  'session_click_share_all', 'session_aid_rank_mean']
    test_dup = test_dup.groupby('session').agg({'click_cvr_unique':['mean'],
                                          'cart_cvr_unique':['mean'],
                                          'click_share_all':['mean'],
                                          'aid_rank_mean':['mean'],
                                         }).reset_index()
    test_dup.columns = ['session', 'session_click_cvr_unique', 'session_cart_cvr_unique',
                         'session_click_share_all', 'session_aid_rank_mean']

    test_dup = test_dup.fillna(0)

    # --- 3. 转换数据类型 ---
    float32_cols = ['session_click_cvr_unique', 'session_cart_cvr_unique',
                    'session_click_share_all', 'session_aid_rank_mean']
    test_dup[float32_cols] = test_dup[float32_cols].astype(np.float32)

    # --- 4. 保存结果 ---
    print('正在保存session-action特征文件...')
    test_dup.to_parquet(output_path + prefix + 'session_use_aid_feat_df.parquet')
    print(f'session-action特征文件已经保存到{output_path + prefix}session_use_aid_feat_df.parquet中.')


if __name__ == '__main__':
    for prefix in ['test_', 'train_']:

        print(f'正在处理{prefix}数据...')



        print(f'--- 正在为 {prefix} 计算加载数据 ---')
        test_actions = load_data(prefix, type='test')
        gc.collect()



        print(f'--- 正在为 {prefix} 计算aid daily特征 ---')
        gen_aid_day_features(prefix, test_actions)
        gc.collect()



        print(f'--- 正在为 {prefix} 计算session特征 ---')
        gen_session_feature(prefix, test_actions)
        del test_actions
        gc.collect()



        print(f'--- 正在为 {prefix} 计算aid 特征 ---')
        test_actions, train_actions = load_data(prefix)
        merge_actions = cudf.concat([test_actions, train_actions])
        del train_actions
        gc.collect()

        gen_aid_feature(prefix, merge_actions, test_actions)
        
        del merge_actions, test_actions
        gc.collect()



        print(f"--- 正在为 {prefix} 计算last-chunk seession-aid 特征 ---")
        test_actions = load_data(prefix, type='test')
        gen_last_chunk_session_aid(prefix, test_actions)
        del test_actions
        gc.collect()



        print(f"--- 正在为 {prefix} 计算user-aid特征 ---")
        test_actions = load_data(prefix, type='test')
        gen_user_aid_feature(prefix, test_actions)
        del test_actions
        gc.collect()


        print(f"--- 正在为 {prefix} 计算session-action特征 ---")
        test_actions = load_data(prefix, type='test')
        gen_session_action_feature(prefix, test_actions)