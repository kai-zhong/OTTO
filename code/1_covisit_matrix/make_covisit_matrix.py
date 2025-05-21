import pandas as pd
import numpy as np
import gc
import cudf
import itertools
from tqdm import tqdm
from cuml.neighbors import NearestNeighbors

cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
dm_save_path = '../../data/feature/'
w2v_path = '../../data/preprocess/'


def load_data(raw_opt_path, preprocess_path, get_type, use_full_data):
    """
    读取训练或测试数据，并根据需求返回指定类型的数据。
    
    参数:
    - raw_opt_path(str): 原始数据存储路径，即完整数据集包括训练集+测试集
    - preprocess_path(str): 预处理数据存储路径，即由原始训练集分为的训练集和验证集
    - get_type(str): 指定返回的数据类型，可选值包括：
        - 'train': 返回训练数据
        - 'test': 返回测试数据
        - 'merge': 返回合并后的训练和测试数据
        - 其他: 返回训练、测试和合并数据三者
    - use_full_data(bool): 指定处理的数据集类型，True 表示完整数据集包括训练集+测试集，False 表示由原始训练集分为的训练集和验证集
    
    返回:
    - 根据 get_type 参数返回对应的 cudf.DataFrame 对象或元组
    """
    train = None
    test = None
    merge = None
    
    # 根据 use_full_data 和 get_type 选择性加载数据
    if use_full_data == True:
        # 从原始数据路径加载完整数据集
        if get_type == 'train':
            train = cudf.read_parquet(raw_opt_path + 'train.parquet')  # 读取训练数据
            train = train.sort_values(['session', 'ts'])  # 按会话ID和时间戳排序
        elif get_type == 'test':
            test = cudf.read_parquet(raw_opt_path + 'test.parquet')   # 读取测试数据
            test = test.sort_values(['session', 'ts'])  # 按会话ID和时间戳排序
        else:
            # 需要 merge 或返回所有数据时，加载完整数据集
            train = cudf.read_parquet(raw_opt_path + 'train.parquet')  # 读取训练数据文件
            test = cudf.read_parquet(raw_opt_path + 'test.parquet')    # 读取测试数据文件
            train = train.sort_values(['session', 'ts'])
            test = test.sort_values(['session', 'ts'])
            merge = cudf.concat([train, test])  # 合并训练和测试数据
    else:
        # 从预处理数据路径加载训练集和验证集
        if get_type == 'train':
            train = cudf.read_parquet(preprocess_path + 'train.parquet')  # 读取训练数据
            train = train.sort_values(['session', 'ts'])  # 按会话ID和时间戳排序
        elif get_type == 'test':
            test = cudf.read_parquet(preprocess_path + 'test.parquet')    # 读取测试（验证）数据
            test = test.sort_values(['session', 'ts'])  # 按会话ID和时间戳排序
        else:
            # 需要 merge 或返回所有数据时，加载完整数据集
            train = cudf.read_parquet(preprocess_path + 'train.parquet')  # 读取训练数据文件
            test = cudf.read_parquet(preprocess_path + 'test.parquet')    # 读取测试（验证）数据文件
            train = train.sort_values(['session', 'ts'])
            test = test.sort_values(['session', 'ts'])
            merge = cudf.concat([train, test])  # 合并训练和测试数据

    # 清理内存，释放不再需要的临时变量
    gc.collect()

    # 根据 get_type 参数返回指定的数据
    if get_type == 'train':
        return train  # 返回训练数据
    elif get_type == 'test':
        return test   # 返回测试数据
    elif get_type == 'merge':
        return merge  # 返回合并后的数据
    else:
        # 如果 get_type 参数不在预期范围内，返回所有数据
        return train, test, merge  # 返回训练、测试和合并数据的元组


def gen_user_behavior_features(data):
    """
    从数据中提取用户行为特征，包括最后一次操作、热门点击、最近一小时和一天的点击数据，以及购物车/订单数据。
    
    参数:
    - data (cudf.DataFrame): 包含 'session'（会话ID）、'ts'（时间戳）、'aid'（商品ID）和 'type'（事件类型）列(使用的是test数据)
    
    返回:
    - data_last_action (cudf.DataFrame): 每个会话的最后一次操作数据, [session, aid]
    - data_top_click (cudf.DataFrame): 每个会话的热门点击数据（对于每个session，取在这个session中被点击的次数排名前30%的aid）, ['session', 'aid']
    - data_last_hour (cudf.DataFrame): 每个会话最近一小时内的交互数据（不含最后一次操作）, [session, aid, ts, type, ts_max, ts_hour, ts_day, ts_week] ts_max（最后一次操作时间戳）、ts_hour（一小时前）、ts_day（一天前）、ts_week（一周前）
    - data_last_day (cudf.DataFrame): 每个会话最近一天内的交互数据（不含最近一小时）, [session, aid, ts, type, ts_max, ts_hour, ts_day, ts_week]
    - data_cart_or_buy (cudf.DataFrame): 每个会话的购物车和订单数据（type != 0）,['session','ts','aid','type']
    (这些返回数据其实需要使用的就只是session和aid列，其他都可以扔掉)
    """
    # === last_action 每个会话的最后一次操作数据 ===
    # 按会话分组，获取每个会话的最大时间戳，用于定位最后一次操作。
    # columns = [session, ts]，ts列是每个session的最大时间戳
    data_max_ts = data.groupby('session')['ts'].max().reset_index()
    
    # 将最大时间戳合并回原始数据，ts_x 是原始时间戳，ts_y 是每个会话的最大时间戳
    # columns = [session, aid, ts_x, type, ts_y]
    data_last_action = data.merge(data_max_ts, on='session', how='left')

    # 筛选出时间戳等于最大时间戳的记录，即最后一次操作
    # 只保留 session 和 aid 列，并去除重复记录，确保每个会话只有一条最后操作记录
    # columns = [session, aid]
    data_last_action = data_last_action[data_last_action['ts_x'] == data_last_action['ts_y']][['session', 'aid']].drop_duplicates().reset_index(drop=True)


    # === top_click 每个会话的热门点击数据（基于点击次数排名前30%） ===
    # 筛选点击事件（type == 0），统计在session中，aid被点击的次数
    data_top_click = data[data['type'] == 0][['session', 'aid']].value_counts().reset_index()
    # 重命名列：session（会话ID）、aid（商品ID）、n（点击次数）
    data_top_click.columns = ['session', 'aid', 'n']
    # 计算每个会话内商品被点击次数的排名百分比（从高到低，pct=True 表示百分比，method='max' 处理并列情况）
    data_top_click['share'] = data_top_click.groupby('session')['n'].rank(ascending=False, pct=True, method='max')
    # 筛选出排名前30%的点击记录（share <= 0.3）
    # 这里可能存在一些问题，就是商品可能被点击得很多，但是因为session交互过的商品少，即时被点击很多了，还是会被舍弃，特别是当session只有一个交互时
    # *****我的想法是把排名改成一个商品点击数占据整个session总点击数的比例来当作权重可能更好*****
    data_top_click = data_top_click[data_top_click['share'] <= 0.3]
    # 计算权重，权重为 1 减去排名百分比（排名越靠前，权重越高）
    data_top_click['weight'] = 1 - data_top_click['share']
    # 只保留 session 和 aid 列，并去除重复记录
    data_top_click = data_top_click[['session', 'aid']].drop_duplicates().reset_index(drop=True)

    # === last1hour 和 last1day 每个会话最近一小时内的交互数据（不含最后一次交互）和 每个会话最近一天内的交互数据（不含最近一小时）===
    # 获取每个会话的最大时间戳，作为时间范围的基准
    # columns = [session, ts]，ts列是每个session的最大时间戳
    last_ts = data.groupby('session')['ts'].max().reset_index()
    # 计算前一小时的时间戳（减去1小时，单位为秒）
    last_ts['ts_hour'] = last_ts['ts'] - (1 * 60 * 60)
    # 计算前一天的时间戳（减去24小时）
    last_ts['ts_day'] = last_ts['ts'] - (24 * 60 * 60)
    # 计算前一周的时间戳（减去7天）
    last_ts['ts_week'] = last_ts['ts'] - (7 * 24 * 60 * 60)
    # 重命名列以便后续使用：ts_max（最后一次操作时间戳）、ts_hour（一小时前）、ts_day（一天前）、ts_week（一周前）
    last_ts.columns = ['session', 'ts_max', 'ts_hour', 'ts_day', 'ts_week']
    # 将时间范围信息合并到原始数据中，how='left' 确保所有原始记录保留
    # columns = [session, aid, ts, type, ts_max, ts_hour, ts_day, ts_week]
    data_last = data.merge(last_ts, on=['session'], how='left')
    # 去除重复记录，确保数据唯一性
    data_last = data_last.drop_duplicates().reset_index(drop=True)

    # 提取最近一小时的点击数据：时间戳在 ts_hour 和 ts_max 之间，但不包括最后一次点击 (ts != ts_max)
    data_last_hour = data_last[(data_last['ts'] >= data_last['ts_hour']) & (data_last['ts'] != data_last['ts_max'])].reset_index(drop=True)
    # 提取最近一天的点击数据：时间戳在 ts_day 和 ts_hour 之间（不含最近一小时）
    data_last_day = data_last[(data_last['ts'] >= data_last['ts_day']) & (data_last['ts'] < data_last['ts_hour'])].reset_index(drop=True)

    # === CartorBuy 每个会话的购物车和订单数据（type != 0）===
    data_cart_or_buy = data[data['type'] != 0].reset_index(drop=True)

    # === 返回所有提取的特征数据 ===
    return data_last_action, data_top_click, data_last_hour, data_last_day, data_cart_or_buy


def compute_cart_cvr(data, chunk=7000):
    """
    计算每个商品 (aid) 的点击到购物车行为的会话级转化率 (CVR)。会话级转化率定义为：点击该商品的会话中是否存在任何商品被加入购物车的比例。
    通过分块处理优化内存使用。（不知道是不是作者写错了，应该考虑的是aid被点击后被加入购物车的比例，后面看着改改）
    
    参数:
    - data (cudf.DataFrame): 合并后的训练和测试数据集，包含 'session'（会话ID）、'aid'（商品ID）、'ts'（操作发生时间戳）和 'type'（事件类型）列
    - chunk (int): 每个分片的aid数量，即每次处理的aid数，分块处理以减少内存压力

    返回:
    - cart_cvr_df (cudf.DataFrame): 包含每个商品的点击会话数、转化会话数和转化率的数据框，列包括 'aid', 'click_n'(商品点击会话数), 'cart_n'(商品加购会话数), 'cart_cvr'(转化率)    
    """
    
    # === 初始化列表 ===
    cart_cvr_df = [] # 存储每个chunk的CVR结果
    # 计算分块数量，基于唯一aid数量除以chunk大小并向上取整
    chunk_num = int(len(data['aid'].drop_duplicates()) / chunk) + 1

    # === 分块处理数据 ===
    for i in tqdm(range(chunk_num)):
        # 计算当前分块的aid范围
        start = i * chunk   # 分块起始aid
        end = (i+1) * chunk # 分块结束aid（不包含）
        
        # 筛选当前分块的点击数据（type == 0 表示点击事件）
        # columns = [session, aid, ts, type]
        row = data[(data['aid'] >= start) & (data['aid'] < end) & (data['type'] == 0)]
        # 将点击数据与购物车数据（type == 1）按会话ID合并，inner join 只保留那些会话中既有点击又有购物车行为的记录（不知道是不是作者写错了，应该考虑的是aid被点击后被加入购物车的比例，后面看着改改）
        # columns = [session(会话号), aid_x(会话中被点击的商品id), ts_x(aid_x被点击的时间戳), type_x(0,点击), aid_y(会话中被加入购物车的商品id), ts_y(aid_y被加入购物车的时间戳), type_y(1,加入购物车)]
        row_cart = row.merge(data[data['type'] == 1], on='session', how='inner')

        # 仅考虑在点击物品后进行的加购行为（但是经过这个筛选之后召回率降了）
        # row_cart = row_cart[row_cart['ts_y'] > row_cart['ts_x']]


        # 计算商品的会话点击总数：每个aid被多少个不同的会话点击过
        # columns = [aid, click_n]
        click_all = row[['aid', 'session']].drop_duplicates()['aid'].value_counts().reset_index()
        # 计算商品的会话购物车总数：对于一个aid，多少个会话存在aid在被点击后，将商品加入购物车的行为（不一定是点击的商品，加购别的商品也算）（基于row_cart中的aid_x）
        # columns = [aid_x, cart_n]
        click_cart = row_cart[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()
        
        # 重命名列以便合并
        click_all.columns = ['aid', 'click_n']  # aid: 商品ID, click_n: 点击会话数
        click_cart.columns = ['aid', 'cart_n']  # aid: 商品ID, cart_n: 转化会话数（会话中有购物车行为）

        # 左连接点击会话数和转化会话数，保留所有点击数据，缺失的转化会话数后续填充
        # columns = [aid, click_n, cart_n]
        click_all = click_all.merge(click_cart, on='aid', how='left')

        # 计算CVR：转化会话数除以点击会话数，表示点击该aid的会话中有购物车行为的比例，结果保留5位小数
        # columns = [aid, click_n, cart_n, 'cart_cvr']
        click_all['cart_cvr'] = (click_all['cart_n'] / click_all['click_n']).round(5)

        # 将当前分块的CVR结果添加到列表
        cart_cvr_df.append(click_all)

    # 合并所有分块的结果为一个DataFrame
    cart_cvr_df = cudf.concat(cart_cvr_df)

    del click_all, click_cart

    # 将cudf DataFrame转换为pandas并填充缺失值（cart_n为NaN的设为0），再转回cudf
    cart_cvr_df = cudf.DataFrame(cart_cvr_df.to_pandas().fillna(0))
    # 将click_n','cart_n'两列数据类型设置为int32，节省内存
    cart_cvr_df[['click_n','cart_n']] = cart_cvr_df[['click_n','cart_n']].astype('int32')

    # 计算所有商品的CVR的平均值，用于后续调整
    mean_cvr = cart_cvr_df['cart_cvr'].mean()
    # 对于点击会话数少于4的aid，CVR调整为原始CVR乘以平均CVR，以提高低频数据的稳定性，其他保持不变
    # 如果一个商品只有 1、2 次点击，但有 1 次加购，看上去 CVR 是 0.5 或 1.0，但这其实不可靠（因为样本太少）。
    # 这种波动性大的 CVR 会干扰模型或排序逻辑
    # 可以改成贝叶斯平滑的处理形式,比如(cart_n + alpha) / (click_n + beta)，alpha和beta可以通过当前数据的极大似然估计得到（问GPT）
    cart_cvr_df['cart_cvr'] = np.where(cart_cvr_df['click_n'].to_pandas() < 4,  
                                        cart_cvr_df['cart_cvr'].to_pandas() * mean_cvr, 
                                        cart_cvr_df['cart_cvr'].to_pandas())

    del data
    gc.collect()

    return cart_cvr_df


def create_covisit_matrix_config():
    """
    创建共现矩阵和datamart的配置字典，用于定义不同的构建模式和参数。
    
    返回:
    - co_dict (dict): 配置字典，键为模式名，值为配置列表。
      每个配置包含：[start_type, end_type, cutline, cut_rank, cut_datamart_last, 
                       cut_datamart_top, cut_datamart_hour, cut_datamart_day, action_pattern_list]

    注释说明：
    该函数封装了共现矩阵和datamart的构建参数，返回一个嵌套字典 co_dict。
    结构：{模式名: [配置列表]}
    每个配置列表包含9个元素：[start_type, end_type, cutline, cut_rank, cut_datamart_last, 
                                cut_datamart_top, cut_datamart_hour, cut_datamart_day, action_pattern_list]
    - start_type (str): 起始事件类型，'click'（点击，type=0）或 'buy'（购买，包括购物车type=1和订单type=2）
    - end_type (str): 结束事件类型，'click' 或 'buy'
    - cutline (int): 高频和低频aid的分界线，用于分割处理（点击或购买次数阈值）
    - cut_rank (int): 共现矩阵中每个aid_x的最大关联aid_y数量（截断排名）aid_x: 共现矩阵的行，表示起始事件的商品 ID。aid_y: 共现矩阵的列，表示与 aid_x 共现的目标商品 ID。
    - cut_datamart_last (int): 'last' datamart的截断排名（最后一次操作）
    - cut_datamart_top (int): 'top' datamart的截断排名（热门点击）
    - cut_datamart_hour (int): 'hour' datamart的截断排名（最近一小时）
    - cut_datamart_day (int): 'day' datamart的截断排名（最近一天）
    - action_pattern_list (list): 指定生成哪些类型的datamart，可选值包括：
      - 'last': 基于最后一次操作
      - 'top': 基于热门点击
      - 'hour': 基于最近一小时
      - 'day': 基于最近一天
      - 'all': 基于所有购买行为（购物车和订单）

    各模式含义：
    - 'allterm': 无时间限制，所有事件对都计入共现矩阵
    - 'dup': 重复事件对计入，不考虑时间顺序
    - 'dup_wlen': 重复事件对计入，权重基于时间差
    - 'dup_hour': 重复事件对计入，限制在一小时内
    - 'base': 基础模式，要求时间顺序（ts_y >= ts_x）
    - 'base_wlen': 基础模式，权重基于时间差
    - 'base_hour': 基础模式，限制在一小时内
    - 'w2v': 使用word2vec方法构建共现矩阵，仅支持点击到点击

    使用场景：
    在 main 函数中调用此函数获取配置字典，然后通过 make_co_matrix 和 make_action_datamart/make_action_hour_day_datamart 函数生成不同模式的共现矩阵和datamart                   
    """
    # 定义共现矩阵和datamart的构建参数字典
    co_dict = {
        # 'allterm' 模式：无时间限制的共现矩阵，考虑所有事件对
        'allterm': [
            # 点击到点击的共现配置
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            # 点击到购买（购物车或订单）的共现配置
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            # 购买到点击的共现配置
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            # 购买到购买的共现配置
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'dup' 模式：重复事件对的共现矩阵，忽略时间顺序
        'dup': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'dup_wlen' 模式：重复事件对的共现矩阵，考虑时间差权重
        'dup_wlen': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'dup_hour' 模式：重复事件对的共现矩阵，限制在一小时内
        'dup_hour': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'hour']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'hour']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'base' 模式：基础共现矩阵，要求时间顺序（ts_y >= ts_x）
        'base': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'base_wlen' 模式：基础共现矩阵，考虑时间差权重
        'base_wlen': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'base_hour' 模式：基础共现矩阵，限制在一小时内
        'base_hour': [
            ['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
            ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
            ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
        ],
        # 'w2v' 模式：基于word2vec的共现矩阵，仅支持点击到点击
        # 因为训练只使用了点击数据来生成商品的embedding，因此只支持点击数据
        'w2v': [
            ['click', 'click', 20, 60, 50, 50, 50, 50, ['last', 'hour']]
        ]
    }

    # 返回配置字典
    return co_dict


def gen_w2v_covisit_matrix(w2v_path, dims=16, knn_num = 60):
    """
    使用预训练的 word2vec 向量构建共现矩阵，基于最近邻算法（KNN）计算商品间的相似性。
    
    参数:
    - w2v_path (str): word2vec 向量文件的存储路径
    - dims (int): 想要使用多少维的 word2vec 向量
    - knn_num (int): 定义最近邻数量（KNN），即每个 aid_x 的最大关联 aid_y 数量

    返回:
    - co_matrix (cudf.DataFrame): 共现矩阵，包含 'aid_x'（源商品ID）、'aid_y'（目标商品ID）和 'share'（余弦距离）
    """
    # === 数据加载 ===
    # 从指定路径加载预训练的 word2vec 向量文件，转换为 cudf DataFrame
    # columns = [aid, vec_1, vec_2, ...]
    w2v = cudf.DataFrame(pd.read_parquet(w2v_path + 'test_' + f'w2v_output_{dims}dims.parquet'))
    # 按 aid（商品ID）排序
    w2v = w2v.sort_values('aid').reset_index(drop=True)

    # === 计算商品最近邻距离 === 
    # 初始化最近邻模型，使用余弦距离作为相似性度量
    model_knn = NearestNeighbors(n_neighbors=knn_num, metric='cosine')
    # 训练模型，输入 word2vec 向量的特征部分（除去 aid 列，从第1列开始）
    model_knn.fit(w2v.iloc[:, 1:])
    # 计算每个 aid 的 KNN 结果，返回距离（distances）和索引（indices）
    # distances: 每个 aid 到其最近邻的余弦距离矩阵(形状为 n x knn_num), distances[i][j] 表示第 i 行样本（第 i 个商品）到它第 j 个最近邻的距离。
    # indices: 每个 aid 的最近邻 aid 索引矩阵(形状为 n x knn_num), indices[i][j] 表示第 i 个商品的第 j 个最近邻在原数据中的行索引。
    distances, indices = model_knn.kneighbors(w2v.iloc[:, 1:])

    # === 构建共现矩阵 ===
    # 创建 aid_x 列：每个 aid 重复 knn_num 次，表示源商品ID，因为aid是从0开始编号且不间断，所以在排序后pd的index就是aid,如果有间断值则需要进行重新映射回aid
    co_matrix = cudf.DataFrame(np.array(([[i] * knn_num for i in range(len(w2v))])).reshape(-1), columns=['aid_x'])
    # 添加 aid_y 列：将 indices 展平为单列，表示目标商品ID，因为aid是从0开始编号且不间断，所以在排序后pd的index就是aid,如果有间断值则需要进行重新映射回aid
    co_matrix['aid_y'] = np.array(indices.to_pandas()).reshape(-1)
    # 添加 dist 列：将 distances 展平为单列，表示 aid_x 和 aid_y 之间的余弦距离
    co_matrix['dist'] = np.array(distances.to_pandas()).reshape(-1)
    # 过滤掉自关联（aid_x == aid_y）的记录，只保留不同商品间的共现关系
    co_matrix = co_matrix[co_matrix['aid_x'] != co_matrix['aid_y']]
    # 重命名列：'dist' 改为 'share'，表示相似性得分（实际是余弦距离，越小越相似）（余弦距离=1-余弦相似度,取值范围为[0,2],越小越相似）
    co_matrix.columns = ['aid_x', 'aid_y', 'share']
    # 转换数据类型为 32 位整数，优化内存使用
    co_matrix['aid_x'] = co_matrix['aid_x'].astype(np.int32)
    co_matrix['aid_y'] = co_matrix['aid_y'].astype(np.int32)
    # 按 aid_x 升序、share 升序（距离从小到大）、aid_y 升序排序，确保结果有序
    co_matrix = co_matrix.sort_values(['aid_x', 'share', 'aid_y'], ascending=[True, True, True])

    del distances, indices
    gc.collect()
    return co_matrix


def split_high_low_frequency_aids(pattern, data, start_type, end_type, chunk=20000, cutline=20):
    """
    根据指定模式和事件类型统计商品 (aid) 的共现次数，并分割为高频和低频两部分。
    共现次数统计：比如对于click-click模式来说，比如物品a在session 1、2、3中出现，session 1、2、3中点击的物品数量(包括a)分别是5,6,7，那么物品a的n就等于4+5+6
    参数:
    - pattern (str): 共现矩阵构建模式，如 'allterm', 'dup', 'base' 等，影响事件筛选逻辑
    - data (cudf.DataFrame): 合并后的训练和测试数据，包含 'session', 'aid', 'type', 'ts' 列
    - start_type (str): 起始事件类型，'click'（type=0）或 'buy'（type=1或2）
    - end_type (str): 结束事件类型，'click' 或 'buy'
    - chunk (int, optional): 分块处理的大小，默认 20000，用于优化内存
    - cutline (int, optional): 高频和低频 aid 的分界线，默认 20，基于共现次数
    
    返回:
    - low_count_aids (list): 低频 aid 列表（共现次数 < cutline）
    - high_count_aids (list): 高频 aid 列表（共现次数 >= cutline）
    - aid_count_df (cudf.DataFrame): 所有 aid 的共现次数统计，包含 'aid' 和 'n'(共现次数，即物品a) 列
    """
    # === 初始化列表，用于存储中间结果 ===
    # 存储每个分块的 aid 计数结果
    aid_count_df = []
    # 计算分块数量，基于唯一 aid 数量除以 chunk 大小并向上取整
    chunk_num = int(len(data['aid'].drop_duplicates()) / chunk) + 1

    # === 分块处理数据，统计 aid 的共现次数 ===
    print(f'正在划分{pattern}模式{start_type}-{end_type}类型高低频商品...')
    for i in tqdm(range(chunk_num)):
        # 计算当前分块的 aid 范围
        start = i * chunk  # 分块起始 aid
        end = (i + 1) * chunk  # 分块结束 aid（不包含）
        # 根据 start_type 筛选起始事件数据
        # row 的 columns = ['session', 'aid', 'type', 'ts']
        if start_type == 'click':
            # 筛选点击事件（type == 0）
            row = data[(data['aid'] >= start) & (data['aid'] < end) & (data['type'] == 0)]
        else:
            # 筛选购买事件（type != 0，包括购物车和订单）
            row = data[(data['aid'] >= start) & (data['aid'] < end) & (data['type'] != 0)]
        
        # 清理内存
        gc.collect()

        # 根据 end_type 合并结束事件数据
        if end_type == 'click':
            # 合并点击事件（type == 0），按 session 做 inner join
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            # _x为start_type，_y为end_type
            row = row.merge(data[data['type'] == 0], on='session', how='inner')
        else:
            # 合并购买事件（type != 0），按 session 做 inner join
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            # _x为start_type，_y为end_type
            row = row.merge(data[data['type'] != 0], on='session', how='inner')
        
        # 根据 pattern 应用不同的筛选规则
        if pattern == 'allterm':
            # 无时间限制，仅保留唯一的三元组
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif pattern == 'base':
            # 要求时间顺序（仅保留 ts_y >= ts_x 的行），去重
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif pattern == 'base_hour':
            # 要求时间顺序且在一小时内（ts_y >= ts_x 且 ts_y - ts_x <= 3600秒），去重
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif pattern == 'base_wlen':
            # 要求时间顺序（仅保留 ts_y >= ts_x 的行），保留时间差信息
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            row = row[row['ts_y'] - row['ts_x'] >= 0]
        elif pattern == 'dup':
            # 无额外筛选，保留所有记录
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            pass
        elif pattern == 'dup_wlen':
            # 无额外筛选，保留所有记录（后续可能加权重）
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            pass
        elif pattern == 'dup_hour':
            # 要求时间顺序且在一小时内
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
        else:
            # 默认无操作，可能是未实现的模式
            pass

        # 统计当前分块中 aid_x 的出现次数，比如对于click-click模式来说，比如物品a在session 1、2、3中出现，session 1、2、3中点击的物品数量(包括a)分别是5,6,7，那么物品a的n就等于4+5+6
        # columns = [aid_x, cnt]
        aid_count = row['aid_x'].value_counts().reset_index()
        # 重命名列为 'aid' 和 'n'（共现次数）
        aid_count.columns = ['aid', 'n']
        # 将结果添加到列表
        aid_count_df.append(aid_count)
    
    # 合并所有分块的 aid 计数结果
    # columns = [aid, n]
    aid_count_df = cudf.concat(aid_count_df)

    # 删除临时变量以释放内存
    del row, aid_count
    gc.collect()

    # 根据 cutline 分割高频和低频 aid
    low_count_aids = list(aid_count_df[aid_count_df['n'] < cutline]['aid'].to_pandas())  # 低频 aid 列表
    high_count_aids = list(aid_count_df[aid_count_df['n'] >= cutline]['aid'].to_pandas())  # 高频 aid 列表
 
    print(f"低频 aid 的数量: {len(low_count_aids)}, 高频 aid 的数量: {len(high_count_aids)}")

    return low_count_aids, high_count_aids, aid_count_df


def gen_chunk_co_matrix(pattern, data, use_aids, 
                        cart_cvr_df, start_type, end_type, 
                        chunk, same_col_share_name, cut_rank):
    """
    分块生成共现矩阵和自关联特征，根据指定模式处理商品 (aid) 的共现关系。
    
    参数:
    - pattern (str): 共现矩阵构建模式，如 'allterm', 'dup', 'base' 等，影响事件筛选和权重逻辑
    - data (cudf.DataFrame): 合并后的训练和测试数据，包含 'session', 'aid', 'type', 'ts' 列
    - use_aids (list): 需要处理的 aid 列表（高频或低频）
    - cart_cvr_df (cudf.DataFrame): 点击到购物车的转化率数据，包含 'aid', 'click_n'(商品点击会话数), 'cart_n'(商品加购会话数), 'cart_cvr'(转化率)  列
    - start_type (str): 起始事件类型，'click'（type=0）或 'buy'（type=1或2）
    - end_type (str): 结束事件类型，'click' 或 'buy'
    - chunk (int): 分块处理的大小，用于优化内存
    - same_col_share_name (str): 自关联特征的列名，例如 'same_click_click_allterm'
    - cut_rank (int): 每个 aid_x 的最大关联 aid_y 数量，截断排名
    
    返回:
    - co_matrix_df (cudf.DataFrame): 共现矩阵，包含 'aid_x', 'aid_y', 'share'(n(aid_x与aid_y共现的次数或权重和) 除以 aid_x 的共现总次数), 'cart_cvr'(aid_y 的转化率), 'rank'（rank列可以理解为每个aid_y对于aid_x的重要程度，越重要排越前面） 列
    - co_matrix_same_df (cudf.DataFrame): 自关联特征数据，包含 'aid' 和 same_col_share_name(比如aid被点击到被购买的总次数same-click-buy-[pattern]) 列
    """
    # === 初始化列表，用于存储每个分块的共现矩阵和自关联特征 ===
    co_matrix_df = []
    co_matrix_same_df = []
    num_covisit = 0 # 累计共现记录数
    num_same = 0    # 累计自关联记录数
    chunk_num = int(len(use_aids) / chunk) + 1 # 分块数量

    # === 分块处理数据 ===
    for i in tqdm(range(chunk_num)):
        # 计算当前分块的 aid 范围
        start = i * chunk  # 分块起始索引
        end = (i + 1) * chunk  # 分块结束索引（不包含）
        # 根据 start_type 筛选起始事件数据
        if start_type == 'click':
            # 筛选点击事件（type == 0），限制在当前分块的 use_aids 内
            row = data[(data['aid'].isin(use_aids[start:end])) & (data['type'] == 0)]
        else:
            # 筛选购买事件（type != 0），限制在当前分块的 use_aids 内
            row = data[(data['aid'].isin(use_aids[start:end])) & (data['type'] != 0)]
        # 清理内存
        gc.collect()
        # 根据 end_type 合并结束事件数据
        if end_type == 'click':
            # 合并点击事件（type == 0），按 session 内联
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            row = row.merge(data[data['type'] == 0], on='session', how='inner')
        else:
            # 合并购买事件（type != 0），按 session 内联
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y']
            row = row.merge(data[data['type'] != 0], on='session', how='inner')


        # === 根据 pattern 应用不同的共现计算规则 ===
        if pattern == 'allterm':
            # 无时间限制，去重后统计 aid_x 和 aid_y 的共现次数,并按照aid_x排序
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            # columns = ['aid_x', 'aid_y', 'count'(aid_x和aid_y总的共现次数)]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif pattern == 'base':   
            # 要求时间顺序（ts_y >= ts_x），去重后统计共现次数,并按照aid_x排序
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            # columns = ['aid_x', 'aid_y', 'count'(aid_x和aid_y总的共现次数)]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
        
        elif pattern == 'base_wlen':
            # 要求时间顺序（ts_y >= ts_x），不去重，计算时间差权重
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            # 计算时间差绝对值
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y', 'ts_diff']
            row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])  
            # 按时间差排名
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y', 'ts_diff', 'diff_rank']
            row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method='min')  
            row['diff_weight'] = 1 / row['diff_rank']  # 权重为排名的倒数
            # columns = ['session', 'aid_x', 'type_x', 'ts_x', 'aid_y', 'type_y', 'ts_y', 'ts_diff', 'diff_weight']
            del row['ts_diff'], row['diff_rank']  # 删除临时列
            # columns = ['aid_x', 'aid_y','diff_weight'] -> ['aid_x', 'aid_y','count'(pair(aid_x-aid_y)的权重和)]
            row = row.groupby(['aid_x', 'aid_y'])['diff_weight'].sum().reset_index()  # 按权重求和
            
        elif pattern == 'base_hour':
            # 要求时间顺序且在一小时内（0 <= ts_y - ts_x <= 3600秒），去重后统计共现次数
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            # columns = ['session', 'aid_x', 'aid_y']
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            # columns = ['aid_x', 'aid_y', 'count'(aid_x和aid_y总的共现次数)]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif pattern == 'dup':
            # 无时间限制，不去重，直接统计共现次数
            # columns = ['aid_x', 'aid_y', 'count'(aid_x和aid_y总的共现次数)]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif pattern == 'dup_wlen':
            # 计算时间差权重，不去重，无时间顺序要求
            row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])  # 计算时间差绝对值
            row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method='min')  # 按时间差排名
            row['diff_weight'] = 1 / row['diff_rank']  # 权重为排名的倒数
            del row['ts_diff'], row['diff_rank']  # 删除临时列
            # columns = ['aid_x', 'aid_y','diff_weight'] -> ['aid_x', 'aid_y','count'(pair(aid_x-aid_y)的权重和)]
            row = row.groupby(['aid_x', 'aid_y'])['diff_weight'].sum().reset_index()  # 按权重求和
        
        elif pattern == 'dup_hour':
            # 要求时间顺序且在一小时内，不去重，统计共现次数
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        # 重命名列：aid_x, aid_y, n（共现次数或权重和）
        row.columns = ['aid_x', 'aid_y', 'n']
        # 计算每个 aid_x 的总共现次数
        # columns = ['aid_x', 'aid_total'（aid_x的共现总次数）]
        aid_x_total = row.groupby(['aid_x'])['n'].sum().reset_index()
        aid_x_total.columns = ['aid_x', 'aid_total']
        # 合并总次数到 row
        # columns = [aid_x, aid_y, n, aid_total]
        row = row.merge(aid_x_total, on='aid_x', how='left')

        # 删除临时变量并清理内存
        del aid_x_total
        gc.collect()

        # === 计算 share（共现比例），n(aid_x与aid_y共现的次数或权重和) 除以 aid_total(aid_x 的共现总次数) ===
        # columns = [aid_x, aid_y, n, aid_total, share]
        row['share'] = row['n'] / row['aid_total']
        row['share'] = row['share'].astype(np.float32)

        # === 添加 aid_y 的转化率 ===
        # columns = [aid_x, aid_y, share, aid, cart_cvr]
        row = row[['aid_x', 'aid_y', 'share']].merge(cart_cvr_df[['aid', 'cart_cvr']], 
                                                     left_on=['aid_y'], right_on=['aid'], how='left')
        # 按 aid_x 升序、share 降序（得分高优先）、cart_cvr 降序（转化率高优先）、aid_y 升序
        # columns = [aid_x, aid_y, share, cart_cvr]
        row = row[['aid_x', 'aid_y', 'share', 'cart_cvr']].sort_values(
            ['aid_x', 'share', 'cart_cvr', 'aid_y'], ascending=[True, False, False, True]).reset_index(drop=True)

        # === 添加排名列 ===
        # rank列可以理解为每个aid_y对于aid_x的重要程度，越重要排越前面
        # columns = [aid_x, aid_y, share, cart_cvr, rank]
        row['rank'] = 1
        row['rank'] = row.groupby('aid_x')['rank'].cumsum()  # 计算每个 aid_x 的累积排名
        row['rank'] = row['rank'].astype(np.int32)  # 转换为 32 位整数

        # 提取自关联记录（aid_x == aid_y）
        # columns = [aid_x, aid_y, share, cart_cvr, rank]
        row_same_aid = row[row['aid_x'] == row['aid_y']] 
        # 过滤掉自关联记录
        # columns = [aid_x, aid_y, share, cart_cvr, rank]
        row = row[row['aid_x'] != row['aid_y']] 
        # 截断排名，每个aid_x 只保留前 cut_rank 个 aid_y
        row = row[row['rank'] <= cut_rank]

        num_covisit += len(row)
        num_same += len(row_same_aid)
        # print(f'{pattern}|{start_type}-{end_type}模式下已生成{num_covisit}条共现记录和{num_same}条自关联记录.')

        # 将当前分块的共现矩阵和自关联特征添加到列表
        co_matrix_df.append(row)
        co_matrix_same_df.append(row_same_aid)

    print(f'正在合并所有分块的共现矩阵...')
    # 合并所有分块的共现矩阵
    # columns = [aid_x, aid_y, share, cart_cvr, rank]
    co_matrix_df = cudf.concat(co_matrix_df)
    print(f'正在合并所有分块的自关联特征...')
    # 合并所有分块的自关联特征
    # columns = [aid_x, aid_y, share, cart_cvr, rank]
    co_matrix_same_df = cudf.concat(co_matrix_same_df)
    # 精简自关联数据，只保留 aid_x 和 share
    # columns = [aid_x, share]
    co_matrix_same_df = co_matrix_same_df[['aid_x', 'share']]
    # 重命名列为 'aid' 和 same_col_share_name
    # columns = [aid_x, same_col_share_name]
    co_matrix_same_df.columns = ['aid', same_col_share_name]
    # 转换 share 列为 32 位浮点数
    co_matrix_same_df[same_col_share_name] = co_matrix_same_df[same_col_share_name].astype(np.float32)
    
    # 返回共现矩阵和自关联特征
    return co_matrix_df, co_matrix_same_df


def aug_data(co_matrix_df, cart_cvr_df, base_co_matrix_df):
    """
    对共现矩阵进行数据增强，通过高关联商品扩展共现关系，并重新计算 share。
    扩展共现关系,如果一个aid_x的共现对象aid_y是一个高关联商品（与另外一个aid_z强相关）,那么再构造一个(aid_x, aid_z)的共现对
    新构造的共现对的共现强度share等于aid_x-aid_y的共现强度share_xy乘上aid_y-aid_z的共现强度share_yz

    参数:
    - co_matrix_df (cudf.DataFrame): 输入的共现矩阵，包含 'aid_x', 'aid_y', 'share'(共现强度), 'cart_cvr'(aid_y 的转化率), 'rank'（rank列可以理解为每个aid_y对于aid_x的重要程度，越重要排越前面） 列
    - cart_cvr_df (cudf.DataFrame): 点击到购物车的转化率数据，包含 'aid', 'click_n'(商品点击会话数), 'cart_n'(商品加购会话数), 'cart_cvr'(转化率)  列
    - base_co_matrix_df (cudf.DataFrame): 基础共现矩阵（未经增强），用于提取高关联商品
    
    返回:
    - co_matrix_df (cudf.DataFrame): 增强后的共现矩阵，包含 共现矩阵，包含 'aid_x', 'aid_y', 'share', 'cart_cvr' 列
    """
    # 转换数据类型为 32 位浮点数
    co_matrix_df['share'] = co_matrix_df['share'].astype(np.float32)
    co_matrix_df['cart_cvr'] = co_matrix_df['cart_cvr'].astype(np.float32)

    # === 提取高关联商品和需要增强的基础数据 ===
    # 获取高关联商品：从 base_co_matrix_df 中筛选 rank <= 2 的记录
    # columns = ['aid_x', 'aid_y', 'share', 'cart_cvr', 'rank']
    high_asso_df = base_co_matrix_df[base_co_matrix_df['rank'] <= 2]
    
    # 从 co_matrix_df 中筛选 rank <= 3 的记录，作为增强的基础数据
    # columns = ['aid_x', 'aid_y', 'share', 'cart_cvr', 'rank']
    high_rank_df = co_matrix_df[co_matrix_df['rank'] <= 3]

    # 删除 co_matrix_df 中的 rank 列
    del co_matrix_df['rank']
    # 清理内存
    gc.collect()

    # === 扩展共现关系 ===
    # 通过 aid_y 和 aid_x 合并，扩展共现关系
    # columns = [aid_x_x, aid_y_x, share_x, cart_cvr, rank, aid_x_y, aid_y_y, share_y]
    # aid_x_x	原始 aid_x，即最终增强的关系左边商品
    # aid_y_x	原始 aid_y，即中间桥梁商品
    # share_x	原始 aid_x → aid_y_x 的共现强度
    # cart_cvr	aid_y_x 的转化率
    # rank	    原始共现对的 rank
    # aid_x_y	中间桥梁商品（与 aid_y_x 相同），用于 join
    # aid_y_y	新的扩展目标商品，即：从 aid_y_x 的 perspective 找到的 aid_y_y
    # share_y	aid_y_x → aid_y_y 的共现 share
    aug_data = high_rank_df.merge(high_asso_df[['aid_x', 'aid_y', 'share']],
                                    left_on=['aid_y'], 
                                    right_on=['aid_x'], how='inner').sort_values(['aid_x_x'])
    # 过滤掉自关联记录（aid_x_x == aid_y_y）
    aug_data = aug_data[aug_data['aid_x_x'] != aug_data['aid_y_y']]
    # 仅保留 aid_x_x, aid_y_y, share_x, share_y
    # columns = [aid_x_x, aid_y_y, share_x, share_y]
    aug_data = aug_data[['aid_x_x', 'aid_y_y', 'share_x', 'share_y']]
    # 计算新的 share：share_x 和 share_y 的乘积，四舍五入到 5 位小数
    # columns = [aid_x_x, aid_y_y, share_x, share_y, share]
    aug_data['share'] = (aug_data['share_x'] * aug_data['share_y']).round(5)
    # columns = [aid_x_x, aid_y_y, share]
    aug_data = aug_data[['aid_x_x', 'aid_y_y', 'share']]
    # 重命名列为标准格式
    aug_data.columns = ['aid_x', 'aid_y', 'share']
    # 按 aid_x 和 aid_y 分组，取 share 的总和（合并重复记录）
    # columns = [aid_x_x, aid_y_y, share]
    aug_data = aug_data.groupby(['aid_x', 'aid_y'])['share'].sum().reset_index()

    # 合并原始共现矩阵和增强数据，仅保留 aid_x, aid_y, share 列
    # columns = [aid_x_x, aid_y_y, share]
    co_matrix_df = cudf.concat([co_matrix_df[['aid_x', 'aid_y', 'share']], aug_data[['aid_x', 'aid_y', 'share']]])
    # 删除临时增强数据
    del aug_data
    # 清理内存
    gc.collect()

    # 按 aid_x 和 aid_y 分组，取 share 的最大值（处理重复记录）
    # columns = [aid_x_x, aid_y_y, share]
    co_matrix_df = co_matrix_df.groupby(['aid_x', 'aid_y'])['share'].max().reset_index()
    # 将增强后的共现矩阵与 cart_cvr_df 关联，添加 aid_y 的转化率
    # columns = [aid_x_x, aid_y_y, share, aid, cart_cvr]
    co_matrix_df = co_matrix_df.merge(cart_cvr_df[['aid', 'cart_cvr']], left_on=['aid_y'], 
                                      right_on=['aid'], how='left')
    
    # 重置索引，删除冗余的 aid 列
    co_matrix_df = co_matrix_df.reset_index(drop=True)
    co_matrix_df.drop(columns=['aid'], inplace=True)

    # 返回增强后的共现矩阵
    return co_matrix_df


def gen_co_matrix(use_full_data, pattern, cart_cvr_df, 
                  start_type, end_type, same_col_share_name, 
                   chunk=7000, cutline=20, cut_rank=151):
    """
    根据指定模式生成共现矩阵，将商品分为高频和低频两部分处理，并合并结果。
    
    参数:
    - use_full_data (bool): 数据集类型
    - pattern (str): 共现矩阵构建模式，如 'allterm', 'dup', 'base' 等，影响事件筛选逻辑
    - cart_cvr_df (cudf.DataFrame): 点击到购物车的转化率数据，包含 'aid', 'click_n'(商品点击会话数), 'cart_n'(商品加购会话数), 'cart_cvr'(转化率)  列
    - start_type (str): 起始事件类型，'click'（type=0）或 'buy'（type=1或2）
    - end_type (str): 结束事件类型，'click' 或 'buy'
    - same_col_share_name (str): 自关联特征的列名，例如 'same_click_click_allterm'
    - chunk (int, optional): 分块处理的大小，默认 7000，用于优化内存
    - cutline (int, optional): 高频和低频 aid 的分界线，默认 20，基于事件计数
    - cut_rank (int, optional): 每个 aid_x 的最大关联 aid_y 数量，默认 151
    
    返回:
    - _co_matrix (cudf.DataFrame): 合并后的共现矩阵，包含 'aid_x', 'aid_y', 'share'（共现强度）, 'cart_cvr'（aid_y的转化率） 列
    - same_aids_features (cudf.DataFrame): 合并后的自关联特征数据，包含 'aid' 和 same_col_share_name 列
    """
    # === 读取合并后的训练和测试数据 ===
    merge = load_data(raw_opt_path, preprocess_path, get_type='merge', use_full_data=use_full_data)

    # === 划分高频和低频 aid ===
    low_count_aids, high_count_aids, _ = split_high_low_frequency_aids(pattern, merge, start_type, 
                                                                       end_type, chunk, cutline)

    # === 处理高频aid的共现矩阵 ===
    print('正在生成高频商品共现矩阵和自关联特征...')
    # high_co_matrix: 高频 aid 的共现矩阵 columns = ['aid_x', 'aid_y', 'share'(共现强度), 'cart_cvr'(aid_y 的转化率), 'rank'（rank列可以理解为每个aid_y对于aid_x的重要程度，越重要排越前面，按shared降序排序，再按cart_cvr降序排列）] 
    # high_same_aids_features: 高频 aid 的自关联特征 columns = ['aid', same_col_share_name(比如aid被点击到被购买的总次数或权重和，相当于共现矩阵中的share项，same-click-buy-[pattern])]
    high_co_matrix, high_same_aids_features = gen_chunk_co_matrix(pattern, merge, high_count_aids, 
                                                         cart_cvr_df, start_type, end_type, 
                                                         chunk, same_col_share_name, cut_rank)
    # 保存高频共现矩阵副本，用于后续数据增强
    # columns = ['aid_x', 'aid_y', 'share', 'cart_cvr', 'rank'] 
    base_aug = high_co_matrix.copy()
    del merge  
    gc.collect()

    # 对高频共现矩阵进行数据增强
    print('正在对高频共现矩阵进行数据增强...')
    # columns = ['aid_x', 'aid_y', 'share', 'cart_cvr']
    high_co_matrix = aug_data(high_co_matrix, cart_cvr_df, base_aug)


    # === 处理低频aid的共现矩阵 ===
    print('正在生成低频商品共现矩阵和自关联特征...')
    merge = load_data(raw_opt_path, preprocess_path, get_type='merge', use_full_data=use_full_data)
    # low_co_matrix: 低频 aid 的共现矩阵 columns = ['aid_x', 'aid_y', 'share'(共现强度), 'cart_cvr'(aid_y 的转化率), 'rank'] 
    # low_same_aids_features: 低频 aid 的自关联特征 columns = ['aid', same_col_share_name]
    low_co_matrix, low_same_aids_features = gen_chunk_co_matrix(pattern, merge, low_count_aids, 
                                                       cart_cvr_df, start_type, end_type, 
                                                       chunk, same_col_share_name, cut_rank)

    del merge
    gc.collect()

    # 对低频共现矩阵进行数据增强
    print('正在对低频共现矩阵进行数据增强...')
    # columns = ['aid_x', 'aid_y', 'share', 'cart_cvr']
    low_co_matrix = aug_data(low_co_matrix, cart_cvr_df, base_aug)

    del base_aug
    gc.collect()

    print('正在合并共现矩阵与自关联特征...')
    # 合并高频和低频共现矩阵
    co_matrix = cudf.concat([high_co_matrix, low_co_matrix])
    
    del low_co_matrix, high_co_matrix
    gc.collect()

    # 转换 share 和 cart_cvr 列为 32 位浮点数
    co_matrix['share'] = co_matrix['share'].astype(np.float32)
    co_matrix['cart_cvr'] = co_matrix['cart_cvr'].astype(np.float32)

    # 合并高频和低频自关联特征
    same_aids_features = cudf.concat([high_same_aids_features, low_same_aids_features])

    print(f'{pattern}|{start_type}-{end_type}模式共现矩阵，自关联特征生成完毕.')

    return co_matrix, same_aids_features


def gen_action_datamart(co_matrix, session_action_df, co_mat_feature_name,
                         rank, dm_save_path, prefix, w2v = False):
    """
    根据共现矩阵和用户行为数据生成datamart（推荐候选集），并保存为 parquet 文件。
    
    参数:
    - co_matrix (cudf.DataFrame): 共现矩阵，包含 'aid_x', 'aid_y', 'share'(共现强度)（非 w2v 模式还包含 'cart_cvr'）
    - session_action_df (cudf.DataFrame): 用户行为数据，包含 'session' 和 'aid' 列（如 data_last_action）
    - co_mat_feature_name (str): 数据市场特征名，例如 'click_click_allterm_last'，就是根据最后一次点击（last）的aid_x的allterm模式的共现矩阵去寻找这个session下一个可能出现的点击商品
    - rank (int): 每个 session 的最大推荐商品数量（截断排名）
    - dm_save_path (str): 数据市场保存路径
    - prefix (str): 文件名前缀，'train_' 或 'test_'
    - w2v (bool, optional): 是否使用 word2vec 模式，默认 False，影响聚合和排序逻辑
    
    返回:
    - 无返回值，直接将结果保存为 parquet 文件
        parquet 文件包含以下列：
        - session: 会话 ID
        - aid: 推荐的候选商品 ID（来自 aid_y）
        - {co_mat_feature_name}: 推荐得分
        - rank: 候选商品在 session 中的推荐顺序（从 1 开始）
    """
    # 创建用户行为数据的副本，避免修改原始数据
    action_df = session_action_df.copy()
    del session_action_df
    gc.collect()

    # === 数据处理（生成推荐） ===
    # 非 word2vec 模式：基于常规共现矩阵
    if w2v == False:
        # 将用户行为数据与共现矩阵按 aid 和 aid_x 合并，inner join 只保留存在匹配的记录
        # 等于是将用户行为数据与共现矩阵连接，找出每个用户点击过的商品的相似商品。
        # 比如最后一次点击的数据，那么aid_x就是用户最后一次点击的商品，就可以根据最后一次点击的商品的共现商品的共现强度，给用户推荐
        # columns = [session, aid, aix_x, aid_y, share, cart_cvr]
        action_df = action_df.merge(co_matrix, left_on='aid', right_on='aid_x', how='inner')
        # 按 session 和 aid_y 分组，聚合 share（共现得分）和 cart_cvr（转化率）
        # 对每个 session 和候选商品 aid_y 聚合：
        #   - share：共现强度之和，表示该推荐商品与该 session 所有点击商品的“总关联强度”。
        #   - cart_cvr：平均加购转化率，表示该商品的潜在吸引力。
        # 最终得到的 `(session, aid_y)` 表格即可用于排序推荐。
        # columns = [session, aid_y, share(与所有邻居aid_x的share的和), cart_cvr(转化率的均值)]
        action_df = action_df.groupby(['session', 'aid_y']).agg({'share': 'sum', 'cart_cvr': 'mean'}).reset_index()
        # 排序：按 session 升序、share 降序（得分高优先）、cart_cvr 降序（转化率高优先）、aid_y 升序
        # 对推荐候选结果排序，为每个用户挑选得分最高的商品作为最终推荐。
        action_df = action_df.sort_values(['session', 'share', 'cart_cvr', 'aid_y'], 
                                          ascending=[True, False, False, True]).reset_index(drop=True)
    else:
        # word2vec 模式：基于余弦距离的共现矩阵
        # 将用户行为数据与共现矩阵按 aid 和 aid_x 合并
        # columns = [session, aid, aix_x, aid_y, share]
        action_df = action_df.merge(co_matrix, left_on='aid', right_on='aid_x', how='inner')
        # 按 session 和 aid_y 分组，计算 share（余弦距离）的平均值
        # columns = [session, aid_y, share]
        action_df = action_df.groupby(['session', 'aid_y'])['share'].mean().reset_index()
        # 排序：按 session 升序、share 升序（余弦距离小优先，表示相似度高），重置索引
        action_df = action_df.sort_values(['session', 'share']).reset_index(drop=True)
        
    # === 每个用户的推荐商品进行排名并截断 ===
    # 仅保留 session、aid_y 和 share 列
    # columns = [session, aid_y, share]
    action_df = action_df[['session', 'aid_y', 'share']]
    # 添加排名列，初始值为 1
    action_df['rank'] = 1
    # 计算每个 session 的累积排名，表示推荐商品的顺序
    action_df['rank'] = action_df.groupby('session')['rank'].cumsum()
    # 转换排名列为 32 位整数
    action_df['rank'] = action_df['rank'].astype(np.int32)
    # 筛选排名小于等于指定 rank 的记录，截断每个 session 的推荐商品数量，仅保留为每个session推荐的前rank个商品
    action_df = action_df[action_df['rank'] <= rank]

    # 重命名列：
    # - 'session': 会话ID
    # - 'aid_y': 推荐商品ID，改为 'aid'
    # - 'share': 共现得分或余弦距离，改为 co_mat_feature_name
    # - 'rank': 推荐排名
    action_df.columns = ['session', 'aid', co_mat_feature_name, 'rank']
    # 转换为 pandas DataFrame 并保存为 parquet 文件
    # 文件名例如test_click_click_allterm_last.parquet
    print(f'数据已保存到{dm_save_path + prefix + co_mat_feature_name}.parquet')
    action_df.to_pandas().to_parquet(dm_save_path + prefix + co_mat_feature_name + '.parquet')
    
    del action_df
    gc.collect()


def gen_action_hour_day_datamart(co_matrix, session_action_df, co_mat_feature_name, cut_datamart, dm_save_path, prefix, w2v = False):
    """
    函数功能：
        基于共现矩阵和用户行为数据，生成用于推荐任务的候选集 datamart。
        函数将数据按 session 分块处理，计算推荐得分（如 share 总和或均值），保留 top-N 推荐结果，并保存为 parquet 格式。

    参数：
        co_matrix(cudf.DataFrame): 共现矩阵
                - 'aid_x': 当前商品 ID
                - 'aid_y': 关联商品 ID
                - 'share': 共现强度
                - 'cart_cvr（w2v=False时存在）': 转化率
        session_action_df(cudf.DataFrame): 行为数据，包含列：
                - 'session': 用户的 session ID
                - 'aid': 用户行为中涉及的商品 ID
        co_mat_feature_name(str):保存 datamart 特征的名称，用作输出文件中该列的列名。
        cut_datamart(int):每个 session 保留的候选商品数量上限（截断 rank）。
        dm_save_path(str):输出 parquet 文件保存路径（需以 / 结尾）。
        prefix(str): 文件名前缀，通常用于标识当前 datamart 来源（如 'click_'、'cart_' 等）。
        w2v(bool):是否使用 word2vec 模式。如果为 True，则对 share 使用 mean 聚合，排序规则相反（值小越靠前）。

    返回值：
        将生成的 datamart 保存为一个 parquet 文件
            parquet 文件包含以下列：
            - session: 会话 ID
            - aid: 推荐的候选商品 ID（来自 aid_y）
            - {co_mat_feature_name}: 推荐得分(原share经过处理得到)
            - rank: 候选商品在 session 中的推荐顺序（从 1 开始）
    """
    # 每次处理 20000 个 session
    chunk = 20000
    # 计算 chunk 的数量
    chunk_num = int(len(session_action_df['session'].drop_duplicates()) / chunk) + 1

    # 用于存储每个 chunk 生成的 datamart 结果
    datamart_list = []
    # 获取所有 session 的列表（转为 pandas 格式，便于切片）
    session_list = list(session_action_df['session'].unique().to_pandas())

    # 遍历每一个 chunk，逐块生成 datamart
    for i in tqdm(range(chunk_num)):
        # 当前 chunk 的起止位置
        start = i * chunk
        end = (i + 1) * chunk

        # 选出当前 chunk 中的 session 对应的行为数据
        # columns = [session, aid, aid_x, aid_y, share, cart_cvr(可选)]
        row = session_action_df[session_action_df['session'].isin(session_list[start:end])].merge(
            co_matrix, left_on = 'aid', right_on = 'aid_x'
        )

        if w2v == False:
            # 非 w2v 模式：按 session 和 aid_y 分组，统计 share（共现强度）的总和
            # columns = [session, aid_y, share]
            row = row.groupby(['session', 'aid_y'])['share'].sum().reset_index()
            # 按 session 升序，share 降序排序（即每个 session 中得分高的排前面）
            row = row.sort_values(['session', 'share'], ascending = [True, False])
        else:
            # w2v 模式：按 session 和 aid_y 分组，统计 share（余弦距离） 的平均值
            # columns = [session, aid_y, share]
            row = row.groupby(['session', 'aid_y'])['share'].mean().reset_index()
            # 按 session 和 share 升序排序（w2v 情况下share代表余弦距离，越小越好）
            row = row.sort_values(['session', 'share']).reset_index(drop=True)

        # 添加排名列，初始值设为 1
        # columns = [session, aid_y, share, rank]
        row['rank'] = 1
        # 对每个 session 内进行累计排名（cumsum），表示推荐顺序
        row['rank'] = row.groupby('session')['rank'].cumsum()
        # 筛选排名在 cut_datamart 以内的商品，即只保留前 cut_datamart 个推荐项
        row = row[row['rank'] <= cut_datamart]

        # 将当前 chunk 的结果加入总 datamart 列表中
        datamart_list.append(row)
    
    # 合并所有 chunk 生成的 datamart
    datamart = cudf.concat(datamart_list)
    # 重命名列名，分别为 session、候选商品 aid、特征值、rank
    # columns = [session, aid, co_mat_feature_name(share), rank]
    datamart.columns = ['session', 'aid', co_mat_feature_name, 'rank']
    # 将 datamart 转为 pandas 并保存为 parquet 文件
    datamart.to_pandas().to_parquet(dm_save_path + prefix + co_mat_feature_name + '.parquet')
    print(f'数据已保存到{dm_save_path + prefix + co_mat_feature_name}.parquet')
    # 释放内存
    del datamart
    gc.collect()



def generate_co_matrix_and_datamarts(co_dict, use_full_data, data_last_action, data_top_action, 
                                     data_last_hour, data_last_day, data_cart_or_buy, cart_cvr_df, 
                                     same_aid_df, dm_save_path, prefix, w2v_path):
    """
    根据配置生成共现矩阵和datamart，并保存结果。
    
    参数:
    - co_dict (dict): 共现矩阵和datamart配置字典，包含模式和参数
    - use_full_data (bool): 指定处理的数据集类型，True 表示完整数据集包括训练集+测试集，False 表示由原始训练集分为的训练集和验证集
    - data_last_action (cudf.DataFrame): 每个会话的最后操作数据, columns = [session, aid]
    - data_top_action (cudf.DataFrame): 每个会话的热门点击数据
    - data_last_hour (cudf.DataFrame): 最近一小时的交互数据
    - data_last_day (cudf.DataFrame): 最近一天的交互数据
    - data_cart_or_buy (cudf.DataFrame): 购物车和订单数据
    - cart_cvr_df (cudf.DataFrame): 点击到购物车的转化率数据
    - same_aid_df (pandas.DataFrame): aid 数据框，包含所有aid，用于合并自关联特征， columns=[aid]
    - dm_save_path (str): datamart保存路径
    - prefix (str): 文件名前缀，'train_' 或 'test_'
    - w2v_path (str): word2vec 数据路径
    
    返回:
    - same_aid_df (pandas.DataFrame): 更新后的自关联特征数据框
    """
    for pattern in co_dict.keys():
        print(f'正在构建{pattern}共现矩阵...')
        # 遍历当前模式下的所有配置（如 'click-click', 'click-buy' 等）
        for j in range(len(co_dict[pattern])):
            # 从配置中提取参数
            start_type = co_dict[pattern][j][0]  # 起始事件类型（'click' 或 'buy'）
            end_type = co_dict[pattern][j][1]    # 结束事件类型（'click' 或 'buy'）
            cutline = co_dict[pattern][j][2]     # 高频和低频 aid 的分界线（点击或购买次数阈值）
            cut_rank = co_dict[pattern][j][3]    # 共现矩阵中每个 aid_x 的最大关联 aid_y 数量
            cut_datamart_last = co_dict[pattern][j][4]  # 'last' datamart的截断排名
            cut_datamart_top = co_dict[pattern][j][5]   # 'top' datamart的截断排名
            cut_datamart_hour = co_dict[pattern][j][6]  # 'hour' datamart的截断排名
            cut_datamart_day = co_dict[pattern][j][7]   # 'day' datamart的截断排名
            action_pattern_list = co_dict[pattern][j][8]  # datamart类型列表（如 ['last', 'top', 'hour', 'day']）

            print(f'{pattern}: 正在处理配置{start_type}-{end_type}...')

            # 根据模式选择共现矩阵的构建方式
            if pattern == 'w2v':
                # 'w2v' 模式：使用 word2vec 方法构建共现矩阵
                # columns = ['aid_x'、'aid_y'和 'share'（余弦距离作为共现强度）]
                co_matrix = gen_w2v_covisit_matrix(w2v_path, dims=16, knn_num=60)  # 从 w2v_path 加载或生成 word2vec 共现矩阵

                # 根据 action_pattern_list 生成对应的datamart
                # 根据 '最后一次操作' 生成推荐结果
                if 'last' in action_pattern_list:
                    # 设置特征名，例如 'click_click_w2v_last_w2v'
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_last_w2v' 
                    # 为最后一次操作生成推荐结果
                    print(f'正在生成{pattern}|{start_type}-{end_type}-last 推荐结果...')
                    gen_action_datamart(co_matrix, data_last_action, co_mat_feature_name, 
                                        cut_datamart_last, dm_save_path, prefix, w2v=True)
                # 根据 '最近一小时点击' 生成推荐结果
                if 'hour' in action_pattern_list:
                    # 特征名
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_hour_w2v'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-hour 推荐结果...')
                    gen_action_hour_day_datamart(co_matrix, data_last_hour, co_mat_feature_name, 
                                        cut_datamart_hour, dm_save_path, prefix, w2v=True)
            else:
                # 其他模式（如 'allterm', 'dup' 等）：使用常规方法构建共现矩阵
                # 自关联特征名，例如 'same_click_click_allterm'
                same_feature_name = f'same_{start_type}_{end_type}_{pattern}'  
                # co_matrix共现矩阵,columns = ['aid_x', 'aid_y', 'share'（共现强度）, 'cart_cvr'（aid_y的转化率）]
                # same_feature自关联特征,columns = ['aid', same_col_share_name]
                co_matrix, same_feature = gen_co_matrix(use_full_data, pattern, cart_cvr_df,
                                                        start_type, end_type, same_feature_name,
                                                        chunk=20000, cutline=cutline, cut_rank=cut_rank)

                gc.collect()  # 清理内存

                # 将自关联特征转换为 pandas 格式并合并到 same_aid_df
                # columns = ['aid', same_col_share_name]
                same_feature = same_feature.to_pandas()
                # columns = ['aid', ...., same_col_share_name],增加一列same_col_share_name,
                # 有很多不同的same_col_share_name列，包含不同的自关联特征,即'same_{start_type}_{end_type}_{pattern}' 
                same_aid_df = same_aid_df.merge(same_feature, on='aid', how='left')  # 左连接，保留所有 aid

                if 'last' in action_pattern_list:
                    # 根据 最后一次操作 生成推荐结果
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_last'  # 特征名，例如 'click_click_allterm_last'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-last 推荐结果...')
                    gen_action_datamart(co_matrix, data_last_action, co_mat_feature_name, 
                                         cut_datamart_last, dm_save_path, prefix)

                if 'top' in action_pattern_list:
                    # 根据 热门点击 生成推荐结果
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_top'  # 特征名，例如 'click_click_allterm_top'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-top 推荐结果...')
                    gen_action_datamart(co_matrix, data_top_action, co_mat_feature_name, 
                                         cut_datamart_top, dm_save_path, prefix)

                if 'hour' in action_pattern_list:
                    # 根据 最近一小时点击 生成推荐结果
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_hour'  # 特征名，例如 'click_click_allterm_hour'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-hour 推荐结果...')
                    gen_action_hour_day_datamart(co_matrix, data_last_hour, co_mat_feature_name, 
                                                  cut_datamart_hour, dm_save_path, prefix)

                if 'day' in action_pattern_list:
                    # 根据 最近一天点击 生成推荐结果
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_day'  # 特征名，例如 'click_click_allterm_day'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-day 推荐结果...')
                    gen_action_hour_day_datamart(co_matrix, data_last_day, co_mat_feature_name, 
                                                  cut_datamart_day, dm_save_path, prefix)

                if 'all' in action_pattern_list:
                    # 根据 所有购买行为（购物车和订单） 生成推荐结果
                    co_mat_feature_name = f'{start_type}_{end_type}_{pattern}_all'  # 特征名，例如 'buy_buy_allterm_all'
                    print(f'正在生成{pattern}|{start_type}-{end_type}-all 推荐结果...')
                    gen_action_datamart(co_matrix, data_cart_or_buy, co_mat_feature_name, 
                                         200, dm_save_path, prefix)  # 固定截断排名为 200

                gc.collect()  # 清理内存

    # 保存自关联特征 same_aid_df
    # columns = [aid, ...., same_col_share_name] 有很多不同的same_col_share_name列，包含不同的自关联特征,
    # 即'same_{start_type}_{end_type}_{pattern}' 
    same_aid_df.to_parquet(dm_save_path + prefix + 'same_aid_df.parquet')
    


if __name__ == '__main__':
    # 对于完整的数据集（训练集加测试集）和用于验证的数据集（训练集与从训练集中分离得到的验证集），
    # 都要进行行为特征提取、共现矩阵构建和datamart生成。
    for use_full_data in [True, False]:
        if use_full_data == True:
            prefix = 'test_'
            print('正在处理完整数据集...')
        else:
            prefix = 'train_'
            print('正在处理验证数据集')

        # ----------------------------------------------------------
        # | === 加载测试数据（这里实际加载的是指定数据集的测试部分）=== |
        # ----------------------------------------------------------
        # 用做最后的推荐，根据后面构造的共现矩阵，给test集中的每个session进行召回操作
        print('正在加载test数据集...')
        test = load_data(raw_opt_path, preprocess_path, 'test', use_full_data)
        

        # ----------------------------------
        # | === 从数据中提取用户行为特征 === |
        # ----------------------------------
        # data_last_action: 最后一次操作数据, columns = [session, aid]
        # data_top_action: 每个会话的热门点击数据（对于每个session，取在这个session中被点击的次数排名前30%的aid）, columns = ['session', 'aid']
        # data_last_hour: 每个会话最近一小时内的交互数据, columns = [session, aid, ts, type, ts_max, ts_hour, ts_day, ts_week]， ts_max（最后一次操作时间戳）、ts_hour（一小时前）、ts_day（一天前）、ts_week（一周前）
        # data_last_day: 每个会话最近一天交互数据， columns = [session, aid, ts, type, ts_max, ts_hour, ts_day, ts_week]
        # data_cart_or_buy: 每个会话的购物车和订单数据（type != 0）,columns = ['session','ts','aid','type']
        print('正在提取用户行为特征...')
        data_last_action, data_top_action, data_last_hour, data_last_day, data_cart_or_buy = gen_user_behavior_features(test)


        # ---------------------------------------
        # | === 计算点击到购物车的转化率（CVR）=== |
        # ---------------------------------------
        print('正在计算点击到加入购物车的转化率...')
        # 加载合并后的训练和测试数据
        merge = load_data(raw_opt_path, preprocess_path, 'merge', use_full_data)
        # 提取所有的aid（商品ID）并转换为DataFrame，用于后续特征合并
        # columns = [aid]
        same_aid_df = merge[['aid']].drop_duplicates().reset_index(drop=True).to_pandas()
        # 计算点击到购物车的转化率（CVR）
        # columns = ['aid', 'click_n'(商品点击会话数), 'cart_n'(商品加购会话数), 'cart_cvr'(转化率) ]
        cart_cvr_df = compute_cart_cvr(merge, chunk = 7000)
        
        del merge       # 释放内存
        gc.collect()    # 垃圾回收


        # -----------------------------
        # | === 配置共现矩阵config === |
        # -----------------------------
        print('正在配置共现矩阵config')
        # 配置了8种共现矩阵的配置，接下来根据配置构造8个共现矩阵
        co_dict = create_covisit_matrix_config()


        # -----------------------
        # | === 构建共现矩阵 === |
        # -----------------------
        print('开始构建共现矩阵...')
        # 遍历 co_dict 中的所有模式（如 'allterm', 'dup', 'w2v' 等），共 8 种模式
        generate_co_matrix_and_datamarts(co_dict, use_full_data, data_last_action, 
                                         data_top_action, data_last_hour, data_last_day,
                                         data_cart_or_buy, cart_cvr_df, same_aid_df,
                                         dm_save_path, prefix, w2v_path)



