import pandas as pd
from gensim.models import Word2Vec

raw_opt_path = '../../data/train_test/'
preprocess_path = '../../data/train_valid/'
output_path = '../../data/preprocess/'

full_data = False    # 控制使用的数据集范围


def gen_w2v_emb(sentences, output_path, dims=16, use_all_data=True):
    """
    使用 Word2Vec 对 aid（商品 ID）进行嵌入训练，并将结果保存为 parquet 文件。
    
    参数:
        sentences (list): 每个元素是一个list，这个list包含一个session按时间顺序点击过的物品的aid
        output_path (str): 词向量输出文件的保存目录。
        dims (int): 词向量的维度，默认为 16，可选值如 16 或 64。
        user_all_data (bool): 是否使用完整数据集（raw_opt_path）
    
    返回:
        None。函数将结果直接保存为 parquet 文件到指定路径。
    """
    if use_all_data == True:
        prefix = 'test_'
    else:
        prefix = 'train_'

    print(f'正在生成{dims}维商品embedding...')

    # 使用 Word2Vec 模型训练 aid 向量
    w2vec = Word2Vec(
        sentences=sentences,
        vector_size=dims,   # 输出商品向量的维度
        window=5,           # 上下文窗口大小
        min_count=1,        # 最低出现次数，设为1表示保留全部的aid
        workers=16         # 使用16个线程训练
    )

    # 提取aid列表
    # w2vec.wv.index_to_key 是一个 列表，里面保存了 Word2Vec 中学到的所有“词”的原始顺序。
    # 比如：w2vec.wv.index_to_key = ['101', '205', '302', '408']
    # 将这个列表变成一列 DataFrame，只有一列，列名为aid
    w2v_df = pd.DataFrame(w2vec.wv.index_to_key, columns=['aid'])

    # 提取 aid 对应的向量
    # w2vec.wv.vectors 是一个 ndarray（二维数组），形状是 (词个数, 向量维度)，即每个词的向量。
    # 这一行把这个 array 变成 DataFrame，每列加上前缀 vec_，比如vec_1,vec_2...
    w2v_vev_df = pd.DataFrame(w2vec.wv.vectors).add_prefix('vec_')

    # 合并 aid 和对应向量
    # 变成 [aid, vec_1, vec_2, ...]
    w2v_df = pd.concat([w2v_df, w2v_vev_df], axis=1)

    print('生成完毕，正在保存...')
    # 保存为parquet文件，命名为test_w2v_output_{dims}dims.parquet或train_w2v_output_{dims}dims.parquet
    w2v_df.to_parquet(output_path + prefix + f'w2v_output_{dims}dims.parquet')
    
    print(f'已保存到{output_path + prefix}' + f'w2v_output_{dims}dims.parquet')


def gen_data(raw_opt_path, preprocess_path, use_all_data=True):
    """
    参数:
        raw_opt_path (str): 原始训练和测试 parquet 文件的目录路径，完整数据集路径（用于最终提交）,应包含 'train.parquet' 和 'test.parquet'。
        preprocess_path (str): 预处理文件的路径, 训练 + 验证数据路径（用于调参）。
        user_all_data (bool): 是否使用完整数据集（raw_opt_path）
    返回:
        sentences (list): 每个元素是一个list，这个list包含一个session按时间顺序点击过的物品的aid
    """
    print('正在读取数据文件...')
    # === 读取训练和测试集 parquet 文件 ===
    if use_all_data == True:
        train = pd.read_parquet(raw_opt_path + 'train.parquet')
        test = pd.read_parquet(raw_opt_path + 'test.parquet')
    else:
        train = pd.read_parquet(preprocess_path + 'train.parquet')
        test = pd.read_parquet(preprocess_path + 'test.parquet')
    # 合并训练集和测试集数据
    merge = pd.concat([train, test])
    del train, test

    # 仅保留用户点击行为（type == 0）
    merge = merge[merge['type'] == 0]
    
    print('正在生成sentences数据...')
    # 以 session 分组，收集每个 session 内的 aid 序列作为一个“句子”
    sentence_df = merge.groupby('session')['aid'].apply(list).reset_index(name='sentence')

    # 转换为 list 格式，供 Word2Vec 使用
    sentences = sentence_df['sentence'].to_list()

    return sentences


if __name__ == '__main__':
    
    sentences = gen_data(raw_opt_path, preprocess_path, full_data)
    # 生成维度为16的商品embedding，保存在../../data/preprocess/w2v_output_16dims.parquet文件中
    gen_w2v_emb(sentences, output_path, dims=16, use_all_data=full_data)
    # 生成维度为64的商品embedding，保存在../../data/preprocess/w2v_output_64dims.parquet文件中
    gen_w2v_emb(sentences, output_path, dims=64, use_all_data=full_data)