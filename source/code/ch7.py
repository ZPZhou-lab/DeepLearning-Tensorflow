import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from . import utils

import random
from collections import Counter
import collections
from sklearn.neighbors import NearestNeighbors

import time
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings('ignore')

import tensorflow_models as tfm
import tensorflow_hub as hub

def load_ptb(path : str):
    with open(path, 'r') as f:
        lines = f.read()
    # 按照换行符分割，抽取得到每个句子
    # 然后按照空格分割，得到每个单词，完成分词
    # 由于 PTB 数据集非常干净，所以不需要做其他的预处理
    return [sentence.split() for sentence in lines.split('\n')]

def down_sampling(sentences : list, vocab, t : float = 1e-4):
    # 下采样高频词

    # 排除掉未知词元 <unk>
    sentences = [[token for token in line if vocab[token] != vocab['<unk>']] for line in sentences]
    
    # 统计词频
    
    counter = Counter([token for line in sentences for token in line])
    num_tokens = sum(counter.values()) # 总词数

    # 定义一个辅助函数，如果在下采样过程中保留该词，则返回 True
    def keep(token):
        return random.uniform(0, 1) < np.sqrt(t / counter[token] * num_tokens)
    
    # 对每个句子的词元进行下采样
    return [[token for token in line if keep(token)] for line in sentences], counter

def create_centers_and_context(corpus : list, max_window_size : int=2):
    # 抽取中心词和上下文词

    # 初始化中心词和上下文词
    centers, contexts = [], []
    # 遍历每个句子
    for line in corpus:
        # 如果句子长度小于 2，跳过
        if len(line) < 2:
            continue
        # 遍历句子中的每个词元，将其作为中心词
        centers += line # 先将句子中的词元添加到中心词列表中
        for index, center in enumerate(line):
            # 随机选择窗口大小
            window_size = random.randint(1, max_window_size)
            # 确认上下文词的索引
            indices = list(range(max(0, index - window_size), min(len(line), index + 1 + window_size)))
            indices.remove(index) # 移除中心词
            contexts.append([line[idx] for idx in indices]) # 添加上下文词
    return centers, contexts

class RandomGenerator:
    def __init__(self, sampling_weights, buffer_size : int=10000) -> None:
        self.population = list(range(len(sampling_weights))) # 总体的顺序索引
        self.samping_weights = sampling_weights
        self.candidates = []
        self.i = 0
        self.buffer_size = buffer_size # 缓存大小
    
    # 采样
    def draw(self):
        # 通过递归，直到采样数目达到 
        if self.i == len(self.candidates):
            # 缓存采样结果
            self.candidates = random.choices(self.population, weights=self.samping_weights, k=self.buffer_size)
            self.i = 0 # 重置索引
        # 每次返回一个采样结果，并将索引加 1
        self.i += 1
        return self.candidates[self.i - 1]

def negative_sampling(contexts, vocab, counter, K : int=5):
    """
    ## negative_sampling
        上下文词元负采样

    Parameters
    ----------
    contexts : list
        上下文词元列表
    vocab : Vocab
        词表
    counter : dict
        词频统计
    K : int, default = 5
        负采样数目
    """
    
    # 计算采样权重，确定采样分布 P(w)
    # 四个特殊词元不会出现在词频统计中，因此使用 get 方法获取词频时，指定默认值 0
    # 0**0.75 = 0，因此这四个特殊词元的采样权重为 0
    sampling_weights = [counter.get(vocab.to_tokens(i),0)**0.75 for i in range(len(vocab))]
    # 初始化采样器和负样本
    sampler = RandomGenerator(sampling_weights)
    negatives = []
    # 遍历每个上下文词元
    for context in contexts:
        negative = [] # 初始化当前上下文词元的负样本
        while len(negative) < len(context) * K: # 每个上下文词元需要采样 K 个负样本
            # 采样
            neg = sampler.draw()
            # 如果采样到的词元是上下文词元，跳过
            if neg not in context:
                negative.append(neg)
        negatives.append(negative)

    return negatives

def create_batch_data(data : list):
    # 计算 max_len
    max_len = max([len(context) + len(negative) for center, context, negative in data])
    # 初始化中心词、上下文词和负样本的索引列表，以及掩码
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        valid_len = len(context) + len(negative) # 当前样本的有效长度
        centers += [center] # 添加中心词
        contexts_negatives += [context + negative + [0] * (max_len - valid_len)] # 添加上下文词和负样本
        masks += [[1] * valid_len + [0] * (max_len - valid_len)] # 有效填充的掩码
        labels += [[1] * len(context) + [0] * (max_len - len(context))] # 正负样本掩码
    return tf.reshape(tf.constant(centers), (-1,1)), tf.constant(contexts_negatives), tf.constant(masks), tf.constant(labels)

def create_PTBdataloader(path : str, batch_size : int=32, max_window_size : int=5, num_noise_words : int=5):
    # 加载文件，并进行分词
    sentences = load_ptb(path)
    vocab = utils.Vocab(sentences, min_freq=10) # 创建词表
    # 下采样高频词元
    down_sampled, counter = down_sampling(sentences,vocab)
    corpus = [vocab[line] for line in down_sampled] # 将词元转换为索引
    centers, contexts = create_centers_and_context(corpus, max_window_size) # 创建中心词和上下文词
    negatives = negative_sampling(contexts, vocab, counter, num_noise_words) # 创建负样本

    # 数据集 PTB 的 DataLoader
    class PTBDataset:
        def __init__(self, centers, contexts, negatives) -> None:
            assert len(centers) == len(contexts) == len(negatives) # 确保三个列表长度相同
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
            self.data = list(zip(self.centers, self.contexts, self.negatives)) # 将三个列表打包为列表
        
        # 每次返回一份批量数据
        def __iter__(self):
            # 打乱数据
            random.shuffle(self.data)
            # 根据 batch_size 划分数据
            for i in range(0, len(self.data), batch_size):
                yield create_batch_data(self.data[i:i+batch_size])

        def __len__(self):
            return len(self.centers)

        def create_dataset(self):
            dataset = tf.data.Dataset.from_generator(self.__iter__, output_types=(tf.int32,tf.int32,tf.int32,tf.int32))
            return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    dataset = PTBDataset(centers, contexts, negatives)
    return dataset.create_dataset(), vocab

def skip_gram(centers, contexts_negatives, embed_v, embed_u):
    # centers 形状：(batch_size, 1)
    # contexts_negatives 形状：(batch_size, max_len)
    
    v = embed_v(centers) # 中心词的词向量表示，形状：(batch_size, 1, embed_size)
    u = embed_u(contexts_negatives) # 上下文词和负样本的词向量表示，形状：(batch_size, max_len, embed_size)
    # 做批量矩阵乘法，得到中心词和上下文词的内积
    # 形状：(batch_size, 1, max_len)
    pred = tf.matmul(v, tf.transpose(u, perm=[0,2,1]))
    return pred

class word2vecSkipGram(tf.keras.Model):
    def __init__(self, vocab_size : int, embed_size : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_size = embed_size
        self.embed_v = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, name='embed_v')
        self.embed_u = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, name='embed_u')
    
    def call(self, centers : tf.Tensor, contexts_negatives : tf.Tensor, *args, **kwargs):
        return skip_gram(centers, contexts_negatives, embed_v=self.embed_v, embed_u=self.embed_u)
    
class SigmoidCELoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, labels, preds, masks, *args, **kwargs):
        # labels, preds, masks 形状：(batch_size, max_len)
        masked_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds) * masks # 形状：(batch_size, max_len)
        loss = tf.reduce_sum(masked_loss, axis=1) # 形状：(batch_size,)
        return loss / tf.reduce_sum(masks, axis=1) # 形状：(batch_size,)
    
def train_word2vec(model, data_iter, lr : float=0.01, Epochs : int=10, verbose : int=1):
    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_func = SigmoidCELoss()

    # 展示训练进度
    animator = utils.Animator(xlabel='epoch', ylabel='loss', xlim=[1, Epochs])

    for epoch in range(Epochs):
        # 存储每个迭代周期的损失和样本量
        loss_batch = tf.constant(0.0)
        train_samples = tf.constant(0)

        for i, batch in enumerate(data_iter):
            centers, contexts_negatives, masks, labels = batch
            with tf.GradientTape() as tape:
                # preds 形状：(batch_size, 1, max_len)
                preds = model(centers, contexts_negatives, training=True)
                # 变换 preds 形状，去掉中间的维度，得到形状：(batch_size, max_len)
                preds = tf.squeeze(preds, axis=1)
                # 为了满足损失计算的要求，需要将 labels, masks 变为 tf.float32 类型
                loss = loss_func(tf.cast(labels, tf.float32), preds, tf.cast(masks, tf.float32))
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

            # 计算损失
            loss_batch += tf.reduce_sum(loss)
            train_samples += tf.reduce_sum(masks)

        if epoch == 0 or (epoch + 1) % verbose == 0:
            train_loss = loss_batch.numpy() / train_samples.numpy()
            animator.add(epoch + 1, (train_loss,))

    return model

def get_similar_tokens(query_token : str, embed : tf.keras.layers.Embedding, vocab, top_k : int=3):
    """
    ## get_similar_tokens
        从词表中找出与查询词语义最相近的词
    
    Parameters
    ----------
    query_token : str
        查询目标词
    embed : tf.keras.layers.Embedding
        word2vec 模型的嵌入层
    vocab : Vocab
        词表
    top_k : int, default = 5
        返回最相近的 top_k 个词
    """
    # 获取 query_token 的词向量
    weights = embed.get_weights()[0] # 形状：(vocab_size, embed_size)
    query = weights[vocab[query_token]] # 形状：(embed_size,)
    # 计算余弦相似度，1e-9 做数值保护
    # weights @ query 形状：(vocab_size, 1)
    cos = (weights @ query[:, None])[:,0] / tf.sqrt(tf.reduce_sum(weights * weights, axis=1) * tf.reduce_sum(query * query) + 1e-9)
    
    # 通过 argsort 函数返回 top_k 个最大的元素的索引
    topk_tokens = tf.argsort(cos, axis=0, direction='DESCENDING')[:top_k+1].numpy().tolist()
    for i in topk_tokens:
        print('cosine sim = %.3f: %s' % (cos[i], vocab.idx_to_token[i]))

def get_max_freq_pair(token_freqs):
    # 初始化字符对统计表
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split() # 拆分token，得到子词符号
        # 对连续的符号对进行计数
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return max(pairs, key=pairs.get) # 返回最大频率的字符对

def merge_symbols(max_freq_pair, token_freqs, symbols):
    # 在子词符号表中添加新的字符对
    symbols.append("".join(max_freq_pair))
    # 初始化新的 token 频率表
    new_token_freqs = {}
    for token, freq in token_freqs.items():
        # 将 token 中的字符对替换为连续字符对
        # 例如 max_freq_pair = ('t', 'a')
        # token = 't a l l e r </w>' -> 'ta l l e r </w>'
        new_token = token.replace(" ".join(max_freq_pair), "".join(max_freq_pair))
        new_token_freqs[new_token] = freq
    return new_token_freqs

def segment_BPE(tokens : list, symbols : list):
    """
    ## segment_BPE
        BPE子词编码算法
    
    Parameters
    ----------
    tokens : list
        待子词编码的 token 列表
    symbols : list
        BPE 子词符号表
    """
    outputs = [] # 初始化输出
    for token in tokens:
        # 滑动窗口法，每次贪心地选择尽可能长的子词
        start, end = 0, len(token)
        cur_output = []
        while start < len(token) and start < end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start = end
                end = len(token)
            else:
                # 不断缩小窗口右侧边界
                # 直到 token[start:end] 在子词符号表中
                end -= 1 
        # 如果 start < len(token)，说明上述循环由 start < end 跳出
        # 说明当前 token 无法被 BPE 子词符号表编码，出现了未知符号
        if start < len(token):
            cur_output.append('<unk>')
        outputs.append(' '.join(cur_output))
    return outputs


class TokenizerEmbedding:
    def __init__(self, path : str):
        # path : 词嵌入文件路径
        self.idx_to_token, self.idx_to_vec = self._load_embedding(path)
        self.unk_idx = 0 # 未知词元的索引
        # 交换词元索引和词元的位置，得到 token_to_idx
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, path : str):
        # 从 path 加载词嵌入结果，并依次创建 idx_to_token 和 idx_to_vec
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(path, 'r', encoding='utf-8') as f:
            # 每一行是一个词元及其对应的词向量
            # 例如：'hello 0.32 0.12 ...'
            for line in f:
                line = line.rstrip()
                if line:
                    idx_to_token.append(line.split()[0]) # 词元
                    idx_to_vec.append(np.array(line.split()[1:], dtype=np.float32)) # 词向量
        # 在 idx_to_vec 中添加一个全 0 向量表示未知词元的嵌入
        idx_to_vec.insert(0, [0.0]*len(idx_to_vec[0]))
        idx_to_vec = np.stack(idx_to_vec) # 拼接得到词向量矩阵
        return idx_to_token, idx_to_vec
    
    # 定义下标访问方法
    def __getitem__(self, tokens : list):
        if isinstance(tokens, str):
            idx = self.token_to_idx.get(tokens, self.unk_idx)
            return self.idx_to_vec[idx]
        vectors = [self.__getitem__(token) for token in tokens]
        return np.array(vectors)
    
    def __len__(self):
        return len(self.idx_to_token)


class CosineSimilarity:
    def __init__(self, embed, topK : int=10) -> None:
        self.embed = embed # 词嵌入模型
        self.topK = topK # 返回最相似的 topK 个词元
        self.vocab_size = len(embed) # 词元数量

        # 余弦相似度距离
        def cosine_distance(x, y):
            # x, y : 词元的索引
            # embed : 词嵌入模型
            # 计算 cos 相似度
            x, y = int(x), int(y) # 转换为整数
            # 获得词向量
            x_vec = self.embed.idx_to_vec[x]
            y_vec = self.embed.idx_to_vec[y]
            # 距离为 1 - cos 相似度
            return 1 - np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec) + 1e-16)

        # 训练
        self.model = NearestNeighbors(n_neighbors=topK,metric=cosine_distance)
        self.model.fit(np.arange(self.vocab_size).reshape(-1,1))
    
    def kneighbors(self, token : str):
        # token : 词元
        idx = self.embed.token_to_idx.get(token, 0) # 词元索引
        # 返回最相似的 topK 个词元的索引
        distance, indices = self.model.kneighbors([[idx]])
        distance, indices = distance[0], indices[0] # 去除掉多余的样本维度
        
        similarities = 1 - distance # 将距离转换为相似度
        tokens = [self.embed.idx_to_token[idx] for idx in indices] # 将索引转换为词元
        
        # 打印查找结果
        for s,token in zip(similarities, tokens):
            print("词元：{}，相似度：{:.3f}".format(token, s))
        return tokens

def get_analogy(token_a, token_b, token_c, embed):
    def find_nearest(weights, target):
        # weights : 所有词元的词向量
        # 计算余弦相似度
        cos = (weights @ target[:, None])[:,0] / tf.sqrt(tf.reduce_sum(weights * weights, axis=1) 
                                                         * tf.reduce_sum(target * target) + 1e-9)
        # 通过 argsort 函数排序
        top_token = tf.argsort(cos, axis=0, direction='DESCENDING')[0].numpy()
        return top_token

    # 转换为词向量
    vec_a, vec_b, vec_c = embed[token_a], embed[token_b], embed[token_c]
    target = vec_b - vec_a + vec_c # 计算目标向量
    return embed.idx_to_token[find_nearest(embed.idx_to_vec, target)]

def get_BERT_tokens_and_segments(tokens_a, tokens_b):
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    # segments 用于区分两个句子，0 表示第一个句子，1 表示第二个句子
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(tf.keras.layers.Layer):
    """
    ### BERTEncdoer
    BERT 编码器
    """
    def __init__(self, vocab_size : int, num_hiddens : int, norm_shape : list, ffn_num_hiddens : int,
                 num_heads : int, num_layers : int, dropout : float, use_bias : bool=True, max_len : int = 1000, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(BERTEncoder, self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        # 词元嵌入
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        # 段落嵌入
        self.segment_embedding = tf.keras.layers.Embedding(2, num_hiddens)
        # 生成可学习的位置编码
        self.pos_embedding = tf.Variable(tf.random.normal(shape=(1, max_len, num_hiddens), dtype=tf.float32), trainable=True)
        # 注意力层，沿用 Transformer 的实现
        self.blocks = [
            utils.EncoderBlock(num_hiddens, norm_shape, ffn_num_hiddens, num_heads, dropout, use_bias) for _ in range(num_layers)
        ]
    
    def call(self, tokens, segments, valid_lens, **kwargs):
        # tokens, segments 形状: (batch_size, seq_len)
        # BERT 输入 = 词元嵌入 + 段落嵌入 + 位置嵌入
        X = self.token_embedding(tokens) + self.segment_embedding(segments) + self.pos_embedding[:, :tokens.shape[1], :]
        # 注意力层计算
        for block in self.blocks:
            X = block(X, valid_lens, **kwargs)
        return X
    
class MaskedLM(tf.keras.Model):
    def __init__(self, vocab_size : int, num_hiddens : int, *args, **kwargs):
        super(MaskedLM, self).__init__(*args, **kwargs)
        self.mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='relu'), # 形状: (batch_size, seq_len, num_hiddens)
            tf.keras.layers.LayerNormalization(), # 形状: (batch_size, seq_len, num_hiddens)
            tf.keras.layers.Dense(vocab_size), # 形状: (batch_size, seq_len, vocab_size)
            tf.keras.layers.Softmax() # 得到每个词元的概率分布
        ])
    
    def call(self, X, pred_positions, training=None, mask=None):
        # X 形状: (batch_size, seq_len, num_hiddens)
        # pred_positions 形状: (batch_size, num_pred)
        num_pred = pred_positions.shape[1] # 预测词元的个数
        pred_positions = tf.reshape(pred_positions, shape=(-1,)) # 拉直为向量，形状: (batch_size * num_pred,)
        batch_size = X.shape[0] # 批量大小

        # 假设 batch_size = 2，num_pred = 3，则下面的 batch_idx = [0,0,0,1,1,1]
        batch_idx = tf.range(0, batch_size) # 批量的索引 0,1,2,...，形状: (batch_size,)
        batch_idx = tf.repeat(batch_idx, repeats=num_pred) # 每个批量重复 num_pred 次，形状: (batch_size * num_pred,)

        # 选取需要预测的词元的隐藏状态，形状: (batch_size * num_pred, num_hiddens)
        masked_X = tf.gather_nd(X, indices=tf.stack((batch_idx, pred_positions), axis=1))
        masked_X = tf.reshape(masked_X, shape=(batch_size, num_pred, -1)) # 形状: (batch_size, num_pred, num_hiddens)
        mlm_Y_hat = self.mlp(masked_X,training=training) # 形状: (batch_size, num_pred, vocab_size)
        return mlm_Y_hat
    
class NextSentencePred(tf.keras.Model):
    def __init__(self, num_hiddens : int, *args, **kwargs):
        super(NextSentencePred, self).__init__(*args, **kwargs)
        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='tanh'),
            tf.keras.layers.Dense(2), # 二分类
            tf.keras.layers.Softmax()
        ])
    
    def call(self, cls):
        # BERT 的 cls 表示，形状: (batch_size, num_hiddens)
        return self.classifier(cls)

class BERTModel(tf.keras.Model):
    def __init__(self, vocab_size : int, num_hiddens : int, norm_shape : list, ffn_num_hiddens : int,
                 num_heads : int, num_layers : int, dropout : float, max_len : int=1000, *args, **kwargs):
        super(BERTModel, self).__init__(*args, **kwargs)
        # 创建 BERT 编码器
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_hiddens, 
                                       num_heads, num_layers, dropout, max_len=max_len)
        # 两个预训练任务
        self.mlm = MaskedLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred(num_hiddens)
    
    def call(self, tokens, segments, valid_lens=None, pred_positions=None, training=None):
        encoded_X = self.encoder(tokens, segments, valid_lens, training=training)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions, training=training)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :]) # cls = encoded_X[:, 0, :]
        return encoded_X, mlm_Y_hat, nsp_Y_hat
    
def read_wikitext(path : str):
    with open(path, 'r') as f:
        lines = f.readlines()
    # 用句号 . 作为分隔符，然后过滤掉长度小于 2 的序列
    # 段落 paragraphs 的每个元素是包含大于两个句子的列表
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs 是一个二维列表，每个元素是包含多个句子的段落
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    # paragraph 是一个段落的句子列表，每个句子是词元列表
    # paragraphs 是包含多个段落的列表
    nsp_data = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        # 用 <cls> 和 <sep> 拼接句子
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_BERT_tokens_and_segments(tokens_a, tokens_b)
        nsp_data.append((tokens, segments, is_next))
    return nsp_data

def get_replace_mlm_tokens(tokens, valid_pred_positions, num_mlm_preds, vocab):
    # 创建 MLM 的输入和标签
    mlm_input = [token for token in tokens]
    pred_positions_and_labels = []

    # 打乱 valid_pred_positions 中的元素顺序
    random.shuffle(valid_pred_positions)
    for mlm_pred_position in valid_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% 的时间，将 token 替换成 <mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% 的时间，将 token 替换成一个随机词元
            if random.random() < 0.5:
                masked_token = random.choice(vocab.idx_to_token)
            # 10% 的时间，保持 token 不变
            else:
                masked_token = tokens[mlm_pred_position]
        # 完成替换
        mlm_input[mlm_pred_position] = masked_token
        # 记录下被替换的词元位置，及其被替换前的词元，作为标签
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input, pred_positions_and_labels

def get_mlm_data_from_tokens(tokens, vocab):
    valid_pred_positions = [] # 初始化可以被掩蔽词元的位置列表
    for i,token in enumerate(tokens):
        # 排除掉特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        valid_pred_positions.append(i)
    
    # 选择 15% 的词元进行掩蔽
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input, pred_positions_and_labels = get_replace_mlm_tokens(tokens, valid_pred_positions, num_mlm_preds, vocab)
    # 根据被掩蔽词元的位置进行排序
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    # 分别取出被掩蔽词元的位置和标签
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    return vocab[mlm_input], pred_positions, vocab[mlm_pred_labels]

def pad_BERT_inputs(examples, max_len, vocab):
    # examples 是一个列表，其中的每个元素都包含 5 个元素
    # 由函数 get_nsp_data_from_paragraph 和 get_mlm_data_from_tokens 生成
    # 分别是：序列词元索引、被掩蔽词元的位置、被掩蔽词元的标签索引、段落标记、下一句预测标签
    max_num_mlm_preds = round(max_len * 0.15) # 最多被掩蔽词元的数量
    # 初始化 BERT 所需要的各个输入和标签
    all_token_ids, all_segments, valid_lens  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # 在末尾添加 <pad> 符号，填充句子到 max_len 长度
        all_token_ids.append(tf.constant(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=tf.int32))
        all_segments.append(tf.constant(segments + [1] * (max_len - len(segments)), dtype=tf.int32))
        # valid_lens 统计除去 <pad> 之外的有效词元的长度
        valid_lens.append(tf.constant(len(token_ids), dtype=tf.float32))

        # 预测位置填充到 max_num_mlm_preds 长度
        all_pred_positions.append(tf.constant(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=tf.int32))
        # weights 和 valid_lens 作用类似，MLM 任务中填充词乘以 weights 从而过滤损失
        all_mlm_weights.append(tf.constant([1.0] * len(mlm_pred_label_ids) + 
                                           [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=tf.float32))
        all_mlm_labels.append(tf.constant(mlm_pred_label_ids + [vocab["<pad>"]] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=tf.int32))
        # is_next 作为 NSP 的标签
        nsp_labels.append(tf.constant(is_next, dtype=tf.int32))
    
    return (all_token_ids, all_segments, valid_lens, 
            all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

class WikiTextDataset:
    def __init__(self, path : str, max_len : int=100, batch_size : int=32) -> None:
        self.max_len = max_len
        self.batch_size = batch_size
        paragraphs = read_wikitext(path) # 读取数据集
        # 进行分词，按单词进行分词
        paragraphs = [utils.english_tokenize(paragraph,token="word") for paragraph in paragraphs]
        # 将段落拆分为句子
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]

        # 创建词表
        self.vocab = utils.Vocab(sentences, min_freq=5)
        # 为词表加入三个特殊词元
        special_tokens = ['<mask>', '<cls>', '<sep>']
        for token in special_tokens:
            self.vocab.idx_to_token.append(token)
            self.vocab.token_to_idx[token] = len(self.vocab.idx_to_token) - 1
        
        # 获得 NSP 任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))
        # 获得 MLM 任务的数据
        examples = [(get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]

        # 填充数据
        (self.all_token_ids, self.all_segments, self.valid_lens, 
         self.all_pred_positions, self.all_mlm_weights, 
         self.all_mlm_labels, self.nsp_labels) = pad_BERT_inputs(examples, max_len, self.vocab)
    
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
                self.all_pred_positions[idx], self.all_mlm_weights[idx],
                self.all_mlm_labels[idx], self.nsp_labels[idx])
    
    def __len__(self):
        return len(self.all_token_ids)

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.all_token_ids, self.all_segments, self.valid_lens,
                                                      self.all_pred_positions, self.all_mlm_weights,
                                                      self.all_mlm_labels, self.nsp_labels))
        dataset = dataset.shuffle(len(self.all_token_ids)).batch(self.batch_size)
        return dataset
    
def get_BERT_pretrain_loss(model, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y):
    # 前向计算
    _, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X, valid_lens_X, pred_positions_X,training=True)
    
    # 计算 MLM 损失
    mlm_l = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=tf.reshape(mlm_Y, (-1,)), y_pred=tf.reshape(mlm_Y_hat, (-1, vocab_size))) * tf.reshape(mlm_weights_X, (-1,))
    mlm_l = tf.reduce_sum(mlm_l) / (tf.reduce_sum(mlm_weights_X) + 1e-8) # 做数值保护

    # 计算 NSP 损失
    nsp_l = tf.keras.losses.sparse_categorical_crossentropy(y_true=nsp_Y, y_pred=nsp_Y_hat)
    nsp_l = tf.reduce_mean(nsp_l)
    # 总体损失
    loss = mlm_l + nsp_l

    return mlm_l, nsp_l, loss

def pretrain_BERT(model, train_iter, vocab, total_steps : int=50, lr : float=0.01):
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 展示训练进度
    animator = utils.Animator(xlabel='epoch', ylabel="loss", xlim=[1, total_steps], legend=['mlm_loss', 'nsp_loss'])

    step = 0
    batch_mlm_loss = tf.constant(0.0)
    batch_nsp_loss = tf.constant(0.0)
    num_batches = tf.constant(0.0)
    num_tokens = 0
    total_time = 0

    while step < total_steps:
        for (tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y) in train_iter:
            start = time.time()
            with tf.GradientTape() as tape:
                mlm_l, nsp_l, loss = get_BERT_pretrain_loss(model, len(vocab), tokens_X, segments_X, valid_lens_X, 
                                                            pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

            batch_mlm_loss += mlm_l
            batch_nsp_loss += nsp_l
            num_batches += 1.0
            num_tokens += tokens_X.shape[0]
            end = time.time()
            total_time += end - start

            animator.add(step+1, ((batch_mlm_loss/num_batches).numpy(), (batch_nsp_loss/num_batches).numpy()))
            step += 1
            if step == total_steps:
                break
    print(f"{num_tokens / total_time:.2f} 句子对 / 每秒")
    return model

def read_imdb(path : str, dataset : str="train"):
    # 初始化数据和标签列表
    data, labels = [], []

    # 拼接路径
    path = os.path.join(path, dataset)
    for label in ["pos", "neg"]:
        sub_path = os.path.join(path, label)
        # 读取文件夹下的所有文件
        for file in os.listdir(sub_path):
            with open(os.path.join(sub_path, file), "r", encoding="utf-8") as f:
                # 读取文件内容
                text = f.read()
                data.append(text)
                labels.append(0 if label == "neg" else 1)
    return data, labels

def load_imdb_dataset(path : str, batch_size : int=64, num_steps : int=300):
    # 读取数据
    train_data, train_labels = read_imdb(path, dataset="train")
    test_data, test_labels = read_imdb(path, dataset="test")
    # 词元化
    train_tokens = utils.english_tokenize(train_data, token="word")
    test_tokens = utils.english_tokenize(test_data, token="word")
    # 创建词表
    vocab = utils.Vocab(train_tokens, min_freq=5)
    # 填充和截断
    train_features = tf.constant([
        vocab[utils.truncate_padding(line, num_steps, vocab["<pad>"])] for line in train_tokens])
    test_features = tf.constant([
        vocab[utils.truncate_padding(line, num_steps, vocab["<pad>"])] for line in test_tokens])
    
    # 创建数据集
    train_iter = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_iter = train_iter.shuffle(buffer_size=len(train_features)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    test_iter = test_iter.batch(batch_size)

    return train_iter, test_iter, vocab

class SentimentRNN(tf.keras.Model):
    def __init__(self, vocab_size : int, embed_size : int, 
                 num_hiddens : int, num_layers : int, *args, **kwargs):
        super(SentimentRNN, self).__init__(*args, **kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        # RNN 编码器获得句子的特征向量
        self.encoder = tf.keras.models.Sequential()
        for i in range(num_layers):
            # 双向 LSTM
            self.encoder.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(num_hiddens, return_sequences=True)))
        # 下游任务网络，二分类
        self.decoder = tf.keras.layers.Dense(2, activation="softmax")
    
    def call(self, X, *args, **kwargs):
        # X 的形状: (batch_size, num_steps)
        # 先做词嵌入
        X = self.embedding(X) # 形状: (batch_size, num_steps, embed_size)
        encoded_X = self.encoder(X, **kwargs) # 形状: (batch_size, num_steps, 2 * num_hiddens)
        # 连接 初始 和 最终 时间步的隐藏状态作为全连接层输入
        encoded_X = tf.concat((encoded_X[:, 0], encoded_X[:, -1]), -1) # 形状: (batch_size, 4 * num_hiddens)
        # 进行分类
        return self.decoder(encoded_X,**kwargs)

def train_classifier(model, train_iter, test_iter, Epochs : int=10, lr : float=0.01, verbose : int=10):
    def batch_predict(data_iter):
        y_prob, y_true = [], []
        for X, y in data_iter:
            y_hat = model(X)
            y_prob.append(y_hat)
            y_true.append(y)
        y_prob = tf.concat(y_prob, axis=0)
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.argmax(y_prob, axis=1, output_type=tf.int32)
        return y_prob, y_true, y_pred
        
    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    # 展示训练进度
    animator = utils.Animator(xlabel="epoch", xlim=[1, Epochs], figsize=(7, 3), ncols=2,
                              legend=(("train loss",), ("train acc", "test acc")),
                              fmts=(('-',), ('m--', 'g-.')))

    for epoch in range(Epochs):
        for step, (x_batch, y_batch) in enumerate(train_iter):
            with tf.GradientTape() as tape:
                y_hat = model(x_batch, training=True)
                loss = loss_func(y_batch, y_hat)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            # grads = utils.grad_clipping(grads, 1) # 梯度裁剪
            optimizer.apply_gradients(zip(grads, weights))
        
            if step == 0 or (step + 1) % verbose == 0:
                # 进行评估
                y_train_prob, y_train_true, y_train_pred = batch_predict(train_iter)
                _, y_test_true, y_test_pred = batch_predict(test_iter)
                # 计算损失
                train_loss = loss_func(y_train_true, y_train_prob).numpy()
                train_acc = tf.equal(y_train_true, y_train_pred).numpy().mean()
                test_acc = tf.equal(y_test_true, y_test_pred).numpy().mean()

                animator.add(epoch + 1 + (step) / len(train_iter), (train_loss,),ax=0)
                animator.add(epoch + 1 + (step) / len(train_iter), (train_acc, test_acc),ax=1)
    
    return model

def predict_sentiment(model, vocab, sentence):
    sentence = tf.constant(vocab[utils.english_tokenize(sentence, token="word")], dtype=tf.int32)
    label = tf.argmax(model(sentence), axis=1)[0]
    return "positive" if label == 1 else "negative"

class textCNN(tf.keras.Model):
    def __init__(self, vocab_size : int, embed_size : int, 
                 kernel_sizes : list, num_channels : list, *args, **kwargs):
        super(textCNN, self).__init__(*args, **kwargs)
        # 嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        # 定义多组卷积层
        self.convs = []
        for num_channel, kernel_size in zip(num_channels, kernel_sizes):
            self.convs.append(tf.keras.layers.Conv1D(num_channel, kernel_size))
        # 定义池化层
        self.pool = tf.keras.layers.GlobalMaxPool1D()

        # 下游任务网络，二分类
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.5), # dropout 减少过拟合
            tf.keras.layers.Dense(2, activation="softmax")
        ])
    
    def call(self, X, *args, **kwargs):
        # X 的形状: (batch_size, num_steps)
        # 先做词嵌入
        X = self.embedding(X) # 形状: (batch_size, num_steps, embed_size)
        # 进行卷积和池化，并进行拼接
        X = tf.concat([self.pool(conv(X)) for conv in self.convs], axis=1)
        X = tf.nn.relu(X) # 做一次 relu 激活，形状：(batch_size, sum(num_channels))

        # 接入全连接层进行分类
        return self.decoder(X,**kwargs)
    
def read_snli(path : str, dataset : str="train"):
    import re
    def extract_text(s):
        # 去掉无用的括号
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 将连续的多个空格替换成一个空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    labels_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
    dataset = "snli_1.0_{}.txt".format(dataset)
    path = os.path.join(path, dataset)
    with open(path, "r") as f:
        # 跳过第一行，并用 \t 分割，得到前提、假设和标签
        lines = [line.split('\t') for line in f.readlines()[1:]]
    
    # line[0] 是标签，line[1] 是前提，line[2] 是假设
    premises = [extract_text(line[1]) for line in lines if line[0] in labels_dict] # 前提
    hypotheses = [extract_text(line[2]) for line in lines if line[0] in labels_dict] # 假设
    labels = [labels_dict[line[0]] for line in lines if line[0] in labels_dict] # 标签
    
    return premises, hypotheses, labels

class SNLIDataloader:
    def __init__(self, path : str, dataset : str, num_steps : int, vocab=None) -> None:
        self.num_steps = num_steps
        # 读取训练数据集
        data = read_snli(path, dataset)
        # 对文本进行词元化
        self.premises = utils.english_tokenize(data[0], token="word")
        self.hypotheses = utils.english_tokenize(data[1], token="word")
        self.labels = tf.constant(data[2])
        # 创建词表
        if vocab is None:
            self.vocab = utils.Vocab(self.premises + self.hypotheses, min_freq=5)
        else:
            self.vocab = vocab

        # 进行填充和截断
        self.premises = self.pad_func(self.premises)
        self.hypotheses = self.pad_func(self.hypotheses)
    
    # 填充和阶段辅助函数
    def pad_func(self, X):
        return tf.constant([
            utils.truncate_padding(self.vocab[line],self.num_steps, self.vocab["<pad>"]) for line in X])
    
    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

    def create_dataset(self, batch_size, shuffle=False):
        data_iter = tf.data.Dataset.from_tensor_slices(((self.premises, self.hypotheses), self.labels))
        if shuffle:
            data_iter = data_iter.shuffle(buffer_size=len(self.labels))
        data_iter = data_iter.batch(batch_size)
        return data_iter

def load_snli_dataset(path : str, num_steps : int, batch_size : int=64):
    train_data = SNLIDataloader(path, "train", num_steps)
    vocab = train_data.vocab
    # 创建测试数据集，并指定词表 vocab
    test_data = SNLIDataloader(path, "test", num_steps, vocab=vocab)
    train_iter = train_data.create_dataset(batch_size, shuffle=True)
    test_iter = test_data.create_dataset(batch_size)
    return train_iter, test_iter, vocab

class Attend(tf.keras.Model):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu")
        ])
    
    def call(self, P, H, **kwargs):
        # P, H 的形状：(batch_size, num_steps, embed_size)
        # f_P, f_H 形状: (batch_size, num_steps, hidden_size)
        f_P, f_H = self.f(P,**kwargs), self.f(H,**kwargs) 

        # 计算注意力分数，(B, m, h) @ (B, h, n) -> (B, m, n)
        score = f_P @ tf.transpose(f_H, perm=[0, 2, 1]) # 形状：(batch_size, m, n)

        # 计算前提 P 的对齐序列 h
        # (B, m, n) @ (B, n, embed_size) -> (B, m, embed_size)
        # 意味着假设 H 被对齐到前提 P 的每个词元
        h = tf.nn.softmax(score, axis=2) @ H  # 形状：(batch_size, m, embed_size)

        # 计算假设 H 的对齐序列 p
        # (B, n, m) @ (B, m, embed_size) -> (B, n, embed_size)
        # 意味着前提 P 被对齐到假设 H 的每个词元
        p = tf.nn.softmax(tf.transpose(score, perm=[0, 2, 1]), axis=2) @ P # 形状：(batch_size, n, embed_size)

        return h, p
    
class Compare(tf.keras.Model):
    def __init__(self, num_hiddens : int, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu")
        ])
    
    def call(self, P, H, h, p, **kwargs):
        # P, H 的形状：(batch_size, m or n, embed_size)
        # h, p 的形状：(batch_size, m or n, num_hiddens)
        
        # concat(P, h) 形状: (batch_size, m, embed_size + embed_size) -> (batch_size, m, num_hiddens)
        # concat(H, p) 形状: (batch_size, n, embed_size + embed_size) -> (batch_size, n, num_hiddens)
        C_P = self.g(tf.concat([P, h], axis=-1), **kwargs)
        C_H = self.g(tf.concat([H, p], axis=-1), **kwargs)
        return C_P, C_H
    
class Aggregate(tf.keras.Model):
    def __init__(self, num_hiddens : int, num_classes : int, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
    
    def call(self, C_P, C_H, **kwargs):
        P_agg = tf.reduce_mean(C_P, axis=1) # (batch_size, num_hiddens)
        H_agg = tf.reduce_mean(C_H, axis=1) # (batch_size, num_hiddens)
        output = self.decoder(tf.concat([P_agg, H_agg], axis=-1), **kwargs)
        return output

class DecomposableAttention(tf.keras.Model):
    def __init__(self, vocab_size : int, embed_size : int, num_hiddens : int, num_classes : int, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        self.aggregate = Aggregate(num_hiddens, num_classes)
    
    def call(self, X, **kwargs):
        # X 的形状：(X[0], X[1]) = (batch_size, num_steps)
        P, H = self.embedding(X[0]), self.embedding(X[1]) # (batch_size, num_steps, embed_size)
        h, p = self.attend(P, H, **kwargs)
        C_P, C_H = self.compare(P, H, h, p, **kwargs)
        output = self.aggregate(C_P, C_H, **kwargs)
        return output

def predict_inference(model, vocab, premise : str, hypothesis : str):
    # 分词和转换为词索引，并添加 batch_size 维度
    premise = tf.constant(vocab[premise.lower().split()], dtype=tf.int32)[None, :]
    hypothesis = tf.constant(vocab[hypothesis.lower().split()], dtype=tf.int32)[None, :]

    # 预测
    label = tf.argmax(model([premise, hypothesis]), axis=1)[0]
    labels_dict = {0 : "entailment", 1 : "contradiction", 2 : "neutral"}
    return labels_dict[label.numpy()]

def load_bert_from_pretrain(num_layers : int=2, num_heads : int=2, num_hiddens : int=128):
    # 加载预训练的 BERT 模型
    bert_path = f"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-{num_layers}_H-{num_hiddens}_A-{num_heads}/2"
    bert_encoder = hub.KerasLayer(bert_path, trainable=True)

    return bert_encoder

class SNLIBert:
    def __init__(self, path : str, dataset : str, num_steps : int, bert_tokenizer) -> None:
        self.num_steps = num_steps
        # 创建分词器和 packer
        bert_preprocessor = hub.load(handle="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.tokenizer = bert_preprocessor.tokenize
        self.packer = hub.KerasLayer(
            bert_preprocessor.bert_pack_inputs,
            arguments={"seq_length": num_steps} # 指定序列长度
        )

        # 读取数据
        data = read_snli(path, dataset)
        self.premises = data[0]
        self.hypotheses = data[1]
        self.labels = data[2]
    
    def bert_input_preprocess(self, premise : str, hypothesis : str, labels : int):
        tokens_a = self.tokenizer(premise)
        tokens_b = self.tokenizer(hypothesis)
        
        packed_inputs = self.packer([tokens_a, tokens_b])
        
        return packed_inputs, labels
    
    def __getitem__(self, idx):
        return self.bert_input_preprocess([self.premises[idx]], [self.hypotheses[idx]], [self.labels[idx]])

    def __len__(self):
        return len(self.labels)

    def create_dataset(self, batch_size : int ,num_examples : int = 20000):
        # 创建数据迭代器
        data_iter = tf.data.Dataset.from_tensor_slices(
            (self.premises[:num_examples], self.hypotheses[:num_examples], self.labels[:num_examples]))
        data_iter = data_iter.batch(batch_size)
        # 调用 bert_input_preprocess 对数据进行预处理
        data_iter = data_iter.map(self.bert_input_preprocess)
        data_iter = data_iter.prefetch(tf.data.experimental.AUTOTUNE)

        return data_iter
    
def bert_predict_inference(model, train_set, premise : str, hypothesis : str):
    bert_input, _ = train_set.bert_input_preprocess([premise], [hypothesis], None)
    label = tf.argmax(model(bert_input), axis=1)[0]

    labels_dict = {0 : "entailment", 1 : "contradiction", 2 : "neutral"}
    return labels_dict[label.numpy()]

