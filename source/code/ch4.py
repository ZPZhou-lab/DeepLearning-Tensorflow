import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from . import utils

import time
import jieba
from tqdm import tqdm

def generate_sin_signal(T : int=1000, show : bool=False) -> tf.Tensor:
    """
    Parameters
    ----------
    T : int, optional
        时间长度
    """
    time = tf.range(1, T + 1, dtype=tf.float32)
    x = tf.sin(0.01 * time) + tf.random.normal([T], 0, 0.2)
    if show:
        fig = plt.figure(figsize=(6, 3))
        plt.plot(time, x)
        plt.xlabel('time')
        plt.ylabel('x')
    return time, x

def generate_training_samples(x : tf.Tensor, tau : int=5):
    """
    Parameters
    ----------
    x : tf.Tensor
        序列样本
    tau : int, default=5
        阶段长度
    """
    T = len(x)
    # 初始化特征
    features = tf.Variable(tf.zeros((T - tau, tau)))
    # 为特征赋值
    for i in range(tau):
        features[:, i].assign(x[i: T - tau + i])
    # 标签赋值
    labels = tf.reshape(x[tau:], (-1, 1))

    return features, labels

def create_sin_signal_model():
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)])
    return net

def train_sin_signal_model(model, x_train, y_train, batch_szie : int=32, lr=0.01, Epochs=5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_func = tf.keras.losses.MeanSquaredError() # 平方损失

    animator = utils.Animator(xlabel='epoch', ylabel='mse', xlim=[1, Epochs], legend=['train'])

    # 创建迭代器
    train_iter = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_szie)
    for epoch in range(Epochs):
        for x_batch,y_batch in train_iter:
            with tf.GradientTape() as tape:
                y_hat = model(x_batch,training=True)
                loss = loss_func(y_hat, y_batch)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
        
        y_hat = model(x_train)
        loss = loss_func(y_hat, y_train).numpy()
        animator.add(epoch + 1, (loss, ))
    
    return model

def tokenize(lines : list, token : str="word"):
    """
    lines : list
        存储文本信息的列表
    token : str, default="word"
        分词方式，"word" 表示按词分词，"char" 表示按字符分词
    """
    tokens = []
    # 依次取出每一行
    # tqdm 可以显示进度条
    for line in tqdm(lines):
        # 去除空格，去除换行符
        line = line.replace(' ', '').replace('\n', '')
        # 分词
        if token == "word":
            words = list(jieba.cut(line)) # 按词分词
        elif token == "char":
            words = list(line) # 按字符分词
        tokens.append(words) # 添加到 tokens 中
    return tokens

# 词表类
class Vocab:
    def __init__(self, tokens=None, min_freq : int=0) -> None:
        if tokens is None:
            tokens = []
        # 统计每个词元出现的频率
        # 结果以字典的形式保存，key 表示词元，value 表示词元频率
        counter = self.count_corpus(tokens)
        # 按出现频率排序，按照 value 降序
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 初始化从索引到词元的映射
        # unk 的索引为 0, bos 的索引为 1，eos 的索引为 2，pad 的索引为 3
        self.idx_to_token = ['<unk>', '<bos>', '<eos>', '<pad>']
        # 初始化从词元到索引的映射
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            # 过滤掉低频词元
            if freq < min_freq:
                break
            # 将未出现过的词元，添加到词表中
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    # 统计词元的频率
    def count_corpus(self, tokens):
        from collections import Counter
        # 这里的 tokens 是 2D 列表
        # tokens 的每个元素是一个包含词元的列表
        # 将词元列表展平成一个列表
        if isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        return Counter(tokens)
    
    # 词表的 len 方法
    def __len__(self):
        return len(self.idx_to_token)

    # 词表的下标访问 getitem 方法
    # 给定词元或词元列表，返回词元的索引
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # 没见过的词，返回 unk 的索引
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]
    
    # 给定词元索引或索引列表，返回词元
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
def chinese_corpus_preprocessing(file : str, num_lines : int=50000, 
                                 min_freq : int=5, token : str="word", concat : bool=False):
    """
    Parameters
    ----------
    file : str
        文件路径
    num_lines : int, default=50000
        读取的行数
    min_freq : int
        过滤低频词元的阈值
    token : str, default="word"
        分词方式，"word" 表示按词分词，"char" 表示按字符分词
    concat : bool, default=False
        是否将语料中的每个句子拼接成一个长句子

    Returns
    -------
    vocab : Vocab
        词表
    corpus : list
        语料数据集
    """
    # 打开和读取文件
    f = open(file)
    lines = f.readlines()
    f.close()

    # 分词
    tokens = tokenize(lines[0:num_lines],token=token)
    # 构建词表
    vocab = Vocab(tokens, min_freq)
    # 创建语料
    if concat:
        corpus = [vocab[token] for line in tokens for token in line]
    else:
        corpus = [vocab[line] for line in tokens]
    
    return vocab, corpus

# 随机采样策略
def seq_data_iter_random(corpus : list, batch_size, num_steps : int):
    """
    使用随机采样生成一个批量的子序列
    """
    import random

    # 如果 corpus 是拼接后的词元列表，即 corpus 是一个 1D 列表
    # 代表 corpus 中的每个元素是一个词元
    if isinstance(corpus[0],int):
        # 生成随机偏移量 offset
        # 对序列进行分区，随机范围包括 num_steps-1
        offset = random.randint(0, num_steps - 1)
        corpus = corpus[offset:]

        # 子序列的数量
        # 减去 1，因为输出的索引是相应输入的索引加 1
        num_subseqs = (len(corpus) - 1) // num_steps

        # 生成长度为 num_steps 的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 随机抽样的策略中，可以打乱索引
        random.shuffle(initial_indices)

        def fetch_data(pos):
            # 返回从 pos 开始的长为 num_steps 的序列
            return corpus[pos:(pos + num_steps)]

        # 批量个数
        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # 每个批量的子序列的起始索引
            initial_indices_per_batch = initial_indices[i:(i + batch_size)]
            # 生成一个批量
            X = [fetch_data(pos) for pos in initial_indices_per_batch]
            Y = [fetch_data(pos + 1) for pos in initial_indices_per_batch]
            
            # 返回特征和标签
            yield tf.constant(X), tf.constant(Y)
    
    # 如果 corpus 的每个元素是一个词元列表，即 corpus 是一个 2D 列表
    # 则顺序或随机从 corpus 的每个词元列表采样
    elif isinstance(corpus[0],list):
        random.shuffle(corpus)
        for each_corpus in corpus:
            yield from seq_data_iter_random(each_corpus,batch_size,num_steps)

def seq_data_iter_sequential(corpus : list, batch_size : int, num_steps : int):
    """
    使用顺序分区生成一个小批量子序列
    """
    import random

    # 如果 corpus 是拼接后的词元列表，即 corpus 是一个 1D 列表
    # 代表 corpus 中的每个元素是一个词元
    if isinstance(corpus[0],int):
        # 生成随机偏移量 offset
        # 对序列进行分区，随机范围包括 num_steps
        offset = random.randint(0, num_steps)
        # 构成样本的词元数量
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

        # 直接顺序选取文本序列
        Xs = tf.constant(corpus[offset:(offset + num_tokens)])
        Ys = tf.constant(corpus[offset + 1:(offset + 1 + num_tokens)])
        # 变换形状，使得第一维为 batch_size
        Xs = tf.reshape(Xs, (batch_size, -1))
        Ys = tf.reshape(Ys, (batch_size, -1))
        # 批量个数
        num_batches = Xs.shape[1] // num_steps

        for i in range(0, num_steps * num_batches, num_steps):
            # 生成一个批量
            X = Xs[:, i:(i + num_steps)]
            Y = Ys[:, i:(i + num_steps)]

            # 返回特征和标签
            yield X, Y
    
    # 如果 corpus 的每个元素是一个词元列表，即 corpus 是一个 2D 列表
    # 则顺序或随机从 corpus 的每个词元列表采样
    elif isinstance(corpus[0],list):
        random.shuffle(corpus)
        for each_corpus in corpus:
            yield from seq_data_iter_random(each_corpus,batch_size,num_steps)

class SeqDataLoader:
    def __init__(self, file : str, token : str="word", min_freq : int=5, num_steps : int=5, 
                 num_lines : int=10000, concat : bool=False,
                 use_random_iter : bool=True, batch_size : int=32) -> None:
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # 创建词元，词表，语料库
        self.vocab, self.corpus = chinese_corpus_preprocessing(
            file, num_lines, min_freq, token, concat)
        
        self.batch_size = batch_size
        self.num_steps = num_steps

    # 生成器函数
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size : int, embed_size : int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        vocab_size : int
            词表大小
        embed_size : int
            词向量维度
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.vocab_size = vocab_size
        # 初始化 Embedding 层的权重
        self.embedding = self.add_weight(
            name="embedding", shape=(vocab_size, embed_size), trainable=True)
        
    def call(self, inputs, *args, **kwargs):
        """
        inputs : tf.Tensor
            输入的词元序列索引，维度为 (batch_size, num_steps)
        """
        # tf.gather 通过索引获取词元嵌入后的词向量
        return tf.gather(self.embedding, inputs)

def get_params(embed_size : int, vocab_size : int, num_hiddens : int):
    # 输入维度 p 等于词嵌入维度 embed_size
    # 输出维度 o 等于词典大小 vocab_size
    num_inputs = embed_size
    num_outputs = vocab_size

    # 正态分布初始化参数
    def normal(shape):
        return tf.random.normal(
            shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

    # 隐藏层参数
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)

    # 输出层参数
    W_ho = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_o = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)

    params = [W_xh, W_hh, b_h, W_ho, b_o]
    return params

# 初始化 RNN 隐藏状态
def init_rnn_state(batch_size : int, num_hiddens : int):
    H = tf.zeros(shape=(batch_size, num_hiddens))
    return (H, )

def rnn(inputs : tf.Tensor, state : tf.Tensor, params : list):
    """
    Parameters
    ----------
    inputs : tf.Tensor
        输入，形状为 (num_steps, batch_size, embed_size)
    state : tf.Tensor
        隐藏状态，每个的形状为 (batch_size, num_hiddens)
    params : list
        RNN 参数列表，包括 [W_xh, W_hh, b_h, W_ho, b_o]
    
    Returns
    -------
    outputs : tf.Tensor
        输出，形状为 (num_steps * batch_size, vocab_size)
    state : tf.Tensor
        更新后的隐藏状态，每个的形状为 (batch_size, num_hiddens)
    """
    # 获取参数
    W_xh, W_hh, b_h, W_ho, b_o = params
    H, = state

    outputs = []
    # 依次计算每个时间步的输出
    # X 的形状为 (batch_size, embed_size)
    for X in inputs:
        H = tf.tanh(X @ W_xh + H @ W_hh + b_h) # 更新隐藏状态
        Y = H @ W_ho + b_o # 输出，形状为 (batch_size, vocab_size)
        Y = tf.nn.softmax(Y, axis=1) # 得到概率分布
        outputs.append(Y)

    # 返回输出，最后一个时间步的隐藏状态
    return tf.concat(outputs, axis=0), (H, )

class RNNBlockScratch:
    def __init__(self, embed_size, vocab_size, num_hiddens, init_state, rnn_func, get_params):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.init_state = init_state
        self.forward = rnn_func

        # 初始化模型参数
        self.trainable_variables = get_params(embed_size, vocab_size, num_hiddens)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

    def __call__(self, X : tf.Tensor, state : tf.Tensor):
        """
        X : tf.Tensor
            输入，形状为 (batch_size, num_steps)
        state : tf.Tensor
            隐藏状态
        """
        X = tf.transpose(X) # 将形状变为 (num_steps, batch_size)
        X = self.embedding(X) # 词嵌入
        return self.forward(X, state, self.trainable_variables)

    # 获取初始时刻的隐藏状态
    def begin_state(self, batch_size : int, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
    
def chinese_text_predict(prefix, num_preds, model, vocab, token : str="char"):
    """
    ### 通过前缀 prefix 来预测后续 num_preds 个字符

    Parameters
    ----------
    prefix : str
        前缀字符，即历史文本序列
    num_preds : int
        预测字符个数
    model : Any
        训练好的模型
    vocab : Vocab
        词表
    """
    # 如果词元是单词，先对prefix进行分词
    if token == "word":
        prefix = list(jieba.cut(prefix))

    state = model.begin_state(batch_size=1,dtype=tf.float32)
    # 初始化输出为前缀的第一个字符
    output = [vocab[prefix[0]]]

    # 对输入做变换的预处理函数
    # 获取 output 的最后一个字符，作为下一次预测的输入
    # 相当于获取 X_t
    def get_input():
        return tf.reshape(tf.constant([output[-1]]), (1, 1))

    # 预热期，先将前缀字符的输出作为输入
    for y in prefix[1:]:
        # 不做预测，只更新隐藏状态
        _, state = model(get_input(), state)
        output.append(vocab[y])
    
    # 预测期，将前一时间步的输出作为当前时间步的输入
    # 让模型开始滚动预测
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        output.append(int(y.numpy().argmax(axis=1).reshape(1)))
    
    # 将输出索引转换为字符串
    output_str = ''.join([vocab.idx_to_token[i] for i in output])
    return output_str

class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, embed_size : int, vocab_size : int, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 定义嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        # 定义输出层
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs : tf.Tensor, state : tf.Tensor, *args, **kwargs):
        """
        inputs : tf.Tensor
            输入，形状为 (batch_size, num_steps)
        """
        # 交换输入的时间步和批量维度，然后进行词嵌入
        X = self.embedding(tf.transpose(inputs))
        
        # RNN 会返回两个输出 Y 和 state
        # *state 会将 RNN 返回的 state 包装为一个列表
        # 这让函数的返回与我们之前接口保持统一
        Y, *state = self.rnn(X, state) # 输出 Y 形状为 (num_steps, batch_size, num_hiddens)

        # 将输出形状变换为 (num_steps * batch_size, num_hiddens)
        Y = tf.reshape(Y, (-1, Y.shape[-1]))
        
        # 计算输出
        output = self.dense(Y) # 形状为 (num_steps * batch_size, vocab_size)
        return output, state

    # 获取初始隐藏状态
    def begin_state(self, *args, **kwargs):
        # 如果是双向 RNN 模型
        if isinstance(self.rnn, tf.keras.layers.Bidirectional):
            forward_state = self.rnn.forward_layer.cell.get_initial_state(*args, **kwargs)
            backward_state = self.rnn.backward_layer.cell.get_initial_state(*args, **kwargs)
            # 将两个列表合并成一个列表
            state = forward_state + backward_state
        else:
            state = self.rnn.cell.get_initial_state(*args, **kwargs)
        return state

def grad_clipping(grads : list, radius : float):
    """
    ### 梯度裁剪

    Parameters
    ----------
    grads : list
        每组参数的梯度组成的列表
    radius : float
        裁剪半径
    """
    radius = tf.constant(radius,dtype=tf.float32)
    new_grads = []

    # 依次取出每组参数的梯度，将梯度转换为张量
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grads.append(tf.convert_to_tensor(grad))
        else:
            new_grads.append(grad)
    
    # 计算梯度范数
    norm = tf.math.sqrt(sum([(tf.reduce_sum(grad ** 2)).numpy() for grad in new_grads]))
    norm = tf.cast(norm, dtype=tf.float32)

    # 范数超过半径，进行裁剪
    if tf.greater(norm, radius):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * radius / norm
    
    return new_grads

def train_text_generation(model, train_iter : SeqDataLoader, use_random_iter : bool=True, 
                          Epochs : int=10, lr : float=0.1, verbose : int=1):
    # 设定优化器和损失函数
    # SparseCategoricalCrossentropy 不需要将标签转换为 one-hot 向量
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    animator = utils.Animator(xlabel='epoch', ylabel='perplexity', 
                              legend=("perplexity",), xlim=[1, Epochs])

    # 存储每个迭代周期的损失和样本量
    loss_batch, samples_batch = 0, 0
    # 记录单词处理速度
    speeds = []

    for epoch in range(Epochs):
        state, start = None, time.time()
        for x_batch, y_batch in train_iter:
            # 如果是随机采样，每次迭代都随机初始化隐藏状态
            if state is None or use_random_iter:
                # 初始化隐藏状态
                state = model.begin_state(batch_size=x_batch.shape[0], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                y_hat, state = model(x_batch, state)
                y = tf.reshape(tf.transpose(y_batch), (-1,))
                loss = loss_func(y, y_hat)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            grads = grad_clipping(grads, 1)
            optimizer.apply_gradients(zip(grads, weights))

            # 将该批量的损失函数值加到总损失函数值上
            loss_batch += loss.numpy() * tf.size(y).numpy()
            samples_batch += tf.size(y).numpy()
        
        end = time.time()
        speeds.append(samples_batch / (end - start))
        
        if epoch == 0 or (epoch + 1) % verbose == 0:
            # 计算困惑都
            ppl = tf.math.exp(loss_batch / samples_batch).numpy()
            animator.add(epoch + 1, [ppl])

    print(f"平均 {np.mean(speeds):.1f} 词元/秒")
    return model

def create_rnn_model(num_hiddens : int, embed_size : int, vocab_size : int):
    """
    创建 RNN 模型
    """
    rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,kernel_initializer="glorot_uniform")
    rnn_layer = tf.keras.layers.RNN(
        rnn_cell, time_major=True, return_sequences=True, return_state=True)
    model = RNNModel(rnn_layer, embed_size, vocab_size)

    return model

# 获取 GRU 模型参数
def get_gru_params(embed_size : int, vocab_size : int, num_hiddens : int):
    # 输入维度 p 等于词嵌入维度 embed_size
    # 输出维度 o 等于词典大小 vocab_size
    num_inputs = embed_size
    num_outputs = vocab_size

    # 正态分布初始化参数
    def normal(shape):
        return tf.random.normal(
            shape=shape, stddev=0.01, mean=0, dtype=tf.float32)
    
    # 重置门，更新门，候选隐藏状态参数
    # 都包含三部分参数
    def three():
        return (
            tf.Variable(normal((num_inputs, num_hiddens))),
            tf.Variable(normal((num_hiddens, num_hiddens))),
            tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32)),
        )
    
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数

    # 输出层的参数
    W_ho = tf.Variable(normal((num_hiddens, num_outputs)))
    b_o = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))
    
    params = [W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_ho, b_o]
    return params

# 初始化 GRU 隐藏状态
def init_gru_state(batch_size : int, num_hiddens : int):
    H = tf.zeros(shape=(batch_size, num_hiddens))
    return (H, )

# GRU 计算逻辑
def gru(inputs : tf.Tensor, state : tf.Tensor, params : list):
    """
    Parameters
    ----------
    inputs : tf.Tensor
        输入，形状为 (num_steps, batch_size, embed_size)
    state : tf.Tensor
        隐藏状态，每个的形状为 (batch_size, num_hiddens)
    params : list
        RNN 参数列表，包括 [W_xh, W_hh, b_h, W_ho, b_o]
    
    Returns
    -------
    outputs : tf.Tensor
        输出，形状为 (num_steps * batch_size, vocab_size)
    state : tf.Tensor
        更新后的隐藏状态，每个的形状为 (batch_size, num_hiddens)
    """

    # 重置门，更新门，候选隐藏状态参数
    W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_ho, b_o = params
    H, = state

    outputs = []
    for X in inputs:
        # X 形状为 (batch_size, embed_size)
        # H 形状为 (batch_size, num_hiddens)
        R = tf.sigmoid(X @ W_xr + H @ W_hr + b_r) # 重置门
        Z = tf.sigmoid(X @ W_xz + H @ W_hz + b_z) # 更新门
        H_hat = tf.tanh(X @ W_xh + (H * R) @ W_hh + b_h) # 候选隐藏状态
        # 隐藏状态
        H = Z * H + (1 - Z) * H_hat

        # 输出
        Y = H @ W_ho + b_o
        Y = tf.nn.softmax(Y, axis=1) # 得到概率分布
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H, )

# 创建 GRU 模型
def create_gru_model(num_hiddens : int, embed_size : int, vocab_size : int):
    gru_cell = tf.keras.layers.GRUCell(num_hiddens,kernel_initializer='glorot_uniform')
    gru_layer = tf.keras.layers.RNN(
        gru_cell, time_major=True, return_sequences=True, return_state=True)
    model = RNNModel(gru_layer, embed_size, vocab_size)

    return model

# 获得 LSTM 模型参数
def get_lstm_params(embed_size : int, vocab_size : int, num_hiddens : int):
    # 输入维度 p 等于词嵌入维度 embed_size
    # 输出维度 o 等于词典大小 vocab_size
    num_inputs = embed_size
    num_outputs = vocab_size

    # 正态分布初始化参数
    def normal(shape):
        return tf.random.normal(
            shape=shape, stddev=0.01, mean=0, dtype=tf.float32)
    
    # 重置门，更新门，候选隐藏状态参数
    # 都包含三部分参数
    def three():
        return (
            tf.Variable(normal((num_inputs, num_hiddens))),
            tf.Variable(normal((num_hiddens, num_hiddens))),
            tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32)),
        )
    
    W_xi, W_hi, b_i = three() # 输入门参数
    W_xf, W_hf, b_f = three() # 遗忘门参数
    W_xo, W_ho, b_o = three() # 输出门参数
    W_xc, W_hc, b_c = three() # 候选记忆元参数

    # 输出层的参数
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)))
    b_q = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))
    
    params = [W_xi, W_hi, b_i, 
              W_xf, W_hf, b_f, 
              W_xo, W_ho, b_o, 
              W_xc, W_hc, b_c, 
              W_hq, b_q]
    return params

# 初始化 LSTM 的隐藏状态
def init_lstm_state(batch_size : int, num_hiddens : int):
    H = tf.zeros(shape=(batch_size, num_hiddens),dtype=tf.float32)
    C = tf.zeros(shape=(batch_size, num_hiddens),dtype=tf.float32)
    # 注意，LSTM 的隐藏状态包含两个元素
    return (H, C)

# LSTM 计算逻辑
def lstm(inputs : tf.Tensor, state : tuple, params : list):
    # 获取参数
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, C = state # 隐藏状态和记忆元

    outputs = []
    for X in inputs:
        # X 形状为 (batch_size, embed_size)
        # H 形状为 (batch_size, num_hiddens)
        # C 形状为 (batch_size, num_hiddens)
        I = tf.sigmoid(X @ W_xi + H @ W_hi + b_i) # 输入门
        F = tf.sigmoid(X @ W_xf + H @ W_hf + b_f) # 遗忘门
        O = tf.sigmoid(X @ W_xo + H @ W_ho + b_o) # 输出门
        C_hat = tf.tanh(X @ W_xc + H @ W_hc + b_c) # 候选记忆元
        C = F * C + I * C_hat # 记忆元
        H = O * tf.tanh(C) # 隐藏状态
        
        # 输出
        Y = H @ W_hq + b_q # 输出
        Y = tf.nn.softmax(Y, axis=1)
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H, C)

# 创建 LSTM 模型
def create_lstm_model(num_hiddens : int, embed_size : int, vocab_size : int):
    lstm_cell = tf.keras.layers.LSTMCell(num_hiddens,kernel_initializer='glorot_uniform')
    lstm_layer = tf.keras.layers.RNN(
        lstm_cell,time_major=True, return_sequences=True, return_state=True)
    model = RNNModel(lstm_layer, embed_size, vocab_size)
    return model

# 创建多隐藏层的 LSTM 模型
def create_multilayer_lstm_model(num_hiddens : int, num_layers : int, embed_size : int, vocab_size : int):
    lstm_cells = tf.keras.layers.StackedRNNCells([
        tf.keras.layers.LSTMCell(num_hiddens,kernel_initializer='glorot_uniform') for _ in range(num_layers)])
    lstm_layer = tf.keras.layers.RNN(
        lstm_cells, time_major=True, return_sequences=True, return_state=True)
    model = RNNModel(lstm_layer, embed_size, vocab_size)

    return model

# 创建两个隐藏层的 LSTM 模型
def create_bidirectional_lstm_model(num_hiddens : int, embed_size : int, vocab_size : int):
    lstm_cell = tf.keras.layers.LSTMCell(num_hiddens,kernel_initializer='glorot_uniform')
    lstm_layer = tf.keras.layers.RNN(
        lstm_cell, time_major=True, return_sequences=True, return_state=True)
    # 用 Bidirectional 包装 RNN 模型
    bi_rnn = tf.keras.layers.Bidirectional(lstm_layer)
    model = RNNModel(bi_rnn, embed_size, vocab_size)

    return model

def create_rnn_model_API(num_hiddens : int, embed_size : int, vocab_size : int):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(5, )), # 序列长度为5
        tf.keras.layers.Embedding(vocab_size, embed_size),
        tf.keras.layers.SimpleRNN(num_hiddens, return_sequences=True),
        tf.keras.layers.Dense(vocab_size),
        tf.keras.layers.Softmax()
    ])
    return model

def performance_comparsion(vocab_size : int, embed_size : int, 
                           allow_cudnn : bool=True, num_hiddens : int=256, ):
    # 定义 LSTM 层，对比性能
    if allow_cudnn:
        rnn_layer = tf.keras.layers.LSTM(num_hiddens,return_sequences=True)
    else:
        rnn_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(num_hiddens),return_sequences=True)
    
    # 创建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_size),
        rnn_layer,
        tf.keras.layers.Dense(vocab_size),
        tf.keras.layers.Softmax()
    ])

    return model

def train_text_generation_API(model, train_iter, 
                              Epochs : int=10, lr : float=0.1, verbose : int=1):
    # 设定优化器和损失函数
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    animator = utils.Animator(xlabel='epoch', ylabel='perplexity', 
                            legend=("perplexity",), xlim=[1, Epochs])
    
    # 存储每个迭代周期的损失和样本量
    loss_batch, samples_batch = 0, 0
    # 记录单词处理速度
    speeds = []

    for epoch in range(Epochs):
        start = time.time()
        # 训练模型
        for x_batch, y_batch in train_iter:
            # x_batch 和 y_batch 形状 : (batch_size, num_steps)
            with tf.GradientTape() as tape:
                y_hat = model(x_batch,training=True) # 形状 : (batch_size, num_steps, vocab_size)
                loss = loss_func(y_batch, y_hat)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            grads = grad_clipping(grads, 1)
            optimizer.apply_gradients(zip(grads, weights))

            # 将该批量的损失函数值加到总损失函数值上
            loss_batch += loss.numpy() * tf.size(y_batch).numpy()
            samples_batch += tf.size(y_batch).numpy()
    
        end = time.time()
        speeds.append(samples_batch / (end - start))
    
        if epoch == 0 or (epoch + 1) % verbose == 0:
            # 计算困惑都
            ppl = tf.math.exp(loss_batch / samples_batch).numpy()
            animator.add(epoch + 1, [ppl])

    print(f"平均 {np.mean(speeds):.1f} 词元/秒")
    return model

def chinese_text_predict_API(prefix, num_preds : int, model, vocab, 
                             num_steps : int=None, token : str="char"):
    # 将输入的前缀填充到符合输入要求的长度 num_steps
    def pad(tokens : list):
        if len(tokens) > num_steps:
            return tokens[-num_steps:]
        else:
            while len(tokens) < num_steps:
                # 在开头用 <pad> 填充
                tokens.insert(0, vocab["<pad>"]) 
            return tokens
    
    if token == "word":
        prefix = list(jieba.cut(prefix))
    
    # 转换为词元列表
    prefix_tokens = [vocab[word] for word in prefix]
    # 填充到符合输入要求的长度
    if num_steps is not None:
        prefix_tokens = pad(prefix_tokens)

    outputs = []

    for _ in range(num_preds):
        inputs = tf.reshape(tf.constant(prefix_tokens), (1, -1))
        y_prob = model(inputs,training=False)
        
        # 取出最后一个时间步的输出
        y_hat = tf.argmax(y_prob[0][-1]).numpy()
        word = vocab.to_tokens(y_hat)

        # 添加到输出
        outputs.append(word)
        # 更新 prefix_tokens
        prefix_tokens.append(y_hat) # 添加预测的词元到末尾
        if num_steps is not None:
            # 如果限制了 num_steps，去掉开头的词元，保持序列长度不变
            prefix_tokens = prefix_tokens[1:] 
            
    return prefix + "".join(outputs)



def preprocessing_en_zh(file : str, num_lines : int = None):
    """
    预处理英文-中文数据集
    """
    import re, jieba
    def no_space(char, prev_char):
        return char in set(",.!?:()‘’“”") and prev_char != " "
    
    # 打开读取文件
    with open(file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    num_lines = num_lines if num_lines is not None else len(lines)

    source, target = [], [] # 保存源语句和目标语句

    for line in tqdm(lines[:num_lines]):
        # 分别获取英文，中文
        en, zh = line.split("\t")

        # 英文预处理
        # 1. '\u202f' 和 '\xa0' 替换为普通空格
        en = en.replace("\u202f", " ").replace("\xa0", " ")
        # 2. 全部转换为小写
        en = en.lower()
        # 3. 在单词和标点符号之间添加空格
        en = [' ' + char if i > 0 and no_space(char, en[i - 1]) else char 
            for i, char in enumerate(en)]
        en = ''.join(en)
        # 4. 利用空格分词
        en = en.split()

        # 中文预处理
        # 1. 用中文标点替换英文标点
        zh = zh.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
        # 1. 用正则表达式保留中文字符
        zh = re.sub(r"[^\u4e00-\u9fa5，、。？！；：（）《》‘’”“0-9]", '', zh)
        # 2. 利用 jieba 分词
        zh = list(jieba.lcut(zh))

        # 将处理后的数据添加到列表中
        if 0 < len(en) < 100 and 0 < len(zh) < 100:
            source.append(en)
            target.append(zh)

    return source, target

def plot_sentence_length_hist(source : list, target : list):
    fig = plt.figure(figsize=(5, 3))
    # 统计英文和中文的句子长度
    source_len = [len(sen) for sen in source]
    target_len = [len(sen) for sen in target]

    # 绘制英文和中文的句子长度直方图
    _,_,patches = plt.hist([source_len, target_len], bins=10, label=["source", "target"])
    # 显示条纹
    for patch in patches[1]:
        patch.set_hatch('///')
    plt.legend()
    plt.xlabel("# tokens per sentence")
    plt.ylabel("frequency")

def truncate_padding(line : list, num_steps : int, padding_token : int):
    """
    截断或者填充句子
    """
    if len(line) > num_steps:
        # 截断句子
        return line[:num_steps]
    else:
        # 填充句子
        return line + [padding_token] * (num_steps - len(line))

def build_translation_array(tokens : list, vocab, num_steps):
    """
    将文本序列转换为训练用的批量数据
    """
    # 将文本词元转换为词元索引
    tokens = [vocab[line] for line in tokens] # 字符串数值化
    # 在句子末尾添加结束标记 <eos>
    tokens = [line + [vocab["<eos>"]] for line in tokens]

    # 对句子进行截断和填充，转换为张量
    array = tf.constant([truncate_padding(line, num_steps, vocab["<pad>"]) for line in tokens])
    # 统计每个句子的有效长度
    valid_len = tf.reduce_sum(tf.cast(array != vocab["<pad>"], tf.int32), axis=1)

    return array, valid_len

def load_translation_en_zh(file : str, batch_size : int, num_steps : int, 
                           num_lines : int=None, min_freq : int = 2):
    """
    ### 加载英文-中文翻译数据集
    
    Parameters
    ----------
    file : str
        数据集文件路径
    batch_size : int
        批量大小
    num_steps : int
        每个句子的时间步
    num_lines : int
        读取的行数
    
    Returns
    -------
    data_iter : tf.data.Dataset
        数据集迭代器
    src_vocab : Vocab
        源语言词表
    tgt_vocab : Vocab
        目标语言词表
    """
    # 读取数据集
    source, target = preprocessing_en_zh(file,num_lines)
    # 构建词表
    src_vocab = Vocab(source, min_freq=min_freq)
    tgt_vocab = Vocab(target, min_freq=min_freq)
    # 构建数据集
    src_array, src_valid_len = build_translation_array(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_translation_array(target, tgt_vocab, num_steps)

    # 构建数据集
    dataset = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 创建迭代器
    data_iter = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size).shuffle(batch_size)

    return data_iter, src_vocab, tgt_vocab

# RNN 编码器
class Seq2SeqEncoder(utils.Encoder):
    def __init__(self, vocab_size : int, embed_size : int, 
                 num_hiddens : int, num_layers : int, dropout : float=0, **kwargs):
        super().__init__(**kwargs)
        from keras.layers import Embedding, GRU, Bidirectional
        # 嵌入层
        self.embedding = Embedding(vocab_size, embed_size)
        # RNN 层
        self.rnn_layers = []
        for i in range(num_layers):
            self.rnn_layers.append(Bidirectional(
                GRU(num_hiddens, return_sequences=True, return_state=True, dropout=dropout)))
    
    def call(self, inputs, *args, **kwargs):
        # 输入形状：(batch_size, num_steps)
        X = self.embedding(inputs) # 先进行词嵌入

        # 保存每一层最终的隐状态
        state = []
        for layer in self.rnn_layers:
            X, *layer_state = layer(X,**kwargs)
            state.append(layer_state) # 保存每一层的隐状态
        
        return X, state

class Seq2SeqDecoder(utils.Decoder):
    def __init__(self, vocab_size : int, embed_size : int, 
                 num_hiddens : int, num_layers : int, dropout : float=0, **kwargs):
        super().__init__(**kwargs)
        from keras.layers import Embedding, GRU, Dense
        # 嵌入层
        self.embedding = Embedding(vocab_size, embed_size)
        # RNN 层
        self.rnn_layers = []
        for i in range(num_layers):
            self.rnn_layers.append(GRU(num_hiddens, return_sequences=True, return_state=True, dropout=dropout))
        # 输出层
        self.dense = Dense(vocab_size, activation='softmax')
    
    def init_state(self, enc_outputs, enc_valid_len=None, *args, **kwargs):
        # enc_outputs 包含两个元素：Y 和 state
        # Y 是编码器最后一层的输出，形状：(batch_size, num_steps, 2*num_hiddens)
        # state 是编码器每一层最后时间步的隐藏状态组成的列表
            # 如果编码器是单向 RNN，则每个元素形状：(batch_size, num_hiddens)
            # 如果编码器是双向 RNN，则包含两个隐状态，每个形状：(batch_size, num_hiddens)
        # 返回一个包含 num_layers 个元素的列表
        state = enc_outputs[1]

        # 我们将前向和后向的隐藏状态拼接在一起
        # 现在每个元素的形状：(batch_size, 2*num_hiddens)
        state = [tf.concat(layer_state, axis=-1) for layer_state in state]
        return state

    def call(self, inputs, state, *args, **kwargs):
        # 输入 inputs 形状：(batch_size, num_steps)
        # state 是包含 num_layers 个元素的列表，每个元素形状：(batch_size, 2*num_hiddens)
        X = self.embedding(inputs) # 先进行词嵌入，形状：(batch_size, num_steps, embed_size)

        # 用编码器最后一层隐藏状态构造上下文变量
        context = state[-1] # 形状：(batch_size, 2*num_hiddens)
        # 最后将 context 扩展到每个时间步，便于与输入 X 在特征维度上拼接
        # context 的形状：(batch_size, num_steps, 2*num_hiddens)
        context = tf.repeat(tf.expand_dims(context, axis=1), repeats=X.shape[1], axis=1)

        # 将输入和上下文变量拼接
        # X_and_context 的形状：(batch_size, num_steps, embed_size + 2*num_hiddens)
        X_and_context = tf.concat([X, context], axis=-1)

        # 依次计算每一层 RNN
        for i,layer in enumerate(self.rnn_layers):
            # X_and_context 的形状：(batch_size, num_steps, 2*num_hiddens)
            # X_and_context 在每层计算中相当于从 H^{(l)} 到 H^{(l+1)}
            X_and_context, state[i] = layer(X_and_context, state[i], **kwargs)
        
        # 输出形状：(batch_size, num_steps, vocab_size)
        output = self.dense(X_and_context)
        return output, state

def predict_seq2seq(model, src_sentence : str, src_vocab, tgt_vocab, 
                    num_preds : int, num_steps : int, save_attention_weights : bool=False):
    """
    save_attention_weights : bool
        是否保存注意力权重，这在下一章的注意力机制中会用到
    """
    # 将源语句按词元切分
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    # 句子的有效长度
    enc_valid_len = tf.constant([len(src_tokens)])
    # 进行截断和填充
    src_tokens = truncate_padding(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = tf.expand_dims(tf.constant(src_tokens), axis=0) # 添加批量大小维度

    # 编码器输出，初始化解码器隐藏状态
    enc_outputs = model.encoder(enc_X, enc_valid_len, training=False)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)

    # 解码器的初始输入，添加批量大小维度
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weights = [], [] # 初始化输出序列和注意力权重
    for _ in range(num_preds):
        # 依次生成一个词元，更新解码器的隐藏状态
        # Y 的形状：(1, 1, vocab_size)
        Y, dec_state = model.decoder(dec_X, dec_state, training=False)
        # 从概率分布中获取词元
        dec_X = tf.argmax(Y, axis=2) # 形状：(1, 1)
        pred = dec_X[0][0].numpy()

        # 保存注意力权重
        if save_attention_weights:
            attention_weights.append(model.decoder.attention_weights)
        
        # 遇到 <eos> 结束预测
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred) # 添加到输出序列
    return "".join(tgt_vocab.to_tokens(output_seq)), attention_weights

def chinese_bleu(label_seq : str, pred_seq : str, k : int):
    # 词元化
    import jieba, math, collections
    label_tokens, pred_tokens = list(jieba.cut(label_seq)), list(jieba.cut(pred_seq))
    # 序列长度
    label_len, pred_len = len(label_tokens), len(pred_tokens)
    
    # 初始化 BLEU 分数
    score = math.exp(min(0, 1 - label_len / pred_len)) # 惩罚项
    
    for n in range(1, k+1):
        num_matches = 0 # 匹配上的 n-gram 个数
        label_subs = collections.defaultdict(int) # 统计 label_tokens 中 n-gram 语法
        for i in range(label_len - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(pred_len - n + 1):
            pred_sub = ''.join(pred_tokens[i: i + n]) # 选取 n-gram 语法
            # 在标签中匹配上了 n-gram 语法
            if label_subs[pred_sub] > 0:
                num_matches += 1 # 匹配上的 n-gram 个数加 1
                label_subs[pred_sub] -= 1 # 减去已经匹配上的 n-gram 语法
        # 计算 BLEU
        score *= math.pow(num_matches / (pred_len - n + 1), math.pow(0.5, n))
    return score