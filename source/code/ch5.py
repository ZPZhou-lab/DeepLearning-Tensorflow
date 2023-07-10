import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from . import utils

import time
import jieba
from tqdm import tqdm

def generate_regression_data(n_train : int=50, n_test : int=50):
    def f(x):
        return 2*tf.sin(x) + x**0.8
    # 输入特征
    x_train = tf.sort(tf.random.uniform(shape=(n_train, ), maxval=5)) # 训练集特征
    x_test = tf.cast(tf.linspace(0, 5, n_test),dtype=tf.float32) # 测试集特征

    # 标签
    y_train= f(x_train) + tf.random.normal(shape=(n_train, ), stddev=0.5) # 训练集标签
    y_test = f(x_test) # 测试集标签

    return x_train, y_train, x_test, y_test

def plot_kernel_regression(x_train, y_train, x_test=None, y_test=None, y_hat=None):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x_train, y_train, 'o', color="orange", alpha=0.7,label="samples")
    if x_test is not None:
        plt.plot(x_test, y_test, "g-", label="y_test")
    if y_hat is not None:
        plt.plot(x_test, y_hat, "m--", label="y_hat")
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.xlim(0, 5); plt.ylim(-1, 5)
    plt.grid(); plt.legend()

def NadarayaWatson_attention_pooling(key, value, query, sigma : float=1):
    """
    Nadaraya-Watson注意力池化

    Parameters
    ----------
    key : tf.Tensor
        不随意线索，形状为 (n_k,)
    value : tf.Tensor
        不随意线索对应的值，形状为 (n_k,)
    query : tf.Tensor
        需要查询的随意线索，形状为 (n_q,)
    sigma : float, optional
        高斯核的标准差，默认值为 1
    """
    # 将 key 复制 n_q 份，变换形状为 (n_q, n_k) 的矩阵
    key_repeat = tf.repeat(tf.expand_dims(key, axis=0), repeats=query.shape[0], axis=0) # (n_q, n_k)
    
    # 计算注意力权重
    # 利用广播机制，先将 query 的形状变为 (n_q, 1)
    # key_repeat 与 query 广播后，形状变为 (n_q, n_k)
    attn_weights = tf.nn.softmax(-(key_repeat - tf.expand_dims(query, axis=1))**2 / (2*sigma**2), axis=1)

    # 注意力池化
    y_hat = attn_weights @ tf.expand_dims(value, axis=1) # (n_q, n_k) @ (n_k, 1) = (n_q, 1)
    y_hat = tf.squeeze(y_hat, axis=1) # (n_q, 1) -> (n_q,)
    return y_hat, attn_weights

class NadarayaWatsonRegression(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        # 初始化参数
        self.sigma = tf.Variable(initial_value=tf.random.uniform(shape=(1,)), name="sigma")
        self.attn_weights = None # 注意力权重

    def call(self, key, value, query, *args, **kwargs):
        # key, value 形状为 (batch_size, n_k)
        # query 形状为 (batch_size = n_q,)，注意 batch_size = n_q
        
        # 重复 query 变换姓张：(batch_size = n_q,) -> (n_q, 1) -> (n_q, n_k)
        query = tf.repeat(tf.expand_dims(query, axis=1), repeats=key.shape[1], axis=1)

        # 计算注意力权重，权重形状为 (batch_size = n_q, n_k)
        self.attn_weights = tf.nn.softmax(-(query- key)**2 / (2*self.sigma**2), axis=-1)
        
        # 进行注意力池化
        # (batch_size = n_q, 1, n_k) @ (batch_size = n_q, n_k, 1) -> (batch_size = n_q, 1, 1)
        y_hat = tf.expand_dims(self.attn_weights, axis=1) @ tf.expand_dims(value, axis=-1)
        y_hat = tf.squeeze(y_hat) # 去掉多余的维度：(batch_size, 1, 1) -> (batch_size = n_q, )
        return y_hat
    
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, query_size : int, key_size : int, num_hiddens : int, dropout : float, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attn_weights = None
    
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        # 将 queries 和 keys 变换到 num_hiddens 维度
        # queries 的形状为 (batch_size, n_q, num_hiddens)
        # keys 的形状为 (batch_size, n_k, num_hiddens)
        queries, keys = self.W_q(queries), self.W_k(keys)

        # 利用广播机制，将 queries 和 keys 相加
        # 扩充 queries 维度，变成 (batch_size, n_q, 1, num_hiddens)
        # 扩充 keys 维度，变成 (batch_size, 1, n_k, num_hiddens)
        # features 的形状为 (batch_size, n_q, n_k, num_hiddens)
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(keys, axis=1)
        features = tf.nn.tanh(features) # 使用 tanh 函数变换

        # 通过 w_v 将输出特征变换为形状为 (batch_size, n_q, n_k, 1) 的张量
        # 去除掉最后多余的维度，形状为 (batch_size, n_q, n_k)
        scores = tf.squeeze(self.w_v(features), axis=-1) # 注意力分数
        self.attn_weights = utils.Masked_Softmax(scores, valid_lens) # 注意力权重
        
        # 计算注意力池化输出，对权重使用 dropout，缓解过拟合
        # (batch_size, n_q, n_k) @ (batch_size, n_v = n_k, v) -> (batch_size, n_q, v)
        Y = self.dropout(self.attn_weights, **kwargs) @ values
        return Y

class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout : float, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attn_weights = None
    
    def call(self, queries, keys, values, valid_lens, *args, **kwargs):
        # queries 的形状为 (batch_size, n_q, d)
        # keys 的形状为 (batch_size, n_k, d)
        # values 的形状为 (batch_size, n_v = n_k, v)
        # valid_lens 的形状为 (batch_size,) 或者 (batch_size, n_q)
        d = queries.shape[-1]
        # 计算注意力分数
        # (batch_size, n_q, d) @ (batch_size, d, n_k) -> (batch_size, n_q, n_k
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d, dtype=tf.float32))
        self.attn_weights = utils.Masked_Softmax(scores, valid_lens) # 注意力权重

        # 计算注意力池化结果
        # (batch_size, n_q, n_k) @ (batch_size, n_v = n_k, v) -> (batch_size, n_q, v)
        Y = self.dropout(self.attn_weights, **kwargs) @ values
        return Y

class AttentionDecoder(utils.Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # 获取当前解码器的注意力权重
    @property
    def attn_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(utils.AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0,**kwargs):
        super().__init__(**kwargs)
        # 注意力层
        self.attention = AdditiveAttention(
            query_size=num_hiddens, key_size=num_hiddens, num_hiddens=num_hiddens, dropout=dropout)
        # 词嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        # RNN 层
        self.rnn = []
        for i in range(num_layers):
            self.rnn.append(tf.keras.layers.GRU(
                num_hiddens, dropout=dropout, return_sequences=True, return_state=True))
        self.dense = tf.keras.layers.Dense(vocab_size) # 输出层

    def init_state(self, enc_outputs, enc_valid_len=None, *args, **kwargs):
        # 解码器输出 enc_outputs 包含两个元素
        # outputs 形状为 (batch_size, num_steps, num_hiddens)
        # hidden_state 是一个列表，包含 num_layers 个元素
        # 每一层 hidden_state[i] 也是列表
        # 如果是单向 RNN，那么 hidden_state[i] 只包含一个元素，双向 RNN 包含两个元素
        # 每个元素形状 (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        hidden_state = [tf.concat(layer_state, axis=1) for layer_state in hidden_state]
        
        # 将 valid_len 一起返回，便于注意力掩蔽
        return (outputs, hidden_state, enc_valid_len) 

    def call(self, X, state, **kwargs):
        # 从初始化状态中取出 enc_outputs, hidden_state, enc_valid_len
        enc_outputs, hidden_state, enc_valid_len = state

        # 进行词嵌入
        X = self.embedding(X) # 形状为 (batch_size, num_steps, embed_size)
        X = tf.transpose(X, perm=[1, 0, 2]) # 转置为 (num_steps, batch_size, embed_size)
        outputs, self._attn_weights = [], [] # 注意力输出和注意力权重

        # 依次取出每个时间步
        for x in X:
            # x 的形状为 (batch_size, embed_size)
            # 初始化 query，形状为 (batch_size, n_q = 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # 利用注意力机制计算上下文变量
            # key, value 的形状为 (batch_size, n_k = n_v = num_steps, num_hiddens)
            # context 的形状为 (batch_size, n_q = 1, num_hiddens)
            # 传入 enc_valid_len 便于注意力掩蔽
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_len, **kwargs)

            # 将上下文和嵌入词特征拼接
            # 拼接后形状为 (batch_size, 1, embed_size + num_hiddens)
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)

            # RNN 层计算
            for i,layer in enumerate(self.rnn):
                # 逐层计算时，同时更新隐藏状态
                x, hidden_state[i] = layer(x, hidden_state[i], **kwargs)
            
            # x 形状为 (batch_size, 1, num_hiddens)
            outputs.append(x) # 添加到输出列表中
            self._attn_weights.append(self.attention.attn_weights) # 添加注意力权重

        # 计算输出层
        outputs = tf.concat(outputs, axis=1) # 形状 (batch_size, num_steps, num_hiddens)
        outputs = tf.nn.softmax(self.dense(outputs), axis=-1) # 形状 (batch_size, num_steps, vocab_size)

        return outputs, (enc_outputs, hidden_state, enc_valid_len)

    # 获取注意力权重
    @property
    def attention_weights(self):
        return self._attn_weights

def transpose_qkv(X, num_heads):
    """
    对输入进行变换，便于多注意力头的并行计算
    """
    # 输入X的形状 (batch_size, n_q 或 n_k, num_hiddens)
    # 先变换形状为 (batch_size, n_q 或 n_k, num_heads, num_hiddens / num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], num_heads, -1))

    # 交换中间两个轴顺序
    # 变换形状为 (batch_size, num_heads, n_q 或 n_k, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))

    # 合并第一第二个轴，形状变为 (batch_size * num_heads, n_q 或 n_k, num_hiddens / num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

def transpose_output(X, num_heads):
    """
    逆转 transpose_qkv 函数的操作
    """
    # 输入 X 的形状 (batch_size * num_heads, n_q, num_hiddens / num_heads)
    # 先拆分第一第二个轴，形状变为 (batch_size, num_heads, n_q, num_hiddens / num_heads)
    X = tf.reshape(X, shape=(-1, num_heads, X.shape[1], X.shape[2]))
    
    # 交换中间两个轴顺序，形状变为 (batch_size, n_q, num_heads, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))

    # 最后合并多头注意力，将最后两个轴合并
    # 输出形状为 (batch_size, n_q, num_hiddens)
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        # 我们会设定 d_p = d_q = d_v = d_o / h，这样可以并行计算多头注意力
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens=None, *args, **kwargs):
        # queries 形状 (batch_size, n_q, num_hiddens)
        # keys, values 形状 (batch_size, n_k = n_v, num_hiddens)
        # valid_lens 形状 (batch_size,) 或者 (batch_size, n_q)

        # 首先变换 qeuries、keys 和 values 的维度到 num_hiddeans
        # 变换后形状变为 (batch_size*num_heads, n_q 或 n_k, num_hiddens / num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads) 
        keys = transpose_qkv(self.W_k(keys), self.num_heads) 
        values = transpose_qkv(self.W_v(values), self.num_heads) 
        
        # 变换 valid_lens 的形状，用于多头注意力
        if valid_lens is not None:
            # 在轴 0 上重复 num_heads 次
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
        
        # 做注意力池化
        # 输出形状为 (batch_size*num_heads, n_q, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens,**kwargs)

        # 最后恢复输出的形状
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat) # 对输出再做一次变换

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens : int, max_len : int=1000, dropout : float=0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的位置编码矩阵 P
        # P 故意多增加了一维，这作为批量维度使用
        # 这样可以直接与形状为 (batch_size, n_q, num_hiddens) 的张量通过广播机制相加
        self.P = np.zeros((1, max_len, num_hiddens))
        # 生成 i / 10000^(2j / num_hiddens) 的矩阵
        # 这里先把 i 通过 reshape(-1,1) 变成列向量，再利用广播机制就可以得到
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1)\
            / np.power(10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:,:,0::2] = np.sin(X) # 偶数列用 sin
        self.P[:,:,1::2] = np.cos(X) # 奇数列用 cos
    
    def call(self, X : tf.Tensor, *args, **kwargs):
        # X 的形状为 (batch_size, n_q, num_hiddens)
        # 通过广播机制将 P 与 X 相加
        # 不需要将 P 用完，只需要使用到前 n_q 行，因为只要对 n_q 长的序列做位置编码
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X,**kwargs)

class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self, ffn_num_hiddens : int, ffn_num_outputs : int,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens, activation='relu')
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)
    
    def call(self, X : tf.Tensor):
        # X 的形状为 (batch_size, num_steps, num_hiddens)
        # 两个全连接层变换仅在最后一维上进行，因此变换与位置无关
        return self.dense2(self.dense1(X))

class AddNorm(tf.keras.layers.Layer):
    def __init__(self, norm_shape, dropout : float=0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)
    
    def call(self, X : tf.Tensor, Y : tf.Tensor, *args, **kwargs):
        # X 和 Y 的形状一样，均为 (batch_size, num_steps, num_hiddens)
        return self.ln(self.dropout(Y, **kwargs) + X)

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_hiddens : int, norm_shape, ffn_num_hiddens : int,
                 num_heads : int, dropout : float=0, use_bias : bool=False,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        # 子层一：多头注意力
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 子层二：基于位置的前馈网络
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def call(self, X : tf.Tensor, valid_lens, *args, **kwargs):
        # 做多头自注意力池化时，query、key、value 都是 X
        # 然后做 AddNorm
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        # 经过前馈网络后，再做 AddNorm
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

class TransformerEncoder(utils.Encoder):
    def __init__(self, vocab_size : int, num_hiddens : int, norm_shape, ffn_num_hiddens : int, 
                 num_heads : int, num_layers : int, dropout : float=0, use_bias : bool=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_embedding = PositionalEmbedding(num_hiddens, dropout=dropout)
        # 创建 num_layers 个 Block
        self.blocks = [
            EncoderBlock(num_hiddens, norm_shape, ffn_num_hiddens, num_heads, dropout, use_bias)
            for _ in range(num_layers)]
    
    def call(self, X : tf.Tensor, valid_lens, *args, **kwargs):
        # 先进行词嵌入，因为位置编码的值在 [-1,1] 之间
        # 因此嵌入向量先乘以嵌入维度的平方根进行一次缩放，然后再与位置编码相加
        X = self.embedding(X) * tf.math.sqrt(tf.cast(self.num_hiddens, tf.float32)) # 缩放
        X = self.pos_embedding(X, **kwargs) # 位置编码

        # 初始化注意力权重
        self.attn_weights = [None] * self.num_layers
        for i,block in enumerate(self.blocks):
            # 逐层计算，并添加到注意力权重列表中
            X = block(X, valid_lens, **kwargs)
            self.attn_weights[i] = block.attention.attention.attn_weights
        
        return X

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_hiddens : int, norm_shape, ffn_num_hiddens : int,
                 num_heads : int, index : int, dropout : float=0, use_bias : bool=False,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.index = index
        # 子层一：多头自注意力
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 子层二：编码器-解码器多头注意力
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 子层三：基于位置的前馈网络
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        
    def call(self, X : tf.Tensor, state, *args, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1] # state 包含了 enc_outputs 和 enc_valid_lens
        # 训练阶段，输出序列的所有词元都在同一时间处理
        # state[2][self.index] 初始化为 None
        # 预测阶段，输出序列是通过词元一个接一个预测得到
        # state[2][self.index] 包含着直到当前时间步的解码器第 index 个 Block 的输出表示
        if state[2][self.index] is None:
            key_values = X # 训练阶段，一次性可以拿到所有时间步的词元
        else:
            # 预测阶段。不断地将之前时间步累计的输出表示与 X 进行拼接
            # 没拼接一次，可以进行自注意力计算的 num_steps + 1
            key_values = tf.concat((state[2][self.index], X), axis=1)
        state[2][self.index] = key_values

        # 训练阶段
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # 解码器有效长度 dec_valid_lens 维度 (batch_size, num_steps)
            # 其中每一行的元素为 [1, 2, ..., num_steps]
            # 这用于进行掩蔽注意力
            dec_valid_lens = tf.repeat(tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)), 
                                       repeats=batch_size, axis=0)
        else:
            dec_valid_lens = None
        
        # 子层一：多头自注意力
        # 将 dec_valid_lens 作为掩盖，时间步 t' 的词元只会和之前的词元计算注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # 子层二：编码器-解码器多头注意力
        # enc_outputs 的形状为 (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        # 子层三：基于位置的前馈网络
        # 返回解码器的输出，和更新后的 state
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state

class TransformerDecoder(utils.Decoder):
    def __init__(self, vocab_size : int, num_hiddens : int, norm_shape, ffn_num_hiddens : int,
                 num_heads : int, num_layers : int, dropout : float=0, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 嵌入层和位置编码
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEmbedding(num_hiddens, dropout=dropout)
        # 创建 num_layers 个 Block
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(
                DecoderBlock(num_hiddens, norm_shape, ffn_num_hiddens, num_heads, index=i, dropout=dropout))
        # 输出层
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 初始化解码器所需的隐藏状态 state
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def call(self, X : tf.Tensor, state, **kwargs):
        # 先进行词嵌入和位置编码
        X = self.embedding(X) * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32))
        X = self.pos_encoding(X, **kwargs)

        # 初始化注意力权重，解码器包含有两种注意力
        # 子层一是多头自注意力，子层二是编码器-解码器多头注意力
        self._attn_weights = [[None] * len(self.blocks) for _ in range(2)]
        for i, block in enumerate(self.blocks):
            # 逐层计算，并添加到注意力权重列表中
            # 同时更新隐藏状态
            X, state = block(X, state, **kwargs)
            self._attn_weights[0][i] = block.attention1.attention.attn_weights
            self._attn_weights[1][i] = block.attention2.attention.attn_weights
        
        Y = tf.nn.softmax(self.dense(X)) # 输出层，并转换为概率分布
        return Y, state
    
    # 获取注意力权重
    @property
    def attention_weights(self):
        return self._attn_weights

def fetch_dec_attn_weights(dec_attn_weights, num_layers : int, num_heads : int, num_steps : int):
    dec_attention_weights_tmp = []
    # 依次拿出每一个预测时间步
    for step in dec_attn_weights:
        # 依次拿出每一种注意力，共有两种
        # 解码器的多头自注意力，编码器-解码器的多头注意力
        for attn in step:
            # 依次取出每一层的注意力权重
            for attn_layer in attn:
                # 依次取出每一个头的注意力权重
                for head in attn_layer:
                    # head 形状：(1, num_steps)
                    dec_attention_weights_tmp.append(head[0])
    # 由于随着推理进行，每个 head 的 num_steps 会逐渐增加
    # 借助 Pandas 的 DataFrame，可以很方便地将不同长度的注意力权重填充到同一个表格中
    # 长度不够的部分会自动用 NaN 填充，然后我们用 fillna(0) 将 NaN 替换为 0
    dec_attention_weights_tmp = tf.constant(
        (pd.DataFrame(dec_attention_weights_tmp).fillna(0).values).astype(np.float32))

    # 变换形状，拆分为 (-1, 2, num_layers, num_heads, num_steps)
    # -1 表示模型预测输出的词元长度 num_preds
    # 2 表示两种注意力，解码器自注意力 和 编码器-解码器注意力
    dec_attention_weights_tmp = tf.reshape(dec_attention_weights_tmp, (-1, 2, num_layers, num_heads, num_steps))
    # 用 transpose 改变轴排列为 (2, num_layers, num_heads, num_preds, num_steps)
    dec_attention_weights_tmp = tf.transpose(dec_attention_weights_tmp, (1, 2, 3, 0, 4))

    # 分别取出解码器自注意力 和 编码器-解码器注意力
    dec_self_attn_weights, enc_dec_attn_weights = dec_attention_weights_tmp[0], dec_attention_weights_tmp[1]

    return dec_self_attn_weights, enc_dec_attn_weights