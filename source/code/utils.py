import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from IPython import display
from typing import Any, Union, Callable, Optional
import time

def gpu_limitation_config(memory : int=30, device : Union[int,list]=0):
    """
    ### 设置所使用的 GPU，以及 GPU 显存\n
    这在多人共用GPU时，能限制Tensorflow所使用的显存资源，方便多人共用GPU\n
    你可以指定所使用的 GPU 编号，以及所使用的显存大小\n

    Parameters
    ----------
    memory : int, default = 30
        设置所使用的GPU显存，单位GB，默认使用 30GB. \n
    device : int, default = 0
        设置所使用的 GPU 编号，默认使用第 0 块 GPU
    """
    # 设置所使用的 GPU
    if device is not None:
        if isinstance(device, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        elif isinstance(device, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in device])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 设置所使用的显存大小
    memory = min(memory, 30)
    GPUS = tf.config.list_physical_devices("GPU")

    tf.config.set_logical_device_configuration(
        GPUS[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1000*memory)]
    )

class Animator:
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), 
                 nrows=1, ncols=1, figsize=(5, 3), title : str=None):
        """
        Parameters
        ----------
        xlabel, ylabel, xlim, ylim, xscale, yscale
            横，纵坐标轴相关设置
        legend : list of str
            图例
        title : str
            图标标题
        nrows, ncols, figsize
            子图行数和列数，图像画布大小
        fmts : tuple
            图中每条线的格式配置，例如`g-.`表示用绿色(`green`)绘制点划线`-.`
        """
        # 设置绘图相关信息
        def set_axes(ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.legend(legend)
            ax.set_title(title)
            ax.grid()
            plt.tight_layout()
        
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
            legend = (legend,)
            fmts = (fmts,)
        else:
            self.axes = self.axes.flatten()
        # lambda 函数将配置参数的信息保存到 set_axes() 函数
        self.config_axes = lambda ax : \
            set_axes(self.axes[ax], xlabel, ylabel, xlim, ylim, xscale, yscale, legend[ax], title)
        self.X, self.Y = [None for _ in self.axes], [None for _ in self.axes]
        self.fmts = fmts # 初始化

    def add(self, x, y : list, ax : int=0):
        """
        在现有的图上添加新的点
        """
        # 如果 y 不是序列类型，就转换为列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y) # 共有 n 条线

        # 如果 x 不是序列类型，就转换为列表
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        # 初始化 X
        if not self.X[ax]:
            self.X[ax] = [[] for _ in range(n)] 
        # 初始化 Y
        if not self.Y[ax]:
            self.Y[ax] = [[] for _ in range(n)] 
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[ax][i].append(a) # 添加横坐标
                self.Y[ax][i].append(b) # 添加纵坐标

        self.axes[ax].cla() # 清空画布
        for x, y, fmt in zip(self.X[ax], self.Y[ax], self.fmts[ax]):
            self.axes[ax].plot(x, y, fmt) # 绘制曲线
        
        self.config_axes(ax) # 配置画布
        display.display(self.fig) # 展示画布
        display.clear_output(wait=True) # 延迟清除


def show_images(images : list, labels : list, pred_labels : list=None, 
                nrows: int=1, ncols : int=5, figsize : tuple=None, label_names : list=None):
    if figsize is None:
        figsize = (2.5*ncols,2.5*nrows) # 设置 figsize
    # 创建子图
    fig, ax = plt.subplots(nrows,ncols,figsize=figsize) 
    ax = ax.flatten()

    if pred_labels is None:
        pred_labels = [None for _ in range(len(labels))]

    # 将标签的编码转换为标签名称
    if label_names is not None:
        labels = [label_names[i] for i in labels]
        pred_labels = [label_names[i] if i is not None else None for i in pred_labels]
    
    # 在每个子图上绘制图像，并展示标签
    for i, (image,label,pred) in enumerate(zip(images,labels,pred_labels)):
        ax[i].imshow(image)
        ax[i].set_title("Label: %s\nPred: %s"%(label,pred))

    plt.tight_layout()


def plot_confusion_matrix():
    ...

def classification_predict(model : tf.keras.Model, inputs : tuple, 
                           batch_size : int=32, label : bool=True):
    # 创建批量数据生成器
    data_iter = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)

    y_pred = []
    for batch in data_iter:
        y_pred.append(model(*batch))
    
    # 拼接预测结果
    y_pred = tf.concat(y_pred,axis=0)

    # 是否将预测概率转换为标签
    if label:
        y_pred = tf.argmax(y_pred,axis=-1)
    
    return y_pred

def apply_augmentation(image, aug, nrows : int=2, ncols : int=4):
    """
    对图像`image`进行`aug`变换，用于演示
    """
    images = [aug(image,training=True) for _ in range(ncols*nrows)]
    fig,ax = plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
    ax = ax.flatten()
    for i,img in enumerate(images):
        ax[i].imshow(img)
    plt.tight_layout()

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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    
    def call(self, inputs, *args, **kwargs):
        raise NotImplementedError
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, enc_valid_len=None, *args, **kwargs):
        raise NotImplementedError

    def call(self, inputs, state, *args, **kwargs):
        raise NotImplementedError

class EncoderDecoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, enc_inputs, dec_inputs, *args, **kwargs):
        # 编码器负责将输入序列编码为隐状态
        enc_outputs = self.encoder(enc_inputs, *args, **kwargs)
        # 解码器负责根据编码器的输出和输入序列来预测输出序列
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_inputs, dec_state, **kwargs)

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

# 在序列中屏蔽不相关的项
def sequence_mask(X : tf.Tensor, valid_len : tf.Tensor, value : float = 0):
    # X 形状：(batch_size, num_steps) 或者 (batch_size, num_steps, p)
    # valid_len 形状：(batch_size,)
    maxlen = X.shape[1] # 序列长度
    
    # lens 生成从 0 到 maxlen-1 的序列
    lens = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] # 形状：(1, num_steps)
    # valid_len[:, None] 将 valid_len 的形状变换为 (batch_size, 1)
    valid_len = tf.cast(valid_len[:, None], dtype=tf.float32) # 形状：(batch_size, 1)
    # 利用广播机制生成掩码 mask，形状：(batch_size, num_steps)
    # mask[i, j] = 1 if j < valid_len[i]; 0 otherwise
    mask = lens < valid_len
    
    # 分两种情况，将 mask 为 0 的元素替换为 value
    # 1. X 是一个三维张量，形状 (batch_size, num_steps, p)
    # 需要把 mask 也变成三维张量，形状 (batch_size, num_steps, 1)
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    # 2. X 是一个二维张量，形状 (batch_size, num_steps)
    # 则保持 mask 是一个二维张量
    else:
        return tf.where(mask, X, value)

class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    def __init__(self, valid_len, **kwargs):
        # 初始化父类参数 reduction='none'，表示不对损失求均值
        super().__init__(reduction='none',**kwargs)
        self.valid_len = valid_len
    
    def call(self, y_true, y_pred):
        # y_true 形状：(batch_size, num_steps)
        # y_pred 形状：(batch_size, num_steps, vocab_size)
        # 初始化权重，形状：(batch_size, num_steps)
        weights = tf.ones_like(y_true, dtype=tf.float32)
        # 通过 sequence_mask 函数将不相关项的权重设为 0
        weights = sequence_mask(weights, self.valid_len)

        # 计算无掩码的交叉熵损失
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        unweighted_loss = loss_func(y_true, y_pred) # 形状：(batch_size, num_steps)

        # 将不相关项的损失也设为 0
        # 为了演示，我们这里仅对时间步聚合，保留批量大小维度
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss

def train_seq2seq(model, data_iter, tgt_vocab, Epochs : int=100, lr : float=0.01, verbose : int=1):
    # 初始化优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[1, Epochs], legend=['train'])

    # 存储每个迭代周期的损失和样本量
    loss_batch, samples_batch = 0, 0
    # 记录单词处理速度
    speeds = []

    for epoch in range(Epochs):
        start = time.time() # 计时开始
        for batch in data_iter:
            # 分别拿到四个元素：编码器输入、编码器输入有效长度、解码器输入、解码器输入有效长度
            # X, Y 形状：(batch_size, num_steps)
            # X_valid_len, Y_valid_len 形状：(batch_size,)
            X, X_valid_len, Y, Y_valid_len = batch
            # 为解码器的输入添加 <bos>，形状：(batch_size, 1)
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']]*Y.shape[0]), (-1, 1))
            # 去掉解码器输入的最后一个时间步，在开头加上 <bos>，保持形状不变
            dec_input = tf.concat([bos, Y[:, :-1]], axis=1)

            with tf.GradientTape() as tape:
                # 进行预测，计算损失
                # 注意我们将 X_valid_len 传入给模型，它作为 *args 的一部分
                # 但目前我们的模型并没有使用它，这在后续章节中会改变
                Y_hat, _ = model(X, dec_input, X_valid_len, training=True)
                loss_func = MaskedSoftmaxCELoss(Y_valid_len)
                loss = loss_func(Y, Y_hat) # 形状：(batch_size,)
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            grads = grad_clipping(grads, 1) # 梯度裁剪
            optimizer.apply_gradients(zip(grads, weights))

            # 将该批量的损失函数值加到总损失函数值上
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            loss_batch += tf.reduce_sum(loss).numpy()
            samples_batch += num_tokens
        
        end = time.time() # 计时结束
        speeds.append(samples_batch / (end - start))

        if epoch == 0 or (epoch + 1) % verbose == 0:
            # 计算困惑都
            ce = tf.math.exp(loss_batch / samples_batch).numpy()
            animator.add(epoch + 1, [ce])
    
    print(f"平均 {np.mean(speeds):.1f} 词元/秒")
    return model

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

def show_attention(attn_weights, xlabel, ylabel, titles=None, figsize=(3,3), cmap="Reds"):
    """
    Parameters
    ----------
    attn_weights : tf.Tensor
        注意力权重，形状为 (nrows, ncols, n_q, n_k)\n
        该张量同时保存多个注意力权重，这样可以将每个注意力权重画到一个子图上
    """
    nrows, ncols = attn_weights.shape[0], attn_weights.shape[1]
    # 创建子图
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,sharex=True,sharey=True,squeeze=False)

    # 依次拿出每一行
    for i, (row_axes, row_weights) in enumerate(zip(axes, attn_weights)):
        # 依次拿出每一列
        for j, (ax, mat) in enumerate(zip(row_axes, row_weights)):
            im = ax.imshow(mat.numpy(), cmap=cmap) # 绘制热力图
            if i == nrows - 1:
                ax.set_xlabel(xlabel) # 设置坐标轴标签
            if j == 0:
                ax.set_ylabel(ylabel) # 设置坐标轴标签
            if titles is not None:
                ax.set_title(titles[j]) # 设置标题
    # 添加图例颜色条
    fig.colorbar(im, ax=axes, shrink=0.6)
    return axes

def Masked_Softmax(X : tf.Tensor, valid_lens=None):
    """
    添加掩蔽的 Softmax 操作

    Parameters
    ----------
    X : tf.Tensor
        三维张量，形状为 (batch_size, n_q, n_k)，对应 (batch_size, query 个数, key 个数)
    valid_lens : tf.Tensor
        一维张量或二维张量，形状为 (batch_size, ) 或 (batch_size, n_q)\n
        一维说明批量中的每个 query 样本使用相同的有效长度\n
        二维说明批量中的每个 query 样本使用不同的有效长度\n
    """
    # 不进行掩蔽
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        # 如果 valid_lens 是一维张量，则它只有 batch_size 维度
        # 将其重复 n_q 次，大小变为 (batch_size*n_q,)
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        # 如果 valid_lens 是二维张量，将它展开成一维张量
        # 大小变为 (batch_size*n_q,)
        else:
            valid_lens = tf.reshape(valid_lens, shape=(-1,))
        
        # 被掩蔽的元素使用一个非常大的负值 -1e6 替换，从而其 Softmax 输出为 0
        # 可以借用上一章定义的 sequence_mask 函数
        # 先将 X 的形状变成 (batch_size * n_q, n_k)，便于与 valid_lens 操作
        X = sequence_mask(X=tf.reshape(X, shape=(-1, shape[-1])), 
                          valid_len=valid_lens, value=-1e6)

        # 还原回原来的形状，并在最后一轴，即 n_k 所在维度上做 Softmax 操作
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

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
        self.attn_weights = Masked_Softmax(scores, valid_lens) # 注意力权重
        
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
        self.attn_weights = Masked_Softmax(scores, valid_lens) # 注意力权重

        # 计算注意力池化结果
        # (batch_size, n_q, n_k) @ (batch_size, n_v = n_k, v) -> (batch_size, n_q, v)
        Y = self.dropout(self.attn_weights, **kwargs) @ values
        return Y

class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # 获取当前解码器的注意力权重
    @property
    def attention_weights(self):
        raise NotImplementedError

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

def english_tokenize(lines : list, token : str="word"):
    """
    lines : list
        存储文本信息的列表
    token : str, default="word"
        分词方式，"word" 表示按词分词，"char" 表示按字符分词
    """
    tokens = []
    # 依次取出每一行
    # tqdm 可以显示进度条
    for line in lines:
        # 去除空格，去除换行符
        line = line.replace('\n', '').lower()
        # 分词
        if token == "word":
            words = line.split() # 按词分词
        elif token == "char":
            words = list(line) # 按字符分词
        tokens.append(words) # 添加到 tokens 中
    return tokens

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
    
def plot_confusion_matrix(y_true, y_pred, labels, figsize : tuple=(6,6), xrot : float=0, yrot : float=0):
    """
    plot_confusion_matrix(y_true, y_pred)
        绘制混淆矩阵
        
    Parameters
    ----------
    y_true : np.ndarray
        数据的真实标签
    y_pred : np.ndarray
        模型的预测结果
    labels : list
        各个类别的含义
    """
    import itertools

    acc = accuracy_score(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    print("accuracy: %.4f"%(acc))
    
    # 绘制混淆矩阵
    fig = plt.figure(figsize=figsize)
    plt.imshow(mat,cmap=plt.cm.Blues)
    
    thresh = mat.max() / 2
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        # 在每个位置添加上样本量
        plt.text(j, i, mat[i, j],
                 horizontalalignment="center",
                 color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xticks(range(mat.shape[0]),labels,rotation=xrot)
    plt.yticks(range(mat.shape[0]),labels,rotation=yrot)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')