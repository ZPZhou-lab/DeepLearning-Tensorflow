import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import re
from tqdm import tqdm
from source.code import ch4

def english_tokenize(lines : list, token : str="word"):
    """
    lines : list
        文本信息的列表
    token : str, default="word"
        分词方式，"word" 按单词分词，"char" 按字符分词
    """
    tokens = []
    for line in tqdm(lines):
        # 将所有英文字符转换为小写
        line = line.lower()
        # 去除两端空格和换行符
        line = line.strip()
        line = line.replace('\n', ' ')
        # 通过正则表达式，只保留 line 中的英文字符、英文标点符号 .,!?和空格
        line = re.sub('[^a-z,.!? ]', '', line)
        # 按单词分词
        if token == "word":
            # 按空格分词
            words1 = list(line.split(' '))
            # 还原空格,并将标点符号分开
            words = []
            for word in words1:
                if word == '':
                    continue
                elif word[-1] in [',', '.', '?', '!']:
                    words.append(word[:-1])
                    words.append(word[-1])
                else:
                    words.append(word)
        # 按字符分词
        elif token == "char":
            words = list(line)
        tokens.append(words)
    return tokens

def english_corpus_preprocessing(file : str, num_lines : int=645, 
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
    tokens = english_tokenize(lines[0:num_lines],token=token)
    # 构建词表
    vocab = ch4.Vocab(tokens, min_freq)
    # 创建语料
    if concat:
        corpus = [vocab[token] for line in tokens for token in line]
    else:
        corpus = [vocab[line] for line in tokens]
    
    return vocab, corpus

class TimeTravellerLoader:
    def __init__(self, file : str, token : str="word", min_freq : int=5, num_steps : int=5, 
                 num_lines : int=645, concat : bool=False,
                 use_random_iter : bool=True, batch_size : int=32) -> None:
        # 是否使用随机迭代器
        if use_random_iter:
            self.data_iter_fn = ch4.seq_data_iter_random
        else:
            self.data_iter_fn = ch4.seq_data_iter_sequential
        # 创建词元，词表，语料库
        self.vocab, self.corpus = english_corpus_preprocessing(
            file, num_lines, min_freq, token, concat)
        
        self.batch_size = batch_size
        self.num_steps = num_steps

    # 生成器函数
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps) 

#英文文本预测函数
def english_text_predict(prefix, num_preds, model, vocab, token : str="char"):
    # 去除两端空格和换行符
    prefix = prefix.lower().strip().replace('\n', ' ')
    # 通过正则表达式，只保留 line 中的英文字符、英文标点符号 .,!?以及空格
    prefix = re.sub('[^a-z,.!? ]', '', prefix)
    if token == 'word':
        words = list(prefix.split(' '))
        prefix = []
        for i in range(len(words)):
            if words[i] == '':
                continue
            elif words[i][-1] in [',', '.', '?', '!']:
                prefix.append(words[i][:-1])
                prefix.append(words[i][-1])
            else:
                prefix.append(words[i])
    
    state = model.begin_state(batch_size=1,dtype=tf.float32)
    # 初始化输出为前缀的第一个字符
    output = [vocab[prefix[0]]]

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
    if token == 'word':
        output_str = ' '.join([vocab.idx_to_token[i] for i in output])
    else:
        output_str = ''.join([vocab.idx_to_token[i] for i in output])
    return output_str