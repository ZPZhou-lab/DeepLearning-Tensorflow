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


def load_cora(path : str):
    # 读取节点信息
    vertex, vertex_feat, vertex_class = [], [], []
    class_map = {} # 记录类别和类别索引

    with open(os.path.join(path, 'cora.content'), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # 获取节点编号、节点特征、节点类别
            node, feat, label = int(line[0]), line[1:-1], line[-1]
            feat = [int(x) for x in feat]

            if label not in class_map:
                class_map[label] = len(class_map)
            
            vertex.append(node)
            vertex_feat.append(feat)
            vertex_class.append(class_map[label])
    
    # 读取边信息
    edges = []

    with open(os.path.join(path, 'cora.cites'), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # 获取边的两个节点编号
            node1, node2 = int(line[0]), int(line[1])
            # 三元组用于存储边信息，(node1, node2, 边特征)
            edges.append((node1, node2, None))

    # 对边进行排序
    # 先按照 node1 排序，再按照 node2 排序
    edges = sorted(edges, key=lambda x: (x[0], x[1]))
    
    # 记录每个节点边的起始位置，以便于快速查询
    vertex_edge_loc = {}
    for i, edge in enumerate(edges):
        if edge[0] not in vertex_edge_loc:
            vertex_edge_loc[edge[0]] = [i, i + 1]
        else:
            vertex_edge_loc[edge[0]][1] = i + 1
    
    return vertex, vertex_feat, vertex_class, edges, vertex_edge_loc, class_map

def prepare_graph_info_cora(vertex, edges, num_neighbors : int=25):
    """
    Parameters
    ----------
    vertex : list
        节点列表
    edges : list
        边列表，每个元素为 (node1, node2, edge_feat)
    num_neighbors : int
        每个节点的采样或填充的邻居数量
    """
    # 为每个节点设置索引
    vertex_idx = {}
    for i,node in enumerate(vertex):
        vertex_idx[node] = i

    # 将边的连接关系转换为其索引的表示
    edges_idx = [(vertex_idx[edge[0]], vertex_idx[edge[1]], edge[2]) for edge in edges]
    # 进行排序，先按照第一个节点索引排序，再按照第二个节点索引排序
    edges_idx = sorted(edges_idx, key=lambda x: (x[0], x[1]))

    # 用列表存储每个节点的邻居在 edges_idx 中的起始位置
    neighbors = [[-1, -1] for _ in range(len(vertex))]
    for i,(node1,_,_) in enumerate(edges_idx):
        if neighbors[node1][0] == -1:
            neighbors[node1][0] = i
        neighbors[node1][1] = i + 1

    # 为每个节点构造邻居节点的索引，相连接的边的索引
    # 邻居数小于 num_neighbors 的节点用 0 填充
    # 邻居数大于 num_neighbors 的节点通过随机采样来截断
    # valid_len 保存每个节点的有效邻居数
    neighbors_idx = [[0] * num_neighbors for _ in range(len(vertex))]
    connected_edges_idx = [[0] * num_neighbors for _ in range(len(vertex))]
    valid_lens = [0] * len(vertex)

    for i,(start,end) in enumerate(neighbors):
        # 获取节点的邻居节点索引
        node_neighbors_idx = [edge[1] for edge in edges_idx[start:end]]
        valid_lens[i] = min(len(node_neighbors_idx), num_neighbors) # 记录节点的有效邻居数

        if end - start > num_neighbors:
            # 邻居数大于 num_neighbors 的节点通过随机采样来截断
            connected_edges_idx[i] = np.random.choice(list(range(start,end)), num_neighbors, replace=False)
            neighbors_idx[i] = [edges_idx[idx][1] for idx in connected_edges_idx[i]]
        else:
            # 邻居数小于 num_neighbors 的节点用 0 填充
            neighbors_idx[i][:end-start] = node_neighbors_idx[:]
            connected_edges_idx[i][:end-start] = list(range(start, end))
    
    return vertex_idx, edges_idx, neighbors_idx, connected_edges_idx, valid_lens

def load_cora_nodetask_data(path : str, num_neighbors : int=25, test_size : float=0.4):
    from sklearn.model_selection import train_test_split

    # 加载 cora 数据集  
    vertex, vertex_feat, vertex_class, edges, vertex_edge_loc, class_map = load_cora(path=path)
    # 构造全图信息
    vertex_idx, edges_idx, neighbors_idx, connected_edges_idx, valid_lens \
        = prepare_graph_info_cora(vertex, edges, num_neighbors=num_neighbors)

    # 从 edges_idx 中获取边的特征
    edges_feat = [edge[2] for edge in edges_idx]
    edges_idx = [(edge[0], edge[1]) for edge in edges_idx]

    # 构造图，并将所有信息转换为张量
    graph = {
        # 图上的特征
        'vertex_feat': tf.constant(vertex_feat, dtype=tf.int32),
        'edges_feat': None, # cora 数据集没有边的特征
        'graph_feat': None, # cora 数据集没有图的特征
        # 图的连接信息
        'vertex_idx': vertex_idx, # 节点名称到索引的映射
        'edges_idx': tf.constant(edges_idx, dtype=tf.int32),
        'neighbors_idx': tf.constant(neighbors_idx, dtype=tf.int32),
        'connected_edges_idx': tf.constant(connected_edges_idx, dtype=tf.int32),
        'valid_lens': tf.constant(valid_lens, dtype=tf.int32),
    }
    # 节点的标签
    node_labels = tf.constant(vertex_class, dtype=tf.int32)

    # 切分训练集和验证集节点
    train_nodes, valid_nodes = train_test_split(list(range(len(vertex))), test_size=test_size, random_state=42)

    return graph, node_labels, train_nodes, valid_nodes, class_map

# 给定邻接矩阵，计算图Laplace算子
def graph_laplacian(A):
    # 计算度矩阵
    D = tf.reduce_sum(A, axis=1)
    D = tf.linalg.diag(D) # 转换为对角矩阵

    # 计算Laplace算子
    L = D - A
    return L

# 图Laplace算子 和 多项式滤波器
class GraphLaplace(tf.keras.Model):
    def __init__(self, d : int=1, num_layers : int=1, *args, **kwargs):
        super(GraphLaplace, self).__init__(*args, **kwargs)
        self.d = d # 多项式滤波器的阶数
        self.num_layers = num_layers # GNN 层数
        self.operator = None # 多项式滤波器算子

        # 初始化多项式滤波器的系数
        self.w = []
        for _ in range(self.num_layers):
            # 多项式阶数从 0 到 d，共 d+1 个系数
            self.w.append(tf.Variable(tf.random.normal(shape=(self.d + 1, 1)), dtype=tf.float32))
    
    # 计算多项式滤波器算子
    def prepare_operator(self, A):
        num_nodes = tf.shape(A)[0]

        # 计算图Laplace算子
        L = graph_laplacian(A)
        # 对图Laplace算子进行归一化
        lmax = tf.reduce_max(tf.linalg.eigvalsh(L))
        L = 2 * L / lmax - tf.eye(num_nodes, dtype=tf.float32)

        # 计算多项式滤波器算子
        operator = [tf.eye(num_nodes, dtype=tf.float32)]
        for i in range(self.d):
            operator.append(tf.matmul(operator[-1], L))
        
        return operator
    
    # 重置多项式滤波器算子
    def reset_operator(self):
        self.operator = None

    def call(self, X, A, **kwargs):
        # X : 节点特征矩阵，形状 (num_nodes, num_features)
        # A : 邻接矩阵，形状 (num_nodes, num_nodes)

        # 如果多项式滤波器算子还没有计算，就先计算
        if self.operator is None:
            self.operator = self.prepare_operator(A)
                
        # 计算多项式滤波器的输出
        for i in range(self.num_layers):
            # (num_nodes, num_nodes) @ (num_nodes, num_features) = (num_nodes, num_features)
            pw_L = tf.reduce_sum([self.w[i][j] * op for j,op in enumerate(self.operator)], axis=0)
            X = tf.nn.tanh(pw_L @ X) # 做一次非线性激活
        
        return X
    
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, num_hidddens : int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(GCNLayer, self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.W_dense = tf.keras.layers.Dense(num_hidddens, use_bias=False) # 对邻居节点特征进行投影
        self.B_dense = tf.keras.layers.Dense(num_hidddens, use_bias=False) # 对自身节点特征进行投影

    def call(self, graph, *args, **kwargs):
        # graph : 存储图信息的字典
        # 包含 vertex_feat, edges_feat, graph_feat, edges_idx, neighbors_idx, connections_idx, valid_lens

        # 汇聚邻居节点特征，形状 (num_nodes, num_neighbors, num_features)
        neighbors_feat = tf.gather(graph["vertex_feat"], graph["neighbors_idx"])
        neighbors_feat = tf.reduce_sum(neighbors_feat, axis=1) # 汇聚后，形状 (num_nodes, num_features)

        valid_lens = graph["valid_lens"][:,None] # 添加一个维度，形状 (num_nodes, 1)
        valid_mask = tf.cast(tf.greater(valid_lens, 0), dtype=tf.float32) # 有邻居的节点的掩码
        valid_lens = tf.where(tf.equal(valid_lens, 0), tf.ones_like(valid_lens), valid_lens) # 避免除以 0
        
        neighbors_feat = neighbors_feat / tf.cast(valid_lens, dtype=tf.float32) * valid_mask # 归一化

        # 信息传递，汇聚得到新的节点特征
        vertex_feat = self.W_dense(neighbors_feat,**kwargs) + self.B_dense(graph["vertex_feat"],**kwargs) # (num_nodes, num_hidddens)
        vertex_feat = tf.nn.relu(vertex_feat) # 激活函数

        graph["vertex_feat"] = vertex_feat # 更新节点特征
        return graph
    
class GCNModel(tf.keras.Model):
    def __init__(self, num_hidddens : int, num_classes : int, num_layers : int, *args, **kwargs):
        super(GCNModel, self).__init__(*args, **kwargs)
        self.num_hidddens = num_hidddens
        # 创建节点特征的嵌入层
        self.node_embed = tf.keras.layers.Dense(num_hidddens, use_bias=False)

        # 创建多层 GCN 层
        self.gcn_layers = [GCNLayer(num_hidddens) for _ in range(num_layers)]
        
        # 创建分类器
        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hidddens, activation="sigmoid"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax") # 转换为概率分布
        ])
    
    def call(self, graph, **kwargs):
        # graph : 存储图信息的字典
        # 包含 vertex_feat, edges_feat, graph_feat, edges_idx, neighbors_idx, connections_idx, valid_lens

        # 节点特征的嵌入，形状从 (num_nodes, vocab_size) 变为 (num_nodes, num_hidddens)
        graph["vertex_feat"] = tf.cast(graph["vertex_feat"], dtype=tf.float32)
        graph["vertex_feat"] = self.node_embed(graph["vertex_feat"], **kwargs) \
            / tf.reduce_sum(graph["vertex_feat"], axis=-1, keepdims=True) # 归一化，避免数值过大

        # 多层 GCN 层提取特征
        for layer in self.gcn_layers:
            graph = layer(graph, **kwargs)
        
        # 下游网络负责节点分类
        node_probs = self.classifier(graph["vertex_feat"], **kwargs)
        return node_probs