import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from . import utils

import random
import pyrfume
from collections import Counter
import collections
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem

import time
from tqdm import tqdm
import os


AtomMap = {"C": 0, "N": 1, "O": 2, "S": 3}
BondMap = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

def load_cora(path : str):
    import os
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
            # [0] 表示边的特征，这里没有边的特征
            edges.append((node1, node2, [0]))

    return vertex, vertex_feat, vertex_class, edges, class_map

def compute_vertex_edge_loc(vertex, edges):
    """
    ### 计算每个节点邻居边的起始位置
        重新将边 `edges` 按照节点名称排序，然后计算每个节点邻居边的在 edges 中的起始位置 `vertex_edge_loc`\n
        这样当给定节点 `v` 时，可以通过 `vertex_edge_loc[v]` 得到 `v` 的邻居边在 `edges` 中的起始位置\n
    """

    # 对边进行排序
    # 先按照 node1 排序，再按照 node2 排序
    edges = sorted(edges, key=lambda x: (x[0], x[1]))
    
    # 记录每个节点边的起始位置，以便于快速查询
    # 默认值为 [-1, -1]，表示该节点没有邻居边
    vertex_edge_loc = {node : [-1,-1] for node in vertex}
    for i, edge in enumerate(edges):
        if vertex_edge_loc[edge[0]][0] == -1:
            vertex_edge_loc[edge[0]][0] = i
        vertex_edge_loc[edge[0]][1] = i + 1
    
    return edges, vertex_edge_loc

def build_global_graph(vertex, edges, num_neighbors : int=25, is_undirectional : bool=True):
    """
    Parameters
    ----------
    vertex : list
        节点列表
    edges : list
        边列表，每个元素为 (node1, node2, edge_feat)
    num_neighbors : int
        每个节点的采样或填充的邻居数量
    is_undirectional : bool
        是否为无向图
    """
    # 无向图，边是双向的，所以需要将边复制一份，交换两个节点的位置
    if is_undirectional:
        edges = edges + [(edge[1], edge[0], edge[2]) for edge in edges]
    
    # 为每个节点设置索引
    vertex_idx = {}
    for i,node in enumerate(vertex):
        vertex_idx[node] = i

    # 将边的连接关系转换为其索引的表示
    edges_idx = [(vertex_idx[edge[0]], vertex_idx[edge[1]], edge[2]) for edge in edges]
    # 找到每个节点邻居在 edges_idx 中的起始位置
    edges_idx, vertex_edge_loc = compute_vertex_edge_loc(vertex=vertex_idx.values(), edges=edges_idx)

    # 为每个节点构造邻居节点的索引，相连接的边的索引
    # 邻居数小于 num_neighbors 的节点用 0 填充
    # 邻居数大于 num_neighbors 的节点通过随机采样来截断
    # valid_len 保存每个节点的有效邻居数
    neighbors_idx = [[0] * num_neighbors for _ in range(len(vertex))]
    connected_edges_idx = [[0] * num_neighbors for _ in range(len(vertex))]
    valid_lens = [0] * len(vertex)

    for i, (node_idx, (start, end)) in enumerate(vertex_edge_loc.items()):
        # 获取节点的邻居节点索引
        node_neighbors_idx = [edges_idx[idx][1] for idx in range(start, end)]
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

def load_cora_nodetask_data(path : str, num_neighbors : int=25, is_undirectional : bool=True, test_size : float=0.4):
    from sklearn.model_selection import train_test_split

    # 加载 cora 数据集
    vertex, vertex_feat, vertex_class, edges, class_map = load_cora(path=path)
    # 构造全图信息
    vertex_idx, edges_idx, neighbors_idx, connected_edges_idx, valid_lens \
        = build_global_graph(vertex, edges, num_neighbors=num_neighbors, is_undirectional=is_undirectional)

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

def node_subgraph_sampling(vertex, edges, vertex_feat, k_hops : int=1, 
                           num_neighbors : int=20, is_undirectional : bool=True):
    graphs = [] # 存储子图的列表
    
    # 为每个节点设置索引
    vertex_idx = {}
    for i,node in enumerate(vertex):
        vertex_idx[node] = i
    
    # 每个节点所连接的边的索引
    edges, vertex_edge_loc = compute_vertex_edge_loc(vertex, edges)
    
    # 依次以每个节点为中心，采样其 k_hops 邻居构成子图
    for i,node in enumerate(vertex):
        subgraph = {} # 存储子图信息的字典

        sub_vertex = [node] # 创建该子图的节点索引列表
        sub_edges = [] # 创建该子图的边索引列表

        # 该子图的节点特征
        sub_vertex_feat = [vertex_feat[i]]

        # 递归地采样 k_hops 邻居
        end = 0
        for _ in range(k_hops):
            start, end = end, len(sub_vertex)
            for j in range(start, end):
                node_ = sub_vertex[j] # 当前处理的节点
                
                # 获取节点的邻居节点
                l, r = vertex_edge_loc.get(node_, [-1, -1])

                # 如果邻居数超过 num_neighbors，则随机采样 num_neighbors 个邻居
                neighbors_list = list(range(l,r))
                if len(neighbors_list) > num_neighbors:
                    neighbors_list = np.random.choice(neighbors_list, num_neighbors, replace=False)

                for idx in neighbors_list:
                    # edges[idx][1] 是节点 node_ 的邻居节点
                    from_node = edges[idx][1]

                    # 防止重复添加节点
                    if from_node not in sub_vertex:
                        # 添加节点，以及节点的特征
                        sub_vertex.append(from_node)
                        sub_vertex_feat.append(vertex_feat[vertex_idx[from_node]])

                    # 获取节点相连的边
                    sub_edges.append(edges[idx])
                
        # 构造子图        
        sub_vertex_idx, sub_edges_idx, sub_neighbors_idx, sub_connected_edges_idx, sub_valid_lens \
            = build_global_graph(vertex=sub_vertex, edges=sub_edges, num_neighbors=num_neighbors, is_undirectional=is_undirectional)
        # 拆分出边的特征
        sub_edges_feat = [edge[2] for edge in sub_edges_idx]
        sub_edges_idx = [(edge[0], edge[1]) for edge in sub_edges_idx]

        # 添加到子图信息中
        subgraph["vertex_feat"] = sub_vertex_feat
        subgraph["edges_feat"] = sub_edges_feat
        subgraph["edges_idx"] = sub_edges_idx
        subgraph["neighbors_idx"] = sub_neighbors_idx
        subgraph["connected_edges_idx"] = sub_connected_edges_idx
        subgraph["valid_lens"] = sub_valid_lens

        graphs.append(subgraph) # 将子图添加到列表中
    
    return graphs

class GraphDataLoader:
    def __init__(self, graphs, batch_size : int=64, shuffle : bool=False, 
                 num_node_feats : int=None, num_edge_feats : int=None, num_neighbors : int=None) -> None:
        """
        Parameters
        ----------
        graphs : list
            存储子图信息的列表，每个元素为一个字典，字典中包含子图的信息
        batch_size : int, default = 64
            批量大小
        shuffle : bool, default = False
            在迭代过程中是否打乱数据
        num_node_feats : int, default = None
            节点特征的维度，默认为 `None`，从图数据中自动获取
        num_edge_feats : int, default = None
            边特征的维度，默认为 `None`，从图数据中自动获取
        num_neighbors : int, default = None
            每个节点的邻居数，默认为 `None`，从图数据中自动获取
        """
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.graph_keys = graphs[0].keys()

        # 准备填充的内容
        self.num_node_feats = len(graphs[0]["vertex_feat"][0]) if num_node_feats is None else num_node_feats
        self.num_neighbors = len(graphs[0]["neighbors_idx"][0]) if num_neighbors is None else num_neighbors

        # 对于孤立点，有可能 edges_feat 是空列表，此时访问 graphs[0]["edges_feat"][0] 会报错
        # 因此需要对 edges_feat 进行特殊处理，通过循环找到第一个非空列表，获取特征维度
        if num_edge_feats is None:
            for graph in graphs:
                if len(graph["edges_feat"]) > 0:
                    self.num_edge_feats = len(graph["edges_feat"][0])
                    break
        else:
            self.num_edge_feats = num_edge_feats

        # 在填充时，每个图信息要填充的内容
        self.pad = {
            "vertex_feat": [0] * self.num_node_feats,
            "edges_feat": [0] * self.num_edge_feats,
            "edges_idx": (-1, -1),
            "neighbors_idx": [0] * self.num_neighbors,
            "connected_edges_idx": [0] * self.num_neighbors,
            "valid_lens": 0
        }
    
    # 类的下标访问方法
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    # 类的 len 方法 
    def __len__(self):
        return self.num_graphs
    
    # 填充辅助函数，用变量 pad 填充列表 info，使其长度为 max_len
    def pad_info(self, info : list, pad, max_len : int):
        info += [pad for _ in range(max_len - len(info))]
        return info
    
    def create_dataset(self):
        def padded_batch_generator():
            # 如果 shuffle 为 True，则打乱数据
            idx = np.random.permutation(self.num_graphs) if self.shuffle else np.arange(self.num_graphs)

            for i in range(0, self.num_graphs, self.batch_size):
                # 选取 batch_size 个图
                graph_batch = {}
                for key in self.graph_keys:
                    # 这里用 copy() 是为了防止修改原始数据
                    graph_batch[key] = [self.graphs[j][key].copy() for j in idx[i:i+self.batch_size]]
                
                # 进行填充
                num_sub_nodes = max([len(x) for x in graph_batch["vertex_feat"]]) # 子图中最大节点数
                num_sub_edges = max([len(x) for x in graph_batch["edges_feat"]]) # 子图中最大边数
                
                batch_size = len(graph_batch["vertex_feat"]) # 当前 batch 中图的数量

                valid_nodes, valid_edges = [], [] # 记录每个图的有效节点数和有效边数
                
                # 节点信息填充
                for b in range(batch_size):
                    valid_nodes.append(len(graph_batch["vertex_feat"][b]))
                    for key in ["vertex_feat", "neighbors_idx", "connected_edges_idx", "valid_lens"]:
                        graph_batch[key][b] = self.pad_info(graph_batch[key][b], self.pad[key], num_sub_nodes)
                        
                # 边信息填充
                for b in range(batch_size):
                    valid_edges.append(len(graph_batch["edges_idx"][b]))
                    for key in ["edges_feat", "edges_idx"]:
                        graph_batch[key][b] = self.pad_info(graph_batch[key][b], self.pad[key], num_sub_edges)
                        
                # 将填充后的列表拼装成张量
                for key in self.graph_keys:
                    graph_batch[key] = tf.stack(graph_batch[key])
                graph_batch["valid_nodes"] = tf.constant(valid_nodes)
                graph_batch["valid_edges"] = tf.constant(valid_edges)
                
                yield graph_batch
        
        # 创建 TensorFlow 数据集
        dataset = tf.data.Dataset.from_generator(padded_batch_generator, 
            output_signature={
                "vertex_feat": tf.TensorSpec(shape=(None, None, self.num_node_feats), dtype=tf.float32),
                "edges_feat": tf.TensorSpec(shape=(None, None, self.num_edge_feats), dtype=tf.float32),
                "edges_idx": tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
                "neighbors_idx": tf.TensorSpec(shape=(None, None, self.num_neighbors), dtype=tf.int32),
                "connected_edges_idx": tf.TensorSpec(shape=(None, None, self.num_neighbors), dtype=tf.int32),
                "valid_lens": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "valid_nodes": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "valid_edges": tf.TensorSpec(shape=(None,), dtype=tf.int32)
            })
        # prefetch 通过异步的方式让数据集准备好，提高效率
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

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

        valid_lens = graph["valid_lens"] # 形状 (num_nodes, )
        mask = tf.sequence_mask(valid_lens, tf.shape(neighbors_feat)[1], dtype=tf.float32) # 生成掩码，形状 (num_nodes, num_neighbors)
        mask = tf.expand_dims(mask, axis=-1) # 扩展维度，形状 (num_nodes, num_neighbors, 1)，便于和特征维度进行广播
        valid_lens = tf.where(tf.equal(valid_lens, 0), tf.ones_like(valid_lens), valid_lens) # 避免除以 0
        
        # 乘以掩码，使得无效邻居节点特征为 0，并聚合邻居节点特征，形状 (num_nodes, num_features)
        neighbors_feat = tf.reduce_sum(neighbors_feat * mask, axis=1) 
        valid_lens = tf.where(tf.equal(valid_lens, 0), tf.ones_like(valid_lens), valid_lens) # 防止除以 0
        neighbors_feat = neighbors_feat / tf.expand_dims(valid_lens, axis=-1) # 归一化，形状 (num_nodes, num_features)

        # 信息传递，汇聚得到新的节点特征
        vertex_feat = self.W_dense(neighbors_feat,**kwargs) + self.B_dense(graph["vertex_feat"],**kwargs) # (num_nodes, num_hidddens)
        vertex_feat = tf.nn.relu(vertex_feat) # 激活函数

        graph["vertex_feat"] = vertex_feat # 更新节点特征
        return graph

class GATLayer(tf.keras.layers.Layer):
    def __init__(self, num_hiddens : int, num_heads : int, dropout : float, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        # 多头注意力层
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=num_hiddens//num_heads, dropout=dropout)
        self.attn_weights = None # 保存注意力权重

        # MLP 层做非线性变换
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_hiddens)
        ])

    def call(self, graph, *args, **kwargs):
        # graph : 存储图信息的字典
        # 包含 vertex_feat, edges_feat, graph_feat, edges_idx, neighbors_idx, connections_idx, valid_lens
        
        # 准备注意力层的输入
        # query 的形状应该为 (batch_size, num_queries, num_features)
        # 这里我们把 batch_size 设为 num_nodes，num_query 设为 1 表示每个节点查询自身
        query = graph["vertex_feat"][:, None, :] # (num_nodes, 1, num_features)

        # key / value 的形状应该为 (batch_size, num_keys, num_features)
        # 这里 batch_size = num_nodes，num_keys = num_neighbors 表示每个节点从邻居节点中计算注意力
        value = tf.gather(graph["vertex_feat"], graph["neighbors_idx"]) # (num_nodes, num_neighbors, num_features)
        num_neighbors = tf.shape(value)[1] # 邻居节点数

        # 由于 GAT 的注意力查询时可以用到自身节点，所以把自身节点特征也加入 key / value
        value = tf.concat([query, value], axis=1) # (num_nodes, num_neighbors + 1, num_features)

        # 利用每个节点的有效邻居数，构造注意力掩码，形状为 (batch_size, num_queries, num_key_values)
        # 在这里，mask 的形状为 (num_nodes, 1, num_neighbors + 1)
        valid_lens = graph["valid_lens"] + 1 # 加 1 表示注意力计算时包含自身节点
        mask = tf.sequence_mask(valid_lens, maxlen=num_neighbors + 1, dtype=tf.float32)
        mask = mask[:,None,:] # 添加维度变成 (num_nodes, 1, num_neighbors + 1)

        # 计算注意力
        vertex_feat, attn_weights = self.multi_head_attn(
            query, value, value, attention_mask=mask, return_attention_scores=True, **kwargs)
        vertex_feat = tf.squeeze(vertex_feat) # (num_nodes, 1, num_features) -> (num_nodes, num_features)
        self.attn_weights = tf.squeeze(attn_weights) # (num_nodes, num_heads, num_neighbors + 1)

        # 做一层 MLP
        vertex_feat = self.mlp(vertex_feat, **kwargs)
        graph["vertex_feat"] = vertex_feat # 更新节点特征

        return graph

class GCNModel(tf.keras.Model):
    def __init__(self, num_hiddens : int, num_classes : int, num_layers : int, dropout : float=0.25, *args, **kwargs):
        super(GCNModel, self).__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        # 创建节点特征的嵌入层
        self.node_embed = tf.keras.layers.Dense(num_hiddens, use_bias=False)

        # 创建多层 GCN 层
        self.gcn_layers = [GCNLayer(num_hiddens) for _ in range(num_layers)]
        
        # 创建分类器
        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation="relu"),
            tf.keras.layers.Dropout(dropout),
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

def build_leffingwell_molecules_graph(path, behavier_name : str="pungent"):
    molecules = pd.read_csv(os.path.join(path,"molecules.csv"), index_col=0)
    behavier = pd.read_csv(os.path.join(path,"behavier.csv"), index_col=0)

    smiles = molecules['IsomericSMILES']

    # 原子映射字典 和 化学键映射字典
    AtomMap = {"C": 0, "N": 1, "O": 2, "S": 3}
    BondMap = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

    max_num_nodes = 0 # 所有分子，最大的节点数
    max_num_edges = 0 # 所有分子，最大的边数
    num_neighbors = 4 # 化学分子，每个节点最多有 4 个邻居

    # 存储每个分子的图信息
    graphs_info = {}

    # 依次获得分子的 SMILES 字符串
    for cid, smile in smiles.items():
        # 将 SMILES 字符串转换为分子对象
        molecule = Chem.MolFromSmiles(smile)
        graph = Chem.RWMol(molecule) # 拷贝分子对象，用于创建分子图

        vertex = [] # 节点列表
        vertex_feat = [] # 节点特征列表
        edges = [] # 边列表

        # 获取节点信息
        for atom in graph.GetAtoms():
            vertex.append(atom.GetIdx()) # 节点索引
            # 原子名称作为节点特征，但映射为原子索引，便于后续 One-Hot 编码
            vertex_feat.append(AtomMap[atom.GetSymbol()]) 

        # 获取边信息
        for bond in graph.GetBonds():
            # 边连接关系，原子之间连接的化学键类型作为边特征
            # 化学键类型映射为化学键索引，便于后续 One-Hot 编码
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), BondMap[bond.GetBondType().name]))
        
        # 更新最大节点数和最大边数
        max_num_nodes = max(max_num_nodes, len(vertex))
        max_num_edges = max(max_num_edges, len(edges))
        
        # 添加到分子图中
        graphs_info[cid] = (graph, vertex, vertex_feat, edges)

    # 填充节点特征和边特征，便于每个分子图的信息组装为张量
    graphs = {
        "edges_idx": [],
        "vertex_feat": [],
        "edges_feat": [],
        "neighbors_idx": [],
        "connected_edges_idx": [],
        "valid_lens": [],
        "valid_nodes": [],
        "valid_edges": [],
        "molecule_graph": {}
    }
    labels = [] # 分子标签

    max_num_edges = max_num_edges * 2 # 无向图，边数翻倍

    for cid, (graph, vertex, vertex_feat, edges) in graphs_info.items():
        # 构建每个分子的图，对于分子而言，每个原子最多有 4 个邻居
        _, edges_idx, neighbors_idx, connected_edges_idx, valid_lens \
            = build_global_graph(vertex,edges,num_neighbors=4,is_undirectional=True)
        # 从 edges_idx 中解绑边连接关系和边特征
        edges_feat = [edge[-1] for edge in edges_idx]
        edges_idx = [(edge[0], edge[1]) for edge in edges_idx]

        # 记录有效节点数和有效边数
        graphs["valid_nodes"].append(len(vertex))
        graphs["valid_edges"].append(len(edges))

        # 填充节点特征和边特征
        vertex_feat += [0 for _ in range(max_num_nodes - len(vertex_feat))]
        edges_feat += [0 for _ in range(max_num_edges - len(edges_feat))]
        edges_idx += [(-1, -1) for _ in range(max_num_edges - len(edges_idx))]
        neighbors_idx += [[0] * num_neighbors for _ in range(max_num_nodes - len(neighbors_idx))]
        connected_edges_idx += [[0] * num_neighbors for _ in range(max_num_nodes - len(connected_edges_idx))]
        valid_lens += [0] * (max_num_nodes - len(valid_lens))

        # 对节点特征和边特征进行 One-Hot 编码
        vertex_feat = tf.one_hot(vertex_feat, depth=len(AtomMap))
        edges_feat = tf.one_hot(edges_feat, depth=len(BondMap))

        # 记录每个分子图的信息
        graphs["edges_idx"].append(edges_idx)
        graphs["vertex_feat"].append(vertex_feat)
        graphs["edges_feat"].append(edges_feat)
        graphs["neighbors_idx"].append(neighbors_idx)
        graphs["connected_edges_idx"].append(connected_edges_idx)
        graphs["valid_lens"].append(valid_lens)
        graphs["molecule_graph"][cid] = graph

        # 记录分子标签
        labels.append(behavier.loc[cid,behavier_name])

    # 将每个分子图的信息组装为张量
    graphs["edges_idx"] = tf.stack(graphs["edges_idx"])
    graphs["vertex_feat"] = tf.stack(graphs["vertex_feat"])
    graphs["edges_feat"] = tf.stack(graphs["edges_feat"])
    graphs["neighbors_idx"] = tf.stack(graphs["neighbors_idx"])
    graphs["connected_edges_idx"] = tf.stack(graphs["connected_edges_idx"])
    graphs["valid_lens"] = tf.stack(graphs["valid_lens"])
    graphs["valid_nodes"] = tf.constant(graphs["valid_nodes"])
    graphs["valid_edges"] = tf.constant(graphs["valid_edges"])

    return graphs, labels