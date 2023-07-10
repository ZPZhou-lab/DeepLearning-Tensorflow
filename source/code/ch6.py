import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from . import utils

import time
from tqdm import tqdm

FRUIT_MINI_CLASSES = {
    0: 'orange',
    1: 'apple',
    2: 'banana',
}

VOC_CLASSES = {
    "background": [0, 0, 0],   "aeroplane": [128, 0, 0],    "bicycle": [0, 128, 0],
    "bird": [128, 128, 0],     "boat": [0, 0, 128],         "bottle": [128, 0, 128],
    "bus": [0, 128, 128],      "car": [128, 128, 128],      "cat": [64, 0, 0],
    "chair": [192, 0, 0],      "cow": [64, 128, 0],         "diningtable": [192, 128, 0],
    "dog": [64, 0, 128],       "horse": [192, 0, 128],      "motorbike": [64, 128, 128],
    "person": [192, 128, 128], "potted piant": [0, 64, 0],  "sheep": [128, 64, 0],
    "sofa": [0, 192, 0],       "train": [128, 192, 0],      "tv/monitor": [0, 64, 128]
}

# 边界框的对角表示转换为中心表示
def box_corner_to_center(boxes : tf.Tensor):
    # boxes 形状: （N, 4)，N是边界框的数量
    # 取出边界框的对角坐标，分别左上和右下
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    # 计算边界框的中心坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 计算边界框的宽度和高度
    w = x2 - x1
    h = y2 - y1
    # 重新组合边界框的坐标
    boxes = tf.stack([cx, cy, w, h], axis=-1)
    return boxes

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
def box_corner_to_center_tf(boxes):
    # boxes 形状: （N, 4)，N是边界框的数量
    # 取出边界框的对角坐标，分别左上和右下
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    # 计算边界框的中心坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 计算边界框的宽度和高度
    w = x2 - x1
    h = y2 - y1
    # 重新组合边界框的坐标
    boxes = tf.stack([cx, cy, w, h], axis=-1)
    return boxes

# 边界框的中心表示转换为对角表示
def box_center_to_corner(boxes : tf.Tensor):
    # boxes 形状: （N, 4)，N是边界框的数量
    # 取出边界框的中心坐标和宽度高度
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    # 计算边界框的左上和右下坐标
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # 重新组合边界框的坐标
    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    return boxes

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
def box_center_to_corner_tf(boxes):
    # boxes 形状: （N, 4)，N是边界框的数量
    # 取出边界框的中心坐标和宽度高度
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    # 计算边界框的左上和右下坐标
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # 重新组合边界框的坐标
    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    return boxes

def bbox_to_rect(box, color):
    # box 是对角表示的边界框
    return Rectangle(xy=(box[0], box[1]), width=box[2]-box[0], height=box[3]-box[1],
                     fill=False, edgecolor=color, linewidth=2)

def generate_multi_bbox(img, sizes, ratios):
    """
    ### 生成以每个像素为中心的锚框

    Parameters
    ----------
    img : tf.Tensor
        输入图像，形状为 (N, h, w, C)\n
        N 是图像的数量，C 是通道数，h 和 w 是图像的高和宽
    sizes : list
        锚框缩放比
    ratios : list
        锚框宽高比
    """
    height, width = img.shape[1:3] # 图像的高和宽
    num_sizes, num_ratios = len(sizes), len(ratios) # 锚框缩放比和宽高比的数量
    boxes_per_pixel = num_sizes + num_ratios - 1 # 每个像素的锚框数量
    sizes = tf.constant(sizes, dtype=tf.float32)
    ratios = tf.constant(ratios, dtype=tf.float32)

    # 为了将锚框的中心放在像素的中心，需要设置偏移量 0.5
    # 因为每个像素都是 1x1 的小正方形，偏移 0.5 是小正方形中心
    offset_h, offset_w = 0.5, 0.5
    steps_h, steps_w = 1.0 / height, 1.0 / width # 高度和宽度移动的步长

    # 生成锚框的中心坐标
    # 高度中心，(0.5/height, 1.5/height, ..., (height-0.5)/height)
    center_h = (tf.range(height, dtype=tf.float32) + offset_h) * steps_h
    # 宽度中心，(0.5/width, 1.5/width, ..., (width-0.5)/width)
    center_w = (tf.range(width, dtype=tf.float32) + offset_w) * steps_w
    # 生成网格，shift_y 和 shift_x 形状都是 (height, width)
    # indexij='ij' 表示生成的网格以 center_h 为行，center_w 为列
    shift_y, shift_x = tf.meshgrid(center_h, center_w, indexing='ij') 
    shift_y, shift_x = tf.reshape(shift_y, [-1]), tf.reshape(shift_x, [-1]) # 拉直为向量

    # 生成锚框的高度和宽度
    # 之后就可以和中心坐标一起组合成锚框的对角坐标
    # h = s/sqrt(r), w = s*sqrt(r)
    w = tf.concat((sizes * tf.sqrt(ratios[0]), sizes[0] * tf.sqrt(ratios[1:])), axis=0) * (height / width)
    h = tf.concat((sizes / tf.sqrt(ratios[0]), sizes[0] / tf.sqrt(ratios[1:])), axis=0)
    
    # 第一步 tf.stack 生成 (n + m - 1, 4) = (boxes_per_pixel, 4) 的张量
    # 第二步 tf.tile 先将 anchor_wh 重复为 (boxes_per_pixel*height*width, 4)
    # 最后除以 2，得到半高半宽
    anchor_wh = tf.stack((-w, -h, w, h), axis=1) # 形状：(boxes_per_pixel, 4)
    anchor_wh = tf.tile(anchor_wh, (height*width, 1)) / 2 # 形状：(boxes_per_pixel*height*width, 4)

    # 生成锚框中心
    xy = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1) # 形状：(height*width, 4)
    # 同样将中心坐标复制 boxes_per_pixel 次，准备与 anchor_wh 运算
    xy = tf.repeat(xy, boxes_per_pixel, axis=0) # 形状：(boxes_per_pixel*height*width, 4)
    
    # 将中心坐标与半高半宽相加，得到锚框的对角坐标
    anchor_box = xy + anchor_wh # 形状：(boxes_per_pixel*height*width, 4)
    return anchor_box

@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def generate_multi_bbox_tf(img, sizes, ratios):
    """
    ### 生成以每个像素为中心的锚框

    Parameters
    ----------
    img : tf.Tensor
        输入图像，形状为 (N, h, w, C)\n
        N 是图像的数量，C 是通道数，h 和 w 是图像的高和宽
    sizes : list
        锚框缩放比
    ratios : list
        锚框宽高比
    """
    height, width = tf.shape(img)[1], tf.shape(img)[2] # 图像的高和宽
    num_sizes, num_ratios = tf.shape(sizes)[0], tf.shape(ratios)[0] # 锚框缩放比和宽高比的数量
    boxes_per_pixel = num_sizes + num_ratios - 1 # 每个像素的锚框数量

    # 为了将锚框的中心放在像素的中心，需要设置偏移量 0.5
    # 因为每个像素都是 1x1 的小正方形，偏移 0.5 是小正方形中心
    offset_h, offset_w = 0.5, 0.5
    steps_h, steps_w = 1.0 / tf.cast(height,dtype=tf.float32), 1.0 / tf.cast(width,dtype=tf.float32) # 高度和宽度移动的步长

    # 生成锚框的中心坐标
    # 高度中心，(0.5/height, 1.5/height, ..., (height-0.5)/height)
    center_h = (tf.range(height, dtype=tf.float32) + offset_h) * steps_h
    # 宽度中心，(0.5/width, 1.5/width, ..., (width-0.5)/width)
    center_w = (tf.range(width, dtype=tf.float32) + offset_w) * steps_w
    # 生成网格，shift_y 和 shift_x 形状都是 (height, width)
    # indexij='ij' 表示生成的网格以 center_h 为行，center_w 为列
    shift_y, shift_x = tf.meshgrid(center_h, center_w, indexing='ij') 
    shift_y, shift_x = tf.reshape(shift_y, [height*width, ]), tf.reshape(shift_x, [height*width, ]) # 拉直为向量

    # 生成锚框的高度和宽度
    # 之后就可以和中心坐标一起组合成锚框的对角坐标
    # h = s/sqrt(r), w = s*sqrt(r)
    w = tf.concat((sizes * tf.sqrt(ratios[0]), sizes[0] * tf.sqrt(ratios[1:])), axis=0) * tf.cast(height / width,dtype=tf.float32)
    h = tf.concat((sizes / tf.sqrt(ratios[0]), sizes[0] / tf.sqrt(ratios[1:])), axis=0)
    
    # 第一步 tf.stack 生成 (n + m - 1, 4) = (boxes_per_pixel, 4) 的张量
    # 第二步 tf.tile 先将 anchor_wh 重复为 (boxes_per_pixel*height*width, 4)
    # 最后除以 2，得到半高半宽
    anchor_wh = tf.stack((-w, -h, w, h), axis=1) # 形状：(boxes_per_pixel, 4)
    anchor_wh = tf.tile(anchor_wh, (height*width, 1)) / 2 # 形状：(boxes_per_pixel*height*width, 4)

    # 生成锚框
    xy = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1) # 形状：(height*width, 4)
    # 同样将中心坐标复制 boxes_per_pixel 次，准备与 anchor_wh 运算
    xy = tf.repeat(xy, boxes_per_pixel, axis=0) # 形状：(boxes_per_pixel*height*width, 4)
    
    # 将中心坐标与半高半宽相加，得到锚框的对角坐标
    anchor_box = xy + anchor_wh # 形状：(boxes_per_pixel*height*width, 4)
    return anchor_box

# 绘制锚框和边界框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    # 将变量转换成列表
    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, default_values=['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect) # 添加边界框
        if labels is not None and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w' # 如果用白色作为背景，文字用黑色
            # 在边界框的左上角创建一个文本框来标注标签
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', 
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))

def box_IoU(boxes1, boxes2):
    # 计算两个锚框或边界框列表中成对的 IoU
    # boxes1, boxes2 形状：(boxes1 数量, 4) 和 (boxes2 数量, 4)

    # 计算 box 面积：(x2 - x1) * (y2 - y1)
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # 计算两个 box 的面积
    area1 = box_area(boxes1) # 形状：(boxes1 数量, )
    area2 = box_area(boxes2) # 形状：(boxes2 数量, )

    # 下面使用了广播机制，通过[:, None] 在中间添加了一个维度
    # 交集左上角坐标，形状：(boxes1 数量, boxes2 数量, 2)
    inter_upperleft = tf.maximum(boxes1[:, None, :2], boxes2[:, :2]) 
    # 交集右下角坐标，形状：(boxes1 数量, boxes2 数量, 2)
    inter_lowerright = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    # 先计算交集的高和宽，如果高或宽出现负数，说明相交的面积为 0
    inters = tf.maximum(inter_lowerright - inter_upperleft, 0.0)
    # 交集面积，形状：(boxes1 数量, boxes2 数量)
    inter_area = inters[:, :, 0] * inters[:, :, 1]
    # 并集面积，形状：(boxes1 数量, boxes2 数量)
    # 这里用了 A\cup B = A + B - A\cap B
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area # IoU，形状：(boxes1 数量, boxes2 数量)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 4], dtype=tf.float32)))
def box_IoU_tf(boxes1, boxes2):
    # 计算两个锚框或边界框列表中成对的 IoU
    # boxes1, boxes2 形状：(boxes1 数量, 4) 和 (boxes2 数量, 4)

    # 计算 box 面积：(x2 - x1) * (y2 - y1)
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 4], dtype=tf.float32),))
    def box_area(boxes):
        return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    
    # 计算两个 box 的面积
    area1 = box_area(boxes1) # 形状：(boxes1 数量, )
    area2 = box_area(boxes2) # 形状：(boxes2 数量, )

    # 下面使用了广播机制，通过[:, None] 在中间添加了一个维度
    # 交集左上角坐标，形状：(boxes1 数量, boxes2 数量, 2)
    inter_upperleft = tf.maximum(boxes1[:, None, :2], boxes2[:, :2]) 
    # 交集右下角坐标，形状：(boxes1 数量, boxes2 数量, 2)
    inter_lowerright = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    # 先计算交集的高和宽，如果高或宽出现负数，说明相交的面积为 0
    inters = tf.maximum(inter_lowerright - inter_upperleft, 0.0)
    # 交集面积，形状：(boxes1 数量, boxes2 数量)
    inter_area = inters[:, :, 0] * inters[:, :, 1]
    # 并集面积，形状：(boxes1 数量, boxes2 数量)
    # 这里用了 A\cup B = A + B - A\cap B
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area # IoU，形状：(boxes1 数量, boxes2 数量)

def assign_anchor_to_bbox(ground_truth, anchors, valid_len, IoU_threshold : float=0.5):
    """
    ### 将真实边界框分配给锚框

    Parameters
    ----------
    ground_truth : tf.Tensor
        真实边界框，形状为 (num_gt_boxes, 4)
    anchors : tf.Tensor
        锚框，形状为 (num_anchors, 4)
    valid_len : int
        有效真实边界框的数量
    IoU_threshold : float, default = 0.5
        IoU 阈值，只有 IoU 大于 该阈值，锚框才会被分配
    """

    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    # 计算 IoU
    IoU = tf.Variable(box_IoU(anchors, ground_truth)) # 形状：(num_anchors, num_gt_boxes)
    # 通过有效长度填充 -1，这些真是边界框所在得列不会被分配
    IoU[:, valid_len:].assign(-1.0) # 形状：(num_anchors, num_gt_boxes)

    # 标记每个锚框，分配到真实边界框的张量，默认值为 -1，即不分配
    anchors_bbox_map = tf.fill(num_anchors, -1) # 形状：(num_anchors, )

    # 根据阈值，挑选出参与分配的锚框
    max_IoU, index = tf.reduce_max(IoU, axis=1), tf.argmax(IoU, axis=1, output_type=tf.int32) # 找到每一行的最大 IoU 及其索引
    anc_i = tf.reshape(tf.where(max_IoU >= IoU_threshold), (-1, 1)) # 筛选超过阈值的锚框索引
    box_j = index[max_IoU >= IoU_threshold] # 去除超过阈值的锚框对应的边界框索引
    # 将分配的结果写入 map
    # 在后续流程中，只有这些锚框才会被分配
    # 这实际上完成了算法的第 4 步
    anchors_bbox_map = tf.tensor_scatter_nd_update(anchors_bbox_map, anc_i, box_j)
    anchors_bbox_map = tf.Variable(anchors_bbox_map) # 转换为 Variable，方便更新元素

    # 用于填充分配后的行列，完成丢弃
    col_discard = tf.fill(num_anchors, -1.0)
    row_discard = tf.fill(num_gt_boxes, -1.0)
    
    for _ in range(num_gt_boxes):
        max_idx = tf.argmax(tf.reshape(IoU, (-1,)),output_type=tf.int32) # 选出最大 IoU 的索引
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map[anc_idx].assign(box_idx)

        # 将刚刚分配的锚框和边界框的 IoU 设置为 -1 完成丢弃
        # 这样在下次迭代中，就不会再次分配
        IoU[:, box_idx].assign(col_discard)
        IoU[anc_idx, :].assign(row_discard)
    
    return tf.constant(anchors_bbox_map)

@tf.function(input_signature=(tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int32),
                              tf.TensorSpec(shape=(), dtype=tf.float32)))
def assign_anchor_to_bbox_tf(ground_truth, anchors, valid_len, IoU_threshold : float=0.5):
    """
    ### 将真实边界框分配给锚框

    Parameters
    ----------
    ground_truth : tf.Tensor
        真实边界框，形状为 (num_gt_boxes, 4)
    anchors : tf.Tensor
        锚框，形状为 (num_anchors, 4)
    valid_len : int
        有效长度，即真实边界框的数量，去除掉填充边界框，即类别为 -1 的边界框
    IoU_threshold : float, default = 0.5
        IoU 阈值，只有 IoU 大于 该阈值，锚框才会被分配
    """
    num_anchors, num_gt_boxes = tf.shape(anchors)[0], tf.shape(ground_truth)[0]

    # 计算 IoU
    IoU = box_IoU_tf(anchors, ground_truth) # 形状：(num_anchors, num_gt_boxes)
    # 通过有效长度填充 -1，这些真是边界框所在得列不会被分配
    IoU = tf.concat([IoU[:, :valid_len], 
                     tf.zeros((num_anchors, num_gt_boxes - valid_len),dtype=tf.float32) - 1], axis=1)

    # 标记每个锚框，分配到真实边界框的张量，默认值为 -1，即不分配
    anchors_bbox_map = tf.zeros((num_anchors,),dtype=tf.int32) - 1 # 形状：(num_anchors, )

    # 根据阈值，挑选出参与分配的锚框
    max_IoU, index = tf.reduce_max(IoU, axis=1), tf.argmax(IoU, axis=1, output_type=tf.int32) # 找到每一行的最大 IoU 及其索引
    anc_i = tf.reshape(tf.where(max_IoU >= IoU_threshold), (num_anchors, 1)) # 筛选超过阈值的锚框索引
    box_j = index[max_IoU >= IoU_threshold] # 去除超过阈值的锚框对应的边界框索引
    # 将分配的结果写入 map
    # 在后续流程中，只有这些锚框才会被分配
    # 这实际上完成了算法的第 4 步
    anchors_bbox_map = tf.tensor_scatter_nd_update(anchors_bbox_map, anc_i, box_j)

    # 用于填充分配后的行列，完成丢弃
    col_discard = tf.zeros((num_anchors,1),dtype=tf.float32) - 1 # tf.fill(num_anchors, -1.0)
    row_discard = tf.zeros((1,num_gt_boxes),dtype=tf.float32) - 1# tf.fill(num_gt_boxes, -1.0)
    
    for _ in range(num_gt_boxes):
        max_idx = tf.argmax(tf.reshape(IoU, (num_anchors*num_gt_boxes,)),output_type=tf.int32) # 选出最大 IoU 的索引
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        # anchors_bbox_map = tf.tensor_scatter_nd_update(anchors_bbox_map, [[anc_idx]], [box_idx])
        anchors_bbox_map = tf.concat([anchors_bbox_map[:anc_idx], [box_idx], anchors_bbox_map[anc_idx+1:]], axis=0)

        # # 将刚刚分配的锚框和边界框的 IoU 设置为 -1 完成丢弃
        # # 这样在下次迭代中，就不会再次分配
        IoU = tf.concat([IoU[:anc_idx, :], row_discard, IoU[anc_idx+1:, :]], axis=0)
        IoU = tf.concat([IoU[:, :box_idx], col_discard, IoU[:, box_idx+1:]], axis=1)
    
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bbox, eps : float=1e-6):
    """
    ### 计算偏移量

    Parameters
    ----------
    anchors : tf.Tensor
        锚框
    assigned_bbox : tf.Tensor
        分配的真实边界框
    eps : float, default = 1e-6
        防止数值溢出微小常数
    """
    # 锚框的中心坐标 和 真实边界框的中心坐标
    anchors_center = box_corner_to_center(anchors) # (X_A, Y_A, W_A, H_A)
    assigned_bbox_center = box_corner_to_center(assigned_bbox) # (X_B, Y_B, W_B, H_B)

    # 偏移量计算公式
    offset_xy = 10 * (assigned_bbox_center[:, :2] - anchors_center[:, :2]) / anchors_center[:, 2:]
    offset_wh = 5 * tf.math.log(eps + assigned_bbox_center[:, 2:] / anchors_center[:, 2:])
    offset = tf.concat([offset_xy, offset_wh], axis=1)

    return offset


@tf.function(input_signature=(tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.float32)))
def offset_boxes_tf(anchors, assigned_bbox, eps : float=1e-6):
    """
    ### 计算偏移量

    Parameters
    ----------
    anchors : tf.Tensor
        锚框
    assigned_bbox : tf.Tensor
        分配的真实边界框
    eps : float, default = 1e-6
        防止数值溢出微小常数
    """
    # 锚框的中心坐标 和 真实边界框的中心坐标
    anchors_center = box_corner_to_center_tf(anchors) # (X_A, Y_A, W_A, H_A)
    assigned_bbox_center = box_corner_to_center_tf(assigned_bbox) # (X_B, Y_B, W_B, H_B)

    # 偏移量计算公式
    offset_xy = 10 * (assigned_bbox_center[:, :2] - anchors_center[:, :2]) / anchors_center[:, 2:]
    offset_wh = 5 * tf.math.log(eps + assigned_bbox_center[:, 2:] / anchors_center[:, 2:])
    offset = tf.concat([offset_xy, offset_wh], axis=1)

    return offset

def multibox_target(anchors, labels):
    """
    ### 给锚框标记类别和偏移量

    Parameters
    ----------
    anchors : tf.Tensor
        锚框，形状：(num_anchor, 4)
    labels
        真实边界框的信息，形状：(batch_size, num_gt_boxes, 5)
        5 个元素分别为：类别、左上角坐标、右下角坐标
    
    Returns
    -------
    batch_offset : tf.Tensor
        偏移量标签，形状：(batch_size, num_anchor*4)
    batch_mask : tf.Tensor
        锚框掩码，形状：(batch_size, num_anchor*4)
    batch_class_labels : tf.Tensor
        类别标签，形状：(batch_size, num_anchor)
    """

    batch_size = labels.shape[0]
    # 初始化 batch 的偏移量、掩码、类别标签
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchor = anchors.shape[0]

    for i in range(batch_size):
        # 依次取出每个 batch 的真实边界框，形状 (num_gt_boxes, 5)
        label = labels[i, :, :] 

        # 完成锚框和真实边界框的分配，形状：(num_anchor, )
        # 计算有效长度 valid_len，即真实边界框的数量，去除掉填充边界框，即类别为 -1 的边界框
        valid_len = tf.shape(tf.where(label[:, 0] >= 0))[0]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, valid_len=valid_len)

        # 生成掩码，形状：(num_anchor, 4)
        # 成功分配的锚框对应的掩码值为 1，其余为 0
        # anchors_bbox_map >= 0 是分配了真实边界框的锚框
        bbox_mask = tf.repeat(tf.expand_dims(
            tf.cast(anchors_bbox_map >= 0, dtype=tf.float32), axis=-1), 4, axis=-1)
        
        # 将类别标签和分配的边界框坐标初始化为 0
        class_labels = tf.zeros((num_anchor, ), dtype=np.int32)
        assigned_bbox = tf.zeros((num_anchor, 4), dtype=np.float32)
        
        # 使用真实边界框标记锚框类别
        indices_true = tf.reshape(tf.where(anchors_bbox_map >= 0), (-1, 1)) # 取出被分配的索引
        bbox_index = tf.reshape(tf.gather(anchors_bbox_map, indices_true), (-1)) # 分配得到的真实边界框索引
        # 将 indices_true 对应的 class_labels 和 assigned_bbox 更新为真实边界框的信息
        class_labels = tf.tensor_scatter_nd_update(
            class_labels, indices_true, tf.cast(tf.gather(label[:, 0], bbox_index),dtype=tf.int32) + 1)
        assigned_bbox = tf.tensor_scatter_nd_update(
            assigned_bbox, indices_true, tf.gather(label[:, 1:], bbox_index))

        # 偏移量计算，乘以 bbox_mask，将未分配的偏移量设置为 0
        offset = offset_boxes(anchors, assigned_bbox) * bbox_mask # 形状：(num_anchor, 4)
        # 添加到 batch
        batch_offset.append(tf.reshape(offset, (-1))) # 将 offset 拉直为向量，维度 (num_anchor * 4, )
        batch_mask.append(tf.reshape(bbox_mask, (-1))) # 将 mask 拉直为向量，维度 (num_anchor * 4, ) 
        batch_class_labels.append(class_labels)
    
    # 拼接结果并返回
    bbox_offset = tf.stack(batch_offset, axis=0)
    bbox_mask = tf.stack(batch_mask, axis=0)
    class_labels = tf.stack(batch_class_labels, axis=0)
    return (bbox_offset, bbox_mask, class_labels)


@tf.function(input_signature=(tf.TensorSpec(shape=(None, 4), dtype=tf.float32), 
                              tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32)))
def multibox_target_tf(anchors, labels):
    """
    ### 给锚框标记类别和偏移量

    Parameters
    ----------
    anchors : tf.Tensor
        锚框，形状：(num_anchor, 4)
    labels
        真实边界框的信息，形状：(batch_size, num_gt_boxes, 5)
        5 个元素分别为：类别、左上角坐标、右下角坐标
    
    Returns
    -------
    batch_offset : tf.Tensor
        偏移量标签，形状：(batch_size, num_anchor*4)
    batch_mask : tf.Tensor
        锚框掩码，形状：(batch_size, num_anchor*4)
    batch_class_labels : tf.Tensor
        类别标签，形状：(batch_size, num_anchor)
    """
    def task_single_batch(anchors, label):
        num_anchor = tf.shape(anchors)[0]

        # 完成锚框和真实边界框的分配，形状：(num_anchor, )
        # 计算有效长度 valid_len，即真实边界框的数量，去除掉填充边界框，即类别为 -1 的边界框
        valid_len = tf.shape(tf.where(label[:, 0] >= 0))[0]
        anchors_bbox_map = assign_anchor_to_bbox_tf(label[:, 1:], anchors, valid_len)

        # 生成掩码，形状：(num_anchor, 4)
        # 成功分配的锚框对应的掩码值为 1，其余为 0
        # anchors_bbox_map >= 0 是分配了真实边界框的锚框
        bbox_mask = tf.repeat(tf.expand_dims(
            tf.cast(anchors_bbox_map >= 0, dtype=tf.float32), axis=-1), 4, axis=-1)
        
        # 将类别标签和分配的边界框坐标初始化为 0
        class_labels = tf.zeros((num_anchor, ), dtype=np.int32)
        assigned_bbox = tf.zeros((num_anchor, 4), dtype=np.float32)
        
        # 使用真实边界框标记锚框类别
        indices_true = tf.where(anchors_bbox_map >= 0) # 取出被分配的索引
        num_true = tf.shape(indices_true)[0] # 真实边界框的数量
        bbox_index = tf.reshape(tf.gather(anchors_bbox_map, indices_true), (num_true, )) # 分配得到的真实边界框索引
        # 将 indices_true 对应的 class_labels 和 assigned_bbox 更新为真实边界框的信息
        class_labels = tf.tensor_scatter_nd_update(
            class_labels, indices_true, tf.cast(tf.gather(label[:, 0], bbox_index),dtype=tf.int32) + 1)
        assigned_bbox = tf.tensor_scatter_nd_update(
            assigned_bbox, indices_true, tf.gather(label[:, 1:], bbox_index))

        # 偏移量计算，乘以 bbox_mask，将未分配的偏移量设置为 0
        offset = offset_boxes_tf(anchors, assigned_bbox) * bbox_mask # 形状：(num_anchor, 4)
        offset = tf.reshape(offset, (tf.shape(offset)[0]*4,))
        bbox_mask = tf.reshape(bbox_mask, (tf.shape(bbox_mask)[0]*4,))

        return offset, bbox_mask, class_labels

    # 调用 TensorFlow 的 map_fn 函数，对每个样本调用 task_single_batch 函数
    # 使用 tf.stop_gradient 阻止梯度的传播
    batch_offset, batch_mask, batch_class_labels = tf.nest.map_structure(
        tf.stop_gradient, 
        tf.map_fn(lambda x: task_single_batch(anchors, x), labels, 
                  fn_output_signature=(tf.float32, tf.float32, tf.int32),
                  parallel_iterations=labels.shape[0]))
    
    return batch_offset, batch_mask, batch_class_labels

def offset_inverse(anchors, offset_preds):
    # 将锚框转换为中心表示
    anchors = box_corner_to_center(anchors)
    # 根据偏移量计算预测边界框的中心和宽高
    pred_bbox_xy = (offset_preds[:, :2] * anchors[:, 2:] / 10) + anchors[:, :2]
    pred_bbox_wh = tf.math.exp(offset_preds[:, 2:] / 5) * anchors[:, 2:]
    pred_bbox = tf.concat([pred_bbox_xy, pred_bbox_wh], axis=1)
    # 将预测边界框转换为对角表示
    pred_bbox = box_center_to_corner(pred_bbox)
    return pred_bbox

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 4], dtype=tf.float32)))
def offset_inverse_tf(anchors, offset_preds):
    # 将锚框转换为中心表示
    anchors = box_corner_to_center_tf(anchors)
    # 根据偏移量计算预测边界框的中心和宽高
    pred_bbox_xy = (offset_preds[:, :2] * anchors[:, 2:] / 10) + anchors[:, :2]
    pred_bbox_wh = tf.math.exp(offset_preds[:, 2:] / 5) * anchors[:, 2:]
    pred_bbox = tf.concat([pred_bbox_xy, pred_bbox_wh], axis=1)
    # 将预测边界框转换为对角表示
    pred_bbox = box_center_to_corner_tf(pred_bbox)
    return pred_bbox

def non_maximum_suppression(boxes, probs, iou_threshold, max_num_bbox : int=None):
    """
    ### 非极大值抑制

    Parameters
    ----------
    boxes : tf.Tensor
        预测边界框，形状：(num_pred, 4)
    probs : tf.Tensor
        预测边界框的置信度，形状：(num_pred, )
    iou_threshold : float
        IoU 阈值
    max_num_bbox : int, default = `None`
        最大保留的预测边界框数量，如果为 `None`，则最多保留 `probs` 中的所有元素

    Returns
    -------
    keep : tf.Tensor
        保留的预测边界框索引
    """
    if max_num_bbox is None:
        max_num_bbox =  tf.shape(probs)[0]

    # 根据预测边界框的置信度排序
    order = tf.argsort(probs, direction='DESCENDING') # 降序排列
    # 初始化保留预测边界框的索引
    keep = []

    # 计算 boxes 两两之间的 IoU
    pairwise_IoU = box_IoU_tf(boxes, boxes) # 形状：(num_pred, num_pred)
    pairwise_mask = pairwise_IoU <= iou_threshold
    valid_mask = tf.ones_like(probs, dtype=tf.bool) # 形状：(num_pred, )
    valid_count = tf.shape(probs)[0]

    while valid_count > 0 and len(keep) < max_num_bbox:
        # 获取第一个有效的索引
        i = order[tf.argmax(valid_mask)]
        keep.append(i)

        # 更新 valid_mask
        valid_mask = tf.logical_and(valid_mask, tf.gather(pairwise_mask[i],order))
        # 更新 valid_count
        valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.int32))

    return tf.stack(keep)

def multibox_detection(cls_probs, offset_preds, anchors, 
                       nms_threshold : float=0.5, pos_threshold : float=0.001):
    """
    ### 多尺度目标检测

    Parameters
    ----------
    cls_probs : tf.Tensor
        预测类别概率分布，形状：(batch_size, num_anchor, num_class + 1)
    offset_preds : tf.Tensor
        预测边界框偏移量，形状：(batch_size, 4*num_anchor)
    anchors : tf.Tensor
        锚框，形状：(num_anchor, 4)
    nms_threshold : float, default = 0.5
        非极大值抑制阈值
    pos_threshold : float, default = 0.001
        正类阈值
    
    Returns
    -------
    output : tf.Tensor
        检测结果，形状：(batch_size, num_detection, 6)
    """
    batch_size, num_anchor = cls_probs.shape[0], anchors.shape[0]
    # 初始化输出
    output = []
    for i in range(batch_size):
        # 取出第 i 个样本的预测类别概率和边界框偏移量
        cls_prob  = cls_probs[i] # 形状：(num_anchor, num_class + 1)
        offset_pred = tf.reshape(offset_preds[i], (-1, 4)) # 形状：(num_anchor, 4)
        # 计算非背景类别的置信度 conf，取出对应的类别编号 class_id
        conf = tf.reduce_max(cls_prob[:,1:], axis=1) # 形状：(num_anchor, )
        # 这里将结果转化为 numpy 数组，因为后续要通过索引修改数组元素
        class_id = tf.cast(tf.argmax(cls_prob[:,1:], axis=1), dtype=tf.float32) # 形状：(num_anchor, )

        # 利用偏移量 offset 预测边界框
        pred_bbox = offset_inverse(anchors, offset_pred) # 形状：(num_anchor, 4)
        # NMS 计算保留的预测边界框索引
        keep = non_maximum_suppression(pred_bbox, conf, nms_threshold) # 形状：(num_detection, )

        # 找到所有的 non_keep 索引，并将类设置为背景 
        all_idx = tf.range(num_anchor, dtype=tf.int32) # 形状：(num_anchor, )
        combined = tf.concat([keep, all_idx], axis=0) # 形状：(num_detection + num_anchor, )

        # 找到 combined 中的非重复元素，即为 non_keep 索引
        unique, _, counts = tf.unique_with_counts(combined)
        non_keep = unique[counts == 1] # counts == 1 表示非重复元素
        all_idx_sorted = tf.concat([keep, non_keep], axis=0) # 形状：(num_anchor, )

        # 设置背景类别标签，-1 表示背景类别
        class_id = tf.tensor_scatter_nd_update(
            class_id, non_keep[:,None], tf.fill(non_keep.shape[0], -1.0))
        # 对预测结果进行排序，高置信度的类别排在前面，后面是背景类别
        class_id = tf.gather(class_id, all_idx_sorted)
        conf = tf.gather(conf, all_idx_sorted)
        pred_bbox = tf.gather(pred_bbox, all_idx_sorted)

        # pos_threshold 为正类阈值，如果置信度小于阈值，同样设置为背景
        below_pos_idx = (conf < pos_threshold)
        num_below_pos = tf.reduce_sum(tf.cast(below_pos_idx, tf.int32))
        class_id = tf.tensor_scatter_nd_update(
            class_id, tf.where(below_pos_idx), tf.fill(num_below_pos, -1.0))

        # 修改置信度 conf 中背景部分的概率为 1 - conf，用 tf.where 实现
        conf = tf.where(below_pos_idx, 1 - conf, conf) # 形状：(num_anchor, )

        # 拼接预测信息，形状：(num_anchor, 2 + 4 = 6)
        # 6个元素分别为：类别编号、置信度、边界框坐标
        pred_info = tf.concat([tf.expand_dims(class_id, axis=-1),
                               tf.expand_dims(conf, axis=-1),
                               pred_bbox], axis=-1)
        # 添加到输出
        output.append(pred_info)
    
    return tf.stack(output)

def generate_multi_scale_bbox(fmap_h : int, fmap_w : int, size : float, 
                              width : int, height : int, ratios : list = [1, 2, 0.5]):
    # 生成多尺度锚框
    # 构建特征图 fmap
    fmap = tf.zeros((1, fmap_h, fmap_w, 3)) # 形状：(batch_size, height, width, channels)
    # 生成锚框的相对坐标
    anchors = generate_multi_bbox_tf(fmap, sizes=[size], ratios=ratios)
    bbox_scale = tf.constant([width, height, width, height], dtype=tf.float32)
    return anchors * bbox_scale

def load_detection_data(path : str, dataset : str="train", resize : list=[256, 256]):
    import os
    path_dir = os.path.join(path, dataset) + "/"
    # 初始化图像和标签列表
    images, labels = [], []

    # 获取文件夹下的所有文件名
    filenames = os.listdir(path_dir + "images")
    for file in filenames:
        try:
            # 图像路径 和 标签路径
            image_path = path_dir + "images/" + file
            label_path = path_dir + "labels/" + file[:-4] + ".txt"
            
            # 读取图像
            img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
            img = tf.cast(img, dtype=tf.float32) / 255.0 # 转换为浮点数
            # 转换为相同尺寸
            img = tf.image.resize(img, resize)
            
            # 读取标签
            label = []
            with open(label_path, "r") as f:
                # 读取每一行
                for line in f.readlines():
                    # 获取类别和边界框
                    label.append([float(x) for x in line.split()])
            # 转换为张量
            label = tf.constant(label, dtype=tf.float32)

            # 添加到列表
            images.append(img)
            labels.append(label)
        except:
            continue
    print("read %s dataset with %d images"%(dataset, len(images)))
          
    return images, labels

class DetectionDataLoader:
    def __init__(self, path : str, dataset : str, resize : list=[256,256], batch_size : int=32):
        # 读取数据集
        self.X, self.Y = load_detection_data(path, dataset, resize)
        # 找到每个图像中的最大边界框数
        self.m = max([len(y) for y in self.Y])

        # 将边界框填充为 m 个
        for i,y in enumerate(self.Y):
            # y : tf.Tensor，形状为 (n, 5)，n 是边界框数
            n = y.shape[0]
            if n < self.m:
                # 让左上角坐标为 (0,0), 右下角坐标为 (0,0)，成为无效边界框
                pad = tf.repeat(tf.constant([[-1., 0.0, 0.0, 1.0, 1.0]]), self.m - n, axis=0)
                y = tf.concat([y, pad], axis=0)
                self.Y[i] = y
        
        # 将 X, Y 转换为张量
        self.X = tf.stack(self.X)
        self.Y = tf.stack(self.Y)

        self.bbox_scale = tf.constant([resize[0], resize[1], resize[0], resize[1]], dtype=tf.float32)
        self.batch_size = batch_size
    
    # 读取批量数据
    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))\
            .batch(self.batch_size).shuffle(len(self.X))
        return dataset
    
    # 通过索引获取数据
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ClassPredictor(tf.keras.layers.Layer):
    def __init__(self, num_anchors : int, num_classes : int, dropout : float=0.0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv = tf.keras.layers.Conv2D(
            filters=num_anchors * (num_classes + 1),kernel_size=3,padding='same')
        # 将形状从 (batch, height, width, num_anchors * (num_classes + 1))
        # 转换为 (batch, height * width * num_anchors, num_classes + 1)
        self.flat = tf.keras.layers.Flatten() # 保留批量维度，合并其余维度
        
    def call(self, inputs : tf.Tensor, flatten : bool=False, **kwargs):
        Y = self.conv(self.dropout(inputs,**kwargs))
        if flatten:
            Y = self.flat(Y)
        return Y

class BBoxPredictor(tf.keras.layers.Layer):
    def __init__(self, num_anchors : int, dropout : float=0.0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_anchors = num_anchors
        # 每个锚框预测 4 个偏移量，因此输出通道数为 `num_anchors * 4`
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv = tf.keras.layers.Conv2D(
            filters=num_anchors * 4, kernel_size=3,padding='same')
        # 将形状从 (batch, height, width, num_anchors * (num_classes + 1))
        # 转换为 (batch, height * width * num_anchors * (num_classes + 1))
        self.flat = tf.keras.layers.Flatten() # 保留批量维度，合并其余维度
        
    def call(self, inputs : tf.Tensor, flatten : bool=False, **kwargs):
        Y = self.conv(self.dropout(inputs,**kwargs))
        if flatten:
            Y = self.flat(Y)
        return Y

class ConcatPreds(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, preds : list, *args, **kwargs):
        # preds : 需要进行拼接的多尺度预测结果，用列表存储
        return tf.concat(preds, axis=1)

class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels : int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.block = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        ])
    
    def call(self, inputs : tf.Tensor, **kwargs):
        return self.block(inputs,**kwargs)

def create_base_model(input_size : tuple, channel_list : list=[16, 32, 64]):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_size),
    ])
    for channels in channel_list:
        # 直接使用 DownSampleBlock 的接口
        model.add(DownSampleBlock(out_channels=channels).block)
    return model

def block_call(X : tf.Tensor, block : tf.keras.layers.Layer, sizes : tf.Tensor, ratios : tf.Tensor,
               cls_predictor : ClassPredictor, bbox_predictor : BBoxPredictor,**kwargs):
    # 计算 block 的输出特征图
    Y = block(X,**kwargs)
    # 生成基于特征图的多尺度锚框
    anchors = generate_multi_bbox_tf(img=Y, sizes=sizes, ratios=ratios)
    # 生成类别预测结果 和 边界框预测结果
    cls_preds = cls_predictor(Y, flatten=True,**kwargs)
    bbox_preds = bbox_predictor(Y, flatten=True,**kwargs)
    return (Y, anchors, cls_preds, bbox_preds)

class TinySSD(tf.keras.Model):
    def __init__(self, input_size : tuple, num_classes : int, 
                 sizes, ratios, dropout : float=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        # 每个像素生成的锚框数量
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = [len(sizes[i]) + len(ratios[i]) - 1 for i in range(5)] 

        # 将模型块进行包装
        self.blocks = [
            create_base_model(input_size=input_size), # 基础模型
            DownSampleBlock(out_channels=128), # 第一个下采样特征块
            DownSampleBlock(out_channels=128), # 第二个下采样特征块
            DownSampleBlock(out_channels=128), # 第三个下采样特征块
            tf.keras.layers.GlobalMaxPooling2D(keepdims=True) # 全局最大池化，注意保持维度
        ]
        # 设置类别预测层 和 边界框预测层
        self.class_predictors = [
            ClassPredictor(num_anchors=self.num_anchors[i], num_classes=num_classes, dropout=dropout) for i in range(5)
        ]
        self.bbox_predictors = [
            BBoxPredictor(num_anchors=self.num_anchors[i], dropout=dropout) for i in range(5)
        ]
    
    def call(self, X, training=None, mask=None):
        # 初始化锚框，类别预测结果，边界框预测结果
        anchors, class_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        # 逐层进行预测
        for i in range(5):
            # X 形状：(batch, height, width, num_channels)
            # anchors[i] 形状：(height[i] * width[i] * num_anchors[i], 4)
            # class_preds[i] 形状：(batch, height[i] * width[i] * num_anchors[i] * (num_classes + 1))
            # bbox_preds[i] 形状：(batch, height[i] * width[i] * num_anchors[i] * 4)
            X, anchors[i], class_preds[i], bbox_preds[i] = block_call(
                X, self.blocks[i], self.sizes[i], self.ratios[i], 
                self.class_predictors[i], self.bbox_predictors[i], training=training
            )

        # 将多尺度的锚框拼接在一起
        anchors = tf.concat(anchors, axis=0) # (num_anchors, 4)
        
        # 拼接多尺度的类别，将类别维度恢复在最后一维，中间维度是多尺度锚框的个数
        class_preds = ConcatPreds()(class_preds, axis=1)
        class_preds = tf.reshape(class_preds, shape=(class_preds.shape[0], -1, self.num_classes + 1))
        class_preds = tf.nn.softmax(class_preds, axis=-1) # 使用 softmax 将类别预测结果转换为概率
        
        # 拼接多尺度预测的边界框
        bbox_preds = ConcatPreds()(bbox_preds, axis=1)

        return anchors, class_preds, bbox_preds
    
class SSDLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='none', name=None):
        super().__init__(reduction, name)
        self.class_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        self.bbox_loss = tf.keras.losses.Huber(reduction='none')
    
    def __call__(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        # cls_preds 形状：(batch_size, num_anchors, num_classes + 1)
        # cls_labels 形状：(batch_size, num_anchors)
        # bbox_preds 形状：(batch_size, num_anchors * 4)
        # bbox_labels 形状：(batch_size, num_anchors * 4)
        # bbox_masks 形状：(batch_size, num_anchors * 4)
        cls_loss = self.class_loss(y_true=cls_labels, y_pred=cls_preds)
        cls_loss = tf.reduce_mean(cls_loss, axis=1)
        
        bbox_loss = self.bbox_loss(y_true=bbox_labels * bbox_masks, y_pred=bbox_preds * bbox_masks)
        return cls_loss + bbox_loss

def cls_evaluation(cls_preds, cls_labels):
    # cls_preds 形状：(batch_size, num_anchors, num_classes + 1)
    # cls_labels 形状：(batch_size, num_anchors)
    cls_preds = tf.cast(tf.argmax(cls_preds, axis=-1), dtype=tf.int32)
    return tf.reduce_sum(tf.cast(cls_preds == cls_labels,dtype=tf.int32))

def bbox_evaluation(bbox_preds, bbox_labels, bbox_masks):
    # bbox_preds 形状：(batch_size, num_anchors * 4)
    # bbox_labels 形状：(batch_size, num_anchors * 4)
    # bbox_masks 形状：(batch_size, num_anchors * 4)
    return tf.reduce_sum(tf.abs((bbox_labels - bbox_preds) * bbox_masks))

def train_SSD(model, train_iter, Epochs : int=10, lr : float=0.001, verbose : int=1):
    loss_func = SSDLoss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    animator = utils.Animator(xlabel='epoch', xlim=[1, Epochs], ncols=3,
                              legend=(("loss",), ('accuracy',),('bbox mae',)),
                              fmts=(("-",),("m--",),("g-,",)),figsize=(12,3))
    
    # 记录单词处理速度
    speeds = []

    for epoch in range(Epochs):
        start = time.time()
        # 存储每个迭代周期的损失和样本量
        loss_batch, acc_batch, mae_batch = tf.constant(0.0), tf.constant(0), tf.constant(0.0)
        samples_batch, samples = tf.constant(0), tf.constant(0)

        for x_batch, y_batch in train_iter.create_dataset():
            with tf.GradientTape() as tape:
                # 生成多尺度锚框，并为每个锚框预测类别和偏移量
                # anchors 形状：(num_anchors, 4)
                # class_preds 形状：(batch_size, num_anchors, num_classes + 1)
                # bbox_preds 形状：(batch_size, num_anchors * 4)
                anchors, cls_preds, bbox_preds = model(x_batch,training=True)

                # 为每个锚框分配类别和偏移量
                # bbox_labels 形状：(batch_size, num_anchors * 4)
                # bbox_mask 形状：(batch_size, num_anchors * 4)
                # cls_labels 形状：(batch_size, num_anchors)
                bbox_labels, bbox_mask, cls_labels = multibox_target_tf(anchors, y_batch)

                # 计算损失
                loss = loss_func(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_mask)
                batch_size = tf.shape(loss)[0]
                loss = tf.reduce_mean(loss)
                
            weights = model.trainable_variables
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

            # 计算损失和评估指标
            samples += tf.size(cls_labels)
            samples_batch += batch_size
            loss_batch += loss
            acc_batch += cls_evaluation(cls_preds, cls_labels)
            mae_batch += bbox_evaluation(bbox_preds, bbox_labels, bbox_mask)

        end = time.time()
        speeds.append(samples_batch.numpy() / (end - start))

        # 添加动画
        if epoch == 0 or (epoch + 1) % verbose == 0:
            animator.add(epoch+1,(loss_batch.numpy() / samples_batch.numpy(),),ax=0)
            animator.add(epoch+1,(acc_batch.numpy() / samples.numpy(),),ax=1)
            animator.add(epoch+1,(mae_batch.numpy() / (samples.numpy()*4),),ax=2)
        
    print(f"平均 {np.mean(speeds):.1f} 样本/秒")
    return model

def object_detection_predict(X : tf.Tensor, model):
    # X 形状：(height, width, 3)
    # anchors 形状：(num_anchors, 4)
    # class_prods 形状：(1, num_anchors, num_classes + 1)
    # bbox_preds 形状：(1, num_anchors * 4)
    anchors, cls_probs, bbox_preds = model(tf.expand_dims(X, axis=0))

    # 使用非极大值抑制来移除相似的锚框
    # output 形状：(num_anchors, 6)
    output = multibox_detection(cls_probs, bbox_preds, anchors)[0] # 去除 batch 维度

    # 过滤掉背景类（值为 -1）的预测结果
    non_backgrounds = tf.where(output[:,0] > -1)[:,0]
    return tf.gather(output, non_backgrounds)

def display_object_predict(img, output, label_dict, figsize : tuple=(4,4),
                           threshold : int=5, width : int=256, height : int=256):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    for row in output[0:threshold]:
        label = label_dict[int(row[0])]
        prob = float(row[1])
        bbox = [row[2:6] * tf.constant([width, height, width, height], dtype=tf.float32)]
        show_bboxes(fig.axes[0], bbox, '%s=%.2f' %(label,prob))

class ROIPooling(tf.keras.layers.Layer):
    """
    ### ROIPooling
    兴趣区域池化层
    """
    def __init__(self, output_shape, **kwargs):
        super(ROIPooling, self).__init__(**kwargs)
        self.o_h, self.o_w = tf.constant(output_shape, dtype=tf.int32)
    
    def call(self, fmaps : tf.Tensor, rois : tf.Tensor, input_shape : tuple=None, **kwargs):
        # f_maps 形状：(batch_size, height, width, channels)
        # rois 形状：(batch_size, num_rois, 4)
        # input_shape = (height, width)
        def task_single_batch(fmap, roi):
            return ROIPooling.roi_pool_single_fmap(fmap, roi, self.o_h, self.o_w)

        # 如果输入的是绝对坐标，则转换为相对坐标
        if input_shape is not None:
            rois = rois / tf.constant([input_shape[0], input_shape[1], 
                                       input_shape[0], input_shape[1]], dtype=tf.float32)

        # 输出形状：(batch_size, num_rois, o_h, o_w, channels)
        return tf.map_fn(lambda x: task_single_batch(x[0], x[1]), (fmaps, rois), dtype=tf.float32)
    
    # 处理单张图片的兴趣区域池化
    @staticmethod
    @tf.function(input_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32)))
    def roi_pool_single_fmap(fmap : tf.Tensor, rois : tf.Tensor, o_h : int, o_w : int):
        # fmap 形状：(height, width, channels)
        # rois 形状：(num_rois, 4)
        def task_single_roi(roi):
            return ROIPooling.roi_pool_single_roi(fmap, roi, o_h, o_w)
        
        # 输出形状：(num_rois, o_h, o_w, channels)
        return tf.map_fn(lambda x : task_single_roi(x), rois, dtype=tf.float32)

    # 对每张图片的每个兴趣区域进行池化
    @staticmethod
    @tf.function(input_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(4,), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32)))
    def roi_pool_single_roi(fmap : tf.Tensor, roi : tf.Tensor, o_h : int, o_w : int):
        # fmap 形状：(height, width, channels)
        # roi = (x1, y1, x2, y2)，形状：(4,)，使用相对坐标表示，取值 [0, 1] 之间
        
        # 获取 fmaps 的高度和宽度
        fmap_h, fmap_w, num_channels = tf.shape(fmap)[0], tf.shape(fmap)[1], tf.shape(fmap)[2]
        # 获取兴趣区域的左上角坐标和右下角坐标
        x1 = tf.cast(roi[0] * tf.cast(fmap_w, tf.float32), dtype=tf.int32)
        y1 = tf.cast(roi[1] * tf.cast(fmap_h, tf.float32), dtype=tf.int32)
        x2 = tf.cast(roi[2] * tf.cast(fmap_w, tf.float32), dtype=tf.int32)
        y2 = tf.cast(roi[3] * tf.cast(fmap_h, tf.float32), dtype=tf.int32)

        # 生成网格，因为后面要使用切片索引 x[a:b]，而切片索引不能取到右端点 b
        # 所以这里让右端点加 1
        x_grid = tf.cast(tf.math.ceil(tf.linspace(x1, x2 + 1, o_w + 1)), dtype=tf.int32)
        y_grid = tf.cast(tf.math.ceil(tf.linspace(y1, y2 + 1, o_h + 1)), dtype=tf.int32)

        # 通过 repeat 和 reshape 来生成网格矩阵
        # [x1, x2, x3] -> [x1, x1, x2, x2, x3, x3] -> [[x1, x2], [x2, x3]]
        x_grid = tf.reshape(tf.repeat(x_grid, repeats=2)[1:-1], shape=[o_w, 2]) # 形状：(o_w, 2)
        y_grid = tf.reshape(tf.repeat(y_grid, repeats=2)[1:-1], shape=[o_h, 2]) # 形状：(o_h, 2)
        
        # 通过 ceil 取整，可能会出现相邻网格点的坐标相同的情况
        # 例如 x_gird: [1, 2, 2] -> [1, 1, 2, 2, 2, 2] -> [[1, 2], [2, 2]]
        # 此时将前一个网格点的坐标减一，将 x_gird 变为 [[1, 2], [1, 2]]
        x_grid = tf.concat([tf.where(tf.equal(x_grid[:,0],x_grid[:,1]), x_grid[:,0]-1, x_grid[:,0])[:,None], 
                            x_grid[:,1][:,None]], axis=1)
        y_grid = tf.concat([tf.where(tf.equal(y_grid[:,0],y_grid[:,1]), y_grid[:,0]-1, y_grid[:,0])[:,None],
                            y_grid[:,1][:,None]], axis=1)
        # 将网格信息拼接汇总，形状：(o_h * o_w, 4)
        xy_grid = tf.concat([tf.tile(x_grid, multiples=[o_h, 1]),
                             tf.repeat(y_grid, repeats=o_w, axis=0)], axis=1)
        
        # 进行最大汇聚
        def region_pool(xy : tf.Tensor):
            return tf.reduce_max(fmap[xy[2]:xy[3], xy[0]:xy[1], :], axis=[0, 1])
        
        # 输出形状：(o_h * o_w, channels)
        output = tf.map_fn(region_pool, xy_grid, dtype=tf.float32)
        # reshape 成 (o_h, o_w, channels)
        output = tf.reshape(output, shape=[o_h, o_w, num_channels])
        return output

class ROIAligner(tf.keras.layers.Layer):
    """
    ### ROIAligner
    兴趣区域对齐层
    """
    def __init__(self, output_shape : tuple, **kwargs):
        super(ROIAligner, self).__init__(**kwargs)
        self.o_h, self.o_w = tf.constant(output_shape, dtype=tf.int32)
    
    def call(self, fmaps : tf.Tensor, rois : tf.Tensor, input_shape : tuple=None, **kwargs):
        # fmaps 形状：(batch_size, height, width, channels)
        # rois 形状：(batch_size, num_rois, 4)
        # input_shape = (height, width)
        def task_single_batch(fmap, roi):
            return ROIAligner.roi_align_single_fmap(fmap, roi, self.o_h, self.o_w)

        # 如果输入的是绝对坐标，则转换为相对坐标
        if input_shape is not None:
            rois = rois / tf.constant([input_shape[0], input_shape[1], 
                                       input_shape[0], input_shape[1]], dtype=tf.float32)
        # 输出形状：(batch_size, num_rois, o_h, o_w, channels)
        return tf.map_fn(lambda x: task_single_batch(x[0], x[1]), (fmaps, rois), dtype=tf.float32)
    
    @staticmethod
    @tf.function(input_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32)))
    def roi_align_single_fmap(fmap : tf.Tensor, rois : tf.Tensor, o_h : int, o_w : int):
        # fmap 形状：(height, width, channels)
        # roi 形状：(num_boxes, 4)
        # 获取 fmaps 的高度和宽度
        fmap_h, fmap_w = tf.shape(fmap)[0], tf.shape(fmap)[1]
        # 获取兴趣区域的在 fmap 的左上角坐标和右下角坐标
        x1 = rois[:,0] * tf.cast(fmap_w, tf.float32) # 形状：(num_boxes,)
        y1 = rois[:,1] * tf.cast(fmap_h, tf.float32)
        x2 = rois[:,2] * tf.cast(fmap_w, tf.float32)
        y2 = rois[:,3] * tf.cast(fmap_h, tf.float32)

        # tf.image.crop_and_resize 要求 roi 坐标表示顺序是 [y1, x1, y2, x2]
        rois = tf.stack([y1, x1, y2, x2], axis=1) # 形状：(num_boxes, 4)
        # 重新将 roi 转换为相对坐标
        # 因为像素的取值范围是 [0, fmap_h-1] 或 [0, fmap_w-1]，所以除以 fmap_h 或 fmap_w 转换到 [0, 1]
        rois = rois / tf.cast(tf.stack([fmap_h-1, fmap_w-1, fmap_h-1, fmap_w-1], axis=0), dtype=tf.float32)

        # tf.image.crop_and_resize 需要给定每个 roi 所属的 batch
        # 这里我们只有一个 fmap，因此所有的 roi 都属于第 0 个 batch
        box_indices = tf.zeros(shape=(tf.shape(rois)[0],), dtype=tf.int32)

        # 输出形状：(num_boxes, o_h, o_w, channels)
        # fmap[None, :, :, :] 为 fmap 增加一个 batch_size 维度
        return tf.image.crop_and_resize(
            fmap[None, :, :, :], rois, box_indices, crop_size=(o_h, o_w), method="bilinear")
    
def load_pascalVOC2012_images(path : str, dataset : str="train"):
    # 读取 Pascal VOC2012 数据
    import os
    dataset = "train.txt" if dataset == "train" else "val.txt" # 确认文件路径
    files = os.path.join(path, "ImageSets", "Segmentation", dataset)

    # 读取数据集对应的图像文件名
    with open(files,"r") as f:
        images_names = f.read().split()
    
    features, labels = [], [] # 初始化特征和标签
    for i, file in enumerate(images_names):
        feature_path = os.path.join(path, "JPEGImages", f'{file}.jpg')
        features.append(tf.image.decode_jpeg(tf.io.read_file(feature_path),channels=3))

        label_path = os.path.join(path, "SegmentationClass", f'{file}.png')
        labels.append(tf.image.decode_png(tf.io.read_file(label_path),channels=3))
    
    return features, labels

def show_segmentation_labels(images, labels, figsize=(12,4), ncol=4,dpi=50):
    fig, ax = plt.subplots(2,ncol,figsize=figsize,dpi=dpi)
    ax = ax.flatten()
    for i, (img,label) in enumerate(zip(images,labels)):
        ax[i].imshow(img)
        ax[i+ncol].imshow(label)
    plt.tight_layout()
    return

def create_RGBcolormap_labels(VOC_CLASSES : dict):
    # 构造从 RGB 值到类别索引的映射
    # 三通道数据，每个通道有 0~255 共 256 个取值，所有像素组合数为 256^3
    rgb_to_label = np.zeros((256**3,), dtype=np.int32) 
    for i, (key,value) in enumerate(VOC_CLASSES.items()):
        rgb_to_label[(value[0] * 256 + value[1]) * 256 + value[2]] = i
    return tf.constant(rgb_to_label,dtype=tf.int32)

def get_RGBlabel_index(RGB : tf.Tensor, rgb_to_label : tf.Tensor):
    # 将 RGB 像素信息映射到类别索引
    # RGB 形状：(height, width, channels)
    RGB = tf.cast(RGB,dtype=tf.int32)
    index = (RGB[:,:,0] * 256 + RGB[:,:,1]) * 256 + RGB[:,:,2] # 形状：(height, width)

    # 输出形状：(height, width)
    return tf.gather(rgb_to_label,index)

def random_crop(feature, label, height : int, width : int):
    # 随机裁剪输入特征和标签图像
    concat = tf.concat([feature, label],axis=2) # 先进行拼接，统一裁剪，维度 (h, w, 6)
    concat = tf.image.random_crop(concat, size=(height, width, 6))
    # 重新拆分为特征和标签
    feature, label = concat[:,:,0:3], concat[:,:,3:]
    return feature, label

class SegmentationDataLoader:
    def __init__(self, path : str, dataset : str, crop_size : list=[200, 300], batch_size : int=32) -> None:
        # 读取数据集
        self.X, self.Y = load_pascalVOC2012_images(path, dataset)
        self.crop_size = crop_size # 裁剪大小
        self.batch_size = batch_size
        
        # 过滤数据集
        self.X = self.filter(self.X)
        self.Y = self.filter(self.Y)

        # 对特征进行规范化
        self.mean = tf.constant([0.485, 0.456, 0.406]) # 数据集 RGB 通道均值
        self.std = tf.constant([0.229, 0.224, 0.225]) # 数据集 RGB 通道标准差

        # 创建 RGB 到类别索引的映射
        self.rgb_to_label = create_RGBcolormap_labels(VOC_CLASSES)

        print("read %s dataset with %d images"%(dataset, len(self)))

    # 过滤数据集
    def filter(self, images):
        images = [img for img in images if img.shape[0] >= self.crop_size[0] 
                                       and img.shape[1] >= self.crop_size[1]]
        return images

    # 标准化数据集
    def make_sample(self, img, label):
        feature, label = random_crop(img, label, self.crop_size[0], self.crop_size[1])
        # 做归一化
        feature = self.normalize(feature)
        # 从 RGB 转换到类别索引
        label = get_RGBlabel_index(label, self.rgb_to_label)
        return feature, label
    
    def normalize(self, img):
        return (tf.cast(img, dtype=tf.float32) / 255.0 - self.mean) / self.std
    
    # 给定索引 idx，访问数据
    def __getitem__(self, idx):
        return self.make_sample(self.X[idx], self.Y[idx])
    
    # 创建数据集
    def create_dataset(self):
        # 定义一个迭代器，从 self.X, self.Y 中读取图片
        # 因为 self.X 中每个元素的大小不同，我们没办法使用 tf.data.Dataset.from_tensor_slices
        def help_generator():
            for img, label in zip(self.X,self.Y):
                yield img, label
        
        # 创建 DataLoader
        dataset = tf.data.Dataset.from_generator(help_generator,
            output_signature=(tf.TensorSpec(shape=(None,None,3),dtype=tf.uint8),
                              tf.TensorSpec(shape=(None,None,3),dtype=tf.uint8)))
        # 加工样本，并通过 num_parallel_calls 和 prefetch 加速
        dataset = dataset.map(lambda x,y : self.make_sample(x,y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).shuffle(self.batch_size)
        return dataset
    
    # 数据集大小
    def __len__(self):
        return len(self.X)

def trans_corr2d(X : tf.Tensor, kernel : tf.Tensor):
    # 获取维度
    n_h, n_w = X.shape
    k_h, k_w = kernel.shape
    o_h, o_w = n_h + k_h - 1, n_w + k_w - 1
    # 初始化输出
    H = np.zeros((o_h,o_w),dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            H[i:(i + k_h), j:(j + k_w)] += X[i,j] * kernel
    return tf.constant(H)

def kernel_to_matrix(kernel):
    mat = np.zeros((4,9),dtype=np.float32)
    flat = np.zeros(5, dtype=np.float32)
    flat[0:2], flat[3:5] = kernel[0,:], kernel[1,:]
    mat[0,:5], mat[1, 1:6], mat[2,3:8], mat[3,4:9] = flat, flat, flat, flat
    return tf.constant(mat)

def init_bilinear_kernel(num_channels_in : int, num_channels_out : int, kernel_size : int):
    """
    ### 双线性插值初始化转置卷积参数

    Parameters
    ----------
    num_channels_in : int
        卷积核输入通道数
    num_channels_out : int
        卷积核输出通道数
    kernel_size : int
        卷积核尺寸
    """
    factor = (kernel_size + 1) // 2 # 向上取整
    # 根据卷积核尺寸的奇偶性，确定中心
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    
    og = (tf.reshape(tf.range(kernel_size,dtype=tf.float32), (-1,1)), 
          tf.reshape(tf.range(kernel_size,dtype=tf.float32), (1,-1)))
    # 用广播机制，生成一个 (kernel_size, kernel_size) 的矩阵
    filt = (1 - tf.abs(og[0] - center) / factor) * (1 - tf.abs(og[1] - center) / factor)

    # 初始化双线性插值转置卷积权重
    weight = np.zeros((kernel_size,kernel_size,num_channels_in,num_channels_out))
    # 为权重赋值
    weight[:,:,range(num_channels_in),range(num_channels_out)] = filt[:,:,None]

    return tf.Variable(weight,dtype=tf.float32)

def create_FCN_model(num_class : int, input_shape : tuple=(320,480,3), dropout : float=0.0):
    # 第一块：CNN 特征提取器
    resnet = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    # 从完整网络中选取前 92 层
    cnn_block = tf.keras.Model(
        inputs=resnet.input, outputs=resnet.layers[142].output
    )

    # 第二块：1X1 卷积层变换输出通道
    cnn_1x1 = tf.keras.layers.Conv2D(filters=num_class,kernel_size=1)

    # 第三块：转置卷积
    cnn_trans = tf.keras.layers.Conv2DTranspose(filters=num_class,kernel_size=32,strides=16,padding="same")

    model = tf.keras.models.Sequential([
        cnn_block,
        tf.keras.layers.Dropout(dropout),
        cnn_1x1,
        tf.keras.layers.Dropout(dropout),
        cnn_trans,
        tf.keras.layers.Softmax() # 转换为概率分布
    ])

    return model

def train_SegmentationFCN(model, train_ds, valid_ds, Epochs : int=10, lr : float=1e-3, verbose : int=1):
    # 定义损失函数和优化器
    def loss_func(y_true, y_pred):
        CEloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,reduction="none")
        loss = tf.reduce_mean(CEloss(y_true, y_pred), axis=(1,2)) # 输出形状：(batch_size,)
        return loss
    accuracy = tf.keras.metrics.sparse_categorical_accuracy
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    animator = utils.Animator(xlabel='epoch',legend=(("train loss",),("train acc","test acc")),
                              fmts=(("-",),("m--","g-,")),ncols=2,figsize=(10,4),xlim=[1, Epochs])
    
    for epoch in range(Epochs):
        # 存储每个迭代周期的损失和样本量
        loss_batch, train_acc = tf.constant(0.0), tf.constant(0.0)
        train_samples, train_batchs = tf.constant(0.0), tf.constant(0.0)

        for x_batch, y_batch in train_ds.create_dataset():
            with tf.GradientTape() as tape:
                y_hat = model(x_batch,training=True)
                loss = loss_func(y_true=y_batch,y_pred=y_hat)
                loss = tf.reduce_sum(loss) # 求和
            # 求梯度，更新参数
            weights = model.trainable_weights
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights))

            # 计算损失和评估指标
            batch_size = y_batch.shape[0]
            train_samples += tf.size(y_batch,out_type=tf.float32)
            train_batchs += batch_size
            loss_batch += loss
            train_acc += tf.reduce_sum(accuracy(y_batch,y_hat))
        
        if epoch == 0 or (epoch + 1)%verbose == 0:
            # 进行评估
            valid_acc, valid_samples = tf.constant(0.0), tf.constant(0.0)
            # 迭代验证集
            for x_batch, y_batch in valid_ds.create_dataset():
                y_hat = model(x_batch)
                valid_samples += tf.size(y_batch,out_type=tf.float32)
                valid_acc += tf.reduce_sum(accuracy(y_batch,y_hat))
            
            train_loss = (loss_batch / train_batchs).numpy()
            train_acc = (train_acc / train_samples).numpy()
            valid_acc = (valid_acc / valid_samples).numpy()
            
            animator.add(epoch+1,(train_loss,),ax=0) # 子图1
            animator.add(epoch+1,(train_acc,valid_acc),ax=1) # 子图2

    return model

def segmentation_predict(model, image):
    # 将预测概率转换为类别索引
    y_prob = model(image[None,:]) # 增加批量维度
    y_pred = tf.argmax(y_prob,axis=3,output_type=tf.int32)
    return y_pred[0] # 去掉批量维度

def get_RGBlabel_from_index(y_pred):
    # 将类别索引转换为 RGB 像素
    colormap = tf.constant([VOC_CLASSES[key] for key in VOC_CLASSES.keys()],dtype=tf.uint8)
    return tf.gather(colormap,y_pred)

