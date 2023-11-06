import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from . import utils

from typing import Callable
import time
from tqdm import tqdm
import os

def generate_FGSM_adversarial_perturbation(
    model : tf.keras.Model, x : tf.Tensor, y : tf.Tensor, loss_func : Callable, epsilon : float=0.01):
    """
    ### 生成 FGSM 对抗扰动
    Parameters
    ----------
    model : tf.keras.Model
        训练好的神经网络模型
    x : tf.Tensor
        输入样本，形状 (batch_size, height, width, channels)
    y : tf.Tensor
        输入样本的真实标签，形状 (batch_size,)
    loss_func : Callable
        损失函数 loss_func(y_true, y_pred)
    epsilon : float, default = 0.01
        扰动系数, 默认值为 0.01
    """

    # 计算梯度 \nabla_x J(\theta, x, y)
    x = tf.Variable(x) # 转换为 tf.Variable 类型
    with tf.GradientTape() as tape:
        y_hat = model(x, training=False) # 前向传播
        loss = loss_func(y, y_hat) # 计算损失
    
    # 反向传播，计算梯度
    grad = tape.gradient(loss, x) # 形状 (batch_size, height, width, channels)

    # 计算扰动 eta = epsilon * sign(\nabla_x J(\theta, x, y))
    eta = epsilon * tf.sign(grad)
    # 生成对抗样本
    x_adv = x + eta
    
    return eta, x_adv

def show_adversarial_examples(
    model : tf.keras.Model, x : tf.Tensor, eta : tf.Tensor, x_adv : tf.Tensor):
    """
    ### 展示对抗样本攻击
    Parameters
    ----------
    model : tf.keras.Model
        训练好的神经网络模型
    x : tf.Tensor
        输入样本，形状 (batch_size, height, width, channels)
    eta : tf.Tensor
        对抗扰动，形状 (batch_size, height, width, channels)
    x_adv : tf.Tensor
        对抗样本，形状 (batch_size, height, width, channels)
    """
    def make_prediction(x):
        y_hat_dist = model(x, training=False) # 获得概率分布，形状 (batch_size, num_classes)
        probs = tf.reduce_max(y_hat_dist, axis=-1) # 预测类的概率，形状 (batch_size,)
        y_hat = tf.argmax(y_hat_dist, axis=-1) # 获得预测类，形状 (batch_size,)
        return y_hat, probs
    
    y_hat, probs = make_prediction(x) # 计算原始样本的预测概率
    y_hat_adv, probs_adv = make_prediction(x_adv) # 计算对抗样本的预测概率
    y_hat_eta, probs_eta = make_prediction(eta) # 计算扰动 eta 的预测概率

    # 展示对抗样本攻击
    nrow, ncol = x.shape[0], 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2.5, nrow*2.5))
    # 如何 nrow = 1，需要将 axes 转换为二维数组
    if nrow == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(nrow):
        # 原始样本
        axes[i, 0].imshow(x[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original\npred={y_hat[i]}  prob={probs[i]:.4f}')
        # 扰动 eta
        axes[i, 1].imshow(eta[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Perturbation\npred={y_hat_eta[i]}  prob={probs_eta[i]:.4f}')
        # 对抗样本
        axes[i, 2].imshow(x_adv[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Adversarial\npred={y_hat_adv[i]}  prob={probs_adv[i]:.4f}')
    plt.tight_layout()

def generate_FGSM_target_class_adversarial_perturbation(
    model : tf.keras.Model, x : tf.Tensor, y_target : tf.Tensor, loss_func : Callable, epsilon : float=0.01, Epochs : int=5):
    """
    ### 生成基于 FGSM 的指定类别对抗扰动
    Parameters
    ----------
    model : tf.keras.Model
        训练好的神经网络模型
    x : tf.Tensor
        输入样本，形状 (batch_size, height, width, channels)
    y_target : tf.Tensor
        攻击时指定的类别，形状 (batch_size,)
    loss_func : Callable
        损失函数 loss_func(y_true, y_pred)
    epsilon : float, default = 0.01
        扰动系数, 默认值为 0.01.
    Epochs : int, default = 5
        迭代次数, 默认值为 5.
    """

    x_adv = x
    for i in range(Epochs):
        x_adv = tf.Variable(x_adv)
        with tf.GradientTape() as tape:
            y_hat = model(x_adv, training=False) # 前向传播
            loss = loss_func(y_target, y_hat) # 计算损失
        
        # 计算梯度 \nabla_x J(\theta, x, y)
        # 反向传播，计算梯度
        grad = tape.gradient(loss, x_adv) # 形状 (batch_size, height, width, channels)

        # 计算扰动 eta = epsilon * sign(\nabla_x J(\theta, x, y))
        eta = epsilon * tf.sign(grad)
        # 更新对抗样本
        x_adv = x_adv - eta
    
    eta = x - x_adv # 计算扰动 eta
    
    return eta, x_adv