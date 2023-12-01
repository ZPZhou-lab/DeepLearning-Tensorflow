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

def create_mnist_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def normal_noise_generator(batch_size, noise_dim : int=64):
    return tf.random.normal(shape=(batch_size, noise_dim))

def create_mnist_generator(input_dims : int=64):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dims,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(28 * 28 * 1, activation='sigmoid'),
        tf.keras.layers.Reshape((28, 28, 1))
    ])
    return model

def gan_discriminator_loss(
    x_real : tf.Tensor, z_noise : tf.Tensor, D : tf.keras.Model, G : tf.keras.Model, **kwargs):
    """
    Parameters
    ----------
    x_real : tf.Tensor
        采样自真实数据的样本
    z_noise : tf.Tensor
        采样自噪声先验分布的样本
    D : tf.keras.Model
        判别器
    G : tf.keras.Model
        生成器
    """
    # 生成器生成样本
    x_fake = G(z_noise, **kwargs)

    # 真实样本和生成样本的标签
    y_true = tf.ones(shape=(x_real.shape[0], 1))
    y_fake = tf.zeros(shape=(x_fake.shape[0], 1))

    # 计算判别器损失
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, D(x_real, **kwargs))) + \
            tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_fake, D(x_fake, **kwargs)))
    
    return loss

def gan_generator_loss(z_noise : tf.Tensor, D : tf.keras.Model, G : tf.keras.Model, **kwargs):
    """
    Parameters
    ----------
    z_noise : tf.Tensor
        采样自噪声先验分布的样本
    D : tf.keras.Model
        判别器
    G : tf.keras.Model
        生成器
    """
    # 生成器生成样本
    x_fake = G(z_noise, **kwargs)

    # 生成样本的标签
    y_fake = tf.ones(shape=(x_fake.shape[0], 1))

    # 计算生成器损失
    # 极小化 log(1 - D(G(z))) 等价于极大化 log(D(G(z)))
    # 极大化 log(D(G(z))) 等价于极小化 -log(D(G(z)))
    # 借用二分类交叉熵损失函数，将标签 y_fake 设为 1，等价于极小化 -log(D(G(z)))
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_fake, D(x_fake, **kwargs)))
    
    return loss

def train_gan_model(X : tf.Tensor, D : tf.keras.Model, G : tf.keras.Model, noise_prior : Callable,
                    Epochs : int=10000, k : int=5, batch_size : int=64, 
                    D_lr : float=1e-3, G_lr : float=1e-3, beta_1 : float=0.9, verbose : int=500):
    """
    Parameters
    ----------
    X : tf.Tensor
        真实数据集
    D, G : tf.keras.Model
        判别器和生成器
    noise_prior : Callable
        噪声先验分布
    Epochs : int, default=10000
        GAN 迭代次数
    k : int, default=5
        GAN 每轮迭代中判别器的迭代次数
    D_lr, G_lr : float
        判别器和生成器的学习率
    beta_1 : float, default=0.9
        Adam 优化器的 beta_1 动量项参数
    """

    # 优化器
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr, beta_1=beta_1)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr, beta_1=beta_1)

    # 创建数据迭代器
    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).shuffle(1000).repeat()
    iterator = iter(dataset)

    # 记录训练过程
    # 多增加一列，用于绘制生成样本
    animator = utils.Animator(xlabel='epoch', ylabel='loss', xlim=[1, Epochs], fmts=(('-',), ('m--',)),
                              legend=[("discriminator",), ("generator",)], figsize=(12,3), ncols=3)

    # 训练 GAN
    for epoch in range(Epochs):
        # 判别器训练阶段
        D.trainable, G.trainable = True, False # 判别器可训练，冻结生成器
        for _ in range(k):
            x_real = next(iterator) # 采样自真实数据
            z_noise = noise_prior(batch_size) # 采样自噪声先验分布
            with tf.GradientTape() as tape:
                D_loss = gan_discriminator_loss(x_real, z_noise, D, G, training=True)
            grads = tape.gradient(D_loss, D.trainable_variables)
            D_optimizer.apply_gradients(zip(grads, D.trainable_variables))
        
        # 生成器训练阶段
        D.trainable, G.trainable = False, True # 生成器可训练，冻结判别器
        z_noise = noise_prior(batch_size) # 采样自噪声先验分布
        with tf.GradientTape() as tape:
            G_loss = gan_generator_loss(z_noise, D, G, training=True)
        grads = tape.gradient(G_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(grads, G.trainable_variables))

        # 绘制训练过程
        if epoch == 0 or (epoch + 1) % verbose == 0:
            # 绘制生成样本
            z_noise = noise_prior(1)
            x_fake = G(z_noise, training=False)
            animator.axes[2].cla()
            animator.axes[2].imshow(x_fake[0, :, :, 0], cmap='gray')
            animator.axes[2].axis('off')
            # 绘制损失
            animator.add(epoch + 1, (D_loss.numpy(),), ax=0)
            animator.add(epoch + 1, (G_loss.numpy(),), ax=1)

    return D, G

def create_dcgan_mnist_discriminator():
    def create_cnn_block(filters : int):
        block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            # 用卷积层代替池化层，进行下采样
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same')
        ])
        return block
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        create_cnn_block(8),
        tf.keras.layers.Dropout(0.5),
        create_cnn_block(16),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def create_dcgan_mnist_generator(input_dims : int=64):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dims,)),
        # 做一次投影变换，变换形状到 (7, 7, 128)
        tf.keras.layers.Dense(7 * 7 * 128, use_bias=False),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # 先作用一层卷积，提升学习能力
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # 用转置卷积进行上采样
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='tanh')
    ])
    return model

def create_wgan_mnist_critic():
    def create_cnn_block(filters : int):
        block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            # 用卷积层代替池化层，进行下采样
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same')
        ])
        return block
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        create_cnn_block(16),
        tf.keras.layers.Dropout(0.25),
        create_cnn_block(32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1) # 最后一层不做 sigmoid 激活
    ])

    return model

def wgan_critic_loss(
    x_real : tf.Tensor, z_noise : tf.Tensor, critic : tf.keras.Model, G : tf.keras.Model, **kwargs):
    """
    Parameters
    ----------
    x_real : tf.Tensor
        采样自真实数据的样本
    z_noise : tf.Tensor
        采样自噪声先验分布的样本
    cirtic : tf.keras.Model
        评估者
    G : tf.keras.Model
        生成器
    """
    # 生成器生成样本
    x_fake = G(z_noise, **kwargs)

    # 计算评估者损失
    loss = tf.reduce_mean(critic(x_real, **kwargs)) -\
              tf.reduce_mean(critic(x_fake, **kwargs))
    # 注意 cirtic 是极大化 EM distance，等价于极小化 -EM distance
    # 为了让所有优化器都是极小化损失，这里取负号
    loss = -loss

    return loss

def wgan_generator_loss(z_noise : tf.Tensor, critic : tf.keras.Model, G : tf.keras.Model, **kwargs):
    """
    Parameters
    ----------
    z_noise : tf.Tensor
        采样自噪声先验分布的样本
    cirtic : tf.keras.Model
        评估者
    G : tf.keras.Model
        生成器
    """
    # 生成器生成样本
    x_fake = G(z_noise, **kwargs)

    # 计算生成器损失
    loss = -tf.reduce_mean(critic(x_fake, **kwargs))
    
    return loss

def train_wgan_model(X : tf.Tensor, critic : tf.keras.Model, G : tf.keras.Model, noise_prior : Callable,
                     c : float=0.01, k : int=5, Epochs : int=10000, batch_size : int=64, 
                     critic_lr : float=1e-4, G_lr : float=1e-4, verbose : int=500):
    """
    Parameters
    ----------
    X : tf.Tensor
        真实数据集
    critic, G : tf.keras.Model
        判别器和生成器
    noise_prior : Callable
        噪声先验分布
    c : float, default=0.01
        参数裁剪的阈值
    Epochs : int, default=10000
        WGAN 迭代次数
    k : int, default=5
        WGAN 每轮迭代中 cirtic 的迭代次数
    critic_lr, G_lr : float
        评估者和生成器的学习率
    """

    # 优化器
    critic_optim = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)
    G_optim = tf.keras.optimizers.RMSprop(learning_rate=G_lr)

    # 创建数据迭代器
    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).shuffle(1000).repeat()
    iterator = iter(dataset)

    # 记录训练过程
    # 多增加一列，用于绘制生成样本
    animator = utils.Animator(xlabel='epoch', ylabel='loss', xlim=[1, Epochs], fmts=(('-',), ),
                              legend=[("EM distance",), ], figsize=(7,3), ncols=2)

    # 训练 WGAN
    for epoch in range(Epochs):
        # 评估者训练阶段
        critic.trainable, G.trainable = True, False # 评估者可训练，冻结生成器
        for _ in range(k):
            x_real = next(iterator) # 采样自真实数据
            z_noise = noise_prior(batch_size) # 采样自噪声先验分布
            with tf.GradientTape() as tape:
                critic_loss = wgan_critic_loss(x_real, z_noise, critic, G, training=True)
            grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optim.apply_gradients(zip(grads, critic.trainable_variables))

            # 参数裁剪
            for w in critic.trainable_weights:
                w.assign(tf.clip_by_value(w, -c, c))

        # 生成器训练阶段
        critic.trainable, G.trainable = False, True # 生成器可训练，冻结判别器
        z_noise = noise_prior(batch_size) # 采样自噪声先验分布
        with tf.GradientTape() as tape:
            G_loss = wgan_generator_loss(z_noise, critic, G, training=True)
        grads = tape.gradient(G_loss, G.trainable_variables)
        G_optim.apply_gradients(zip(grads, G.trainable_variables))

        # 绘制训练过程
        if epoch == 0 or (epoch + 1) % verbose == 0:
            # 绘制生成样本
            z_noise = noise_prior(1)
            x_fake = G(z_noise, training=False)
            animator.axes[1].cla()
            animator.axes[1].imshow(x_fake[0, :, :, 0], cmap='gray')
            animator.axes[1].axis('off')
            # 绘制损失
            animator.add(epoch + 1, (-critic_loss.numpy(),), ax=0)

    return critic, G

