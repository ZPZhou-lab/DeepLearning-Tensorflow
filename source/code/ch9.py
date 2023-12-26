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

def create_wgan_gp_fashion_mnist_critic():
    def create_cnn_block(filters : int):
        block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            # 使用 LayerNormalization 替代 BatchNormalization
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            # 用卷积层代替池化层，进行下采样
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same')
        ])
        return block
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        create_cnn_block(32),
        tf.keras.layers.Dropout(0.25),
        create_cnn_block(64),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1) # 最后一层不做 sigmoid 激活
    ])

    return model

def wgan_gp_critic_loss(
    x_real : tf.Tensor, z_noise : tf.Tensor, critic : tf.keras.Model, G : tf.keras.Model, lambd : float=10, **kwargs):
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
    lambd : float, default=10
        梯度惩罚项的系数
    """
    # 生成器生成样本
    x_fake = G(z_noise, **kwargs)

    # 计算评估者损失
    loss = tf.reduce_mean(critic(x_real, **kwargs)) - tf.reduce_mean(critic(x_fake, **kwargs))

    # 梯度惩罚项
    # 采样 U[0, 1] 的随机噪声做插值
    epsilon = tf.random.uniform(shape=(x_real.shape[0], 1, 1, 1))
    x_tilde = epsilon * x_real + (1 - epsilon) * x_fake

    with tf.GradientTape() as tape:
        tape.watch(x_tilde)
        x_tilde_output = critic(x_tilde, **kwargs)
    # 梯度的形状：(batch_size, w, h, c)
    x_tilde_grad = tape.gradient(x_tilde_output, x_tilde)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(x_tilde_grad), axis=[1, 2, 3]))
    GP = tf.reduce_mean(tf.square(grad_norm - 1.0))

    loss = -loss + lambd * GP

    return loss

def train_wgan_gp_model(X : tf.Tensor, critic : tf.keras.Model, G : tf.keras.Model, noise_prior : Callable,
                        lambd : float=10, k : int=3, Epochs : int=10000, batch_size : int=128, 
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
    lambd : float, default=10
        梯度惩罚项的系数
    Epochs : int, default=10000
        WGAN 迭代次数
    k : int, default=3
        WGAN 每轮迭代中 cirtic 的迭代次数
    critic_lr, G_lr : float
        评估者和生成器的学习率
    """

    # 优化器
    critic_optim = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)
    G_optim = tf.keras.optimizers.Adam(learning_rate=G_lr)

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
            bz = x_real.shape[0]
            z_noise = noise_prior(bz) # 采样自噪声先验分布
            with tf.GradientTape() as tape:
                critic_loss = wgan_gp_critic_loss(x_real, z_noise, critic, G, lambd, training=True)
            grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optim.apply_gradients(zip(grads, critic.trainable_variables))
        
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

class AutoEncoderFashionMnist(tf.keras.Model):
    def __init__(self, latent_size : int=32):
        super(AutoEncoderFashionMnist, self).__init__()
        self.latent_size = latent_size
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_size, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Reshape((28, 28))
        ])
    
    def call(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        decoded = self.decoder(encoded, **kwargs)
        return decoded

    # 给定输入，返回编码后的结果
    def encode(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        return encoded
    
def add_normal_noise(x, std : float=0.2):
    """
    向图像中添加高斯噪声

    Parameters
    ----------
    std : float, default = 0.2
        控制噪声大小
    """
    x_noisy = x + std * tf.random.normal(shape=x.shape)
    # 确保像素值在 [0,1] 范围内
    x_noisy = tf.clip_by_value(x_noisy, clip_value_min=0.0, clip_value_max=1.0)

    return x_noisy

class DenoiseAutoEncoderFashionMnist(tf.keras.Model):
    def __init__(self, latent_size : int=32):
        super(DenoiseAutoEncoderFashionMnist, self).__init__()
        self.latent_size = latent_size
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding="same", activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_size, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_size,)),
            tf.keras.layers.Dense(7*7*16),
            tf.keras.layers.Reshape((7, 7, 16)),
            tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation='sigmoid')
        ])
    
    def call(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        decoded = self.decoder(encoded, **kwargs)
        return decoded

    # 给定输入，返回编码后的结果
    def encode(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        return encoded

class TimeSeriesAutoEncoder(tf.keras.Model):
    def __init__(self, latent_size : int=32, seq_len : int=140) -> None:
        super(TimeSeriesAutoEncoder, self).__init__()
        self.latent_size = latent_size
        self.seq_len = seq_len
        # 编码器使用一个双向 LSTM 层
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(seq_len,)),
            tf.keras.layers.Reshape((seq_len, 1)), # RNN 层的输入必须是 3D 张量
            tf.keras.layers.LSTM(32, go_backwards=True),
            tf.keras.layers.Dense(latent_size, activation='relu')
        ])
        # 解码器使用 MLP
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(seq_len, activation='sigmoid'),
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        decoded = self.decoder(encoded, **kwargs)
        return decoded
    
    def encode(self, x, **kwargs):
        encoded = self.encoder(x, **kwargs)
        return encoded
    
class ConvProbEncoder(tf.keras.Model):
    def __init__(self, latent_size : int=8, **kwargs):
        super(ConvProbEncoder, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
            # q(z|x) 需要两个参数：均值 mu 和方差 var = sigma^2
            # 每个参数的维度都是 latent_size，所以总输出维度是 2 * latent_size
            tf.keras.layers.Dense(latent_size * 2)
        ])
    
    def call(self, x, **kwargs):
        """
        x : tf.Tensor, shape = (batch_size, 28, 28, 1)
            输入的图像
        """
        # 编码器的输出，形状：(batch_size, latent_size * 2)
        encoded = self.encoder(x, **kwargs)
        # 将输出切分为 mu 和 var
        # 由于 var > 0，所以我们将全连接层的输出作为 log(var) 可以取值为任意实数
        # mu 和 log_var 的形状都是 (batch_size, latent_size)
        mu, log_var = tf.split(encoded, num_or_size_splits=2, axis=1)

        return mu, log_var

    # 从给定的图像编码得到隐变量 z 的分布（返回分布的参数）
    def encode(self, x):
        mu, log_var = self(x)
        return mu, log_var

    # 重参数化
    def reparameterize(self, mu, log_var, L : int=1):
        """
        Parameters
        ----------
        mu : tf.Tensor, shape = (batch_size, latent_size)
            均值
        log_var : tf.Tensor, shape = (batch_size, latent_size)
            方差的对数
        L : int, default = 1
            采样次数
        
        Returns
        ----------
        z : tf.Tensor, shape = (batch_size, L, latent_size)
            采样得到的隐变量
        """
        batch_size, latent_size = mu.shape
        # 将 mu 和 log_var 的形状都扩展为 (batch_size * L, latent_size)
        mu = tf.repeat(mu, repeats=L, axis=0)
        log_var = tf.repeat(log_var, repeats=L, axis=0)

        # 重参数化 z = g(eps, x) = mu + sigma * eps
        # eps 采样自标准多元高斯分布，形状：(batch_size * L, latent_size)
        eps = tf.random.normal(shape=(batch_size * L, latent_size))
        # 重参数化
        # 注意，exp(0.5 * log(sigma^2)) = exp(0.5 * 2 * log(sigma)) = sigma
        z = mu + tf.exp(0.5 * log_var) * eps

        return z

class ConvProbDecoder(tf.keras.Model):
    def __init__(self, latent_size : int=8, **kwargs):
        super(ConvProbDecoder, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_size,)),
            tf.keras.layers.Dense(7*7*32, activation='relu'),
            tf.keras.layers.Reshape((7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation='relu'),
            # 最后一层不使用 sigmoid 激活输出概率，而是直接输出 logits
            tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding="same")
        ])
    
    def call(self, z, **kwargs):
        """
        z : tf.Tensor, shape = (batch_size, latent_size)
            输入的隐变量
        """
        decoded = self.decoder(z, **kwargs)
        return decoded

    # 从给定的隐变量 z 解码出图像
    def decode(self, z, apply_sigmoid : bool=False):
        logits = self(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

def vae_elbo_loss(encoder : tf.keras.Model, decoder : tf.keras.Model, x : tf.Tensor, L : int=1, **kwargs):
    # 定义一个多元对角高斯分布的 log PDF 函数
    # 对于对角高斯分布，不相关等价于各分量独立
    # 每个分量的 gauss pdf(x) = 1 / sqrt(2 * pi * var) * exp(-0.5 * (x - mu)^2 / var)
    # log gauss pdf(x) = -0.5*log(2 * pi) - 0.5*log_var - 0.5 * (x - mu)^2 / var
    # 因此多元高斯分布的 log pdf 等于各分量的 log pdf 之和
    def log_normal_pdf(x, mu, log_var):
        # x 的形状：(batch_size * L, latent_size)
        # mu 和 log_var 的形状：(batch_size, latent_size)

        # 向将 mu 和 log_var 的形状扩展为 (batch_size * L, latent_size)
        mu = tf.repeat(mu, repeats=L, axis=0)
        log_var = tf.repeat(log_var, repeats=L, axis=0)

        log2pi = tf.math.log(2. * np.pi)
        log_pdf = -0.5 * (log2pi + log_var + tf.square(x - mu) / tf.exp(log_var))
        
        # 返回值对 latent_size 维度求和
        return tf.reduce_sum(log_pdf, axis=1)
        
    # 从给定的图像得到隐变量 z 的分布（返回分布的参数）
    mu, log_var = encoder(x, **kwargs)
    # 重参数化抽样得到 z，形状：(batch_size * L, latent_size)
    z = encoder.reparameterize(mu, log_var, L=L)
    
    # 从隐变量 z 解码得到重构图像，形状：(batch_size * L, w, h, num_channels)
    x_recon_logits = decoder(z, **kwargs)
    
    # 第一项：重构误差 log p(x|z)
    # 由于 x 是二值图像，所以我们使用二元交叉熵损失
    # 通过 tf.repeat 将 x 扩展为 (batch_size * L, w, h, num_channels) 与 x_recon_logits 形状匹配
    # 对图像维度求和，形状：(batch_size, L)，最后乘 -1 将负对数似然转换为似然
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_recon_logits,
        labels=tf.repeat(x, repeats=L, axis=0)),
        axis=[1, 2, 3])
    recon_loss *= -1.0
    
    # 第二项：先验 p(z)，均值为 0，方差为 1
    prior_loss = log_normal_pdf(z, mu=tf.zeros_like(mu), log_var=tf.zeros_like(log_var))
    # 第三项：近似后验 q(z|x)
    posterior_loss = log_normal_pdf(z, mu=mu, log_var=log_var)

    # 对 axis=0 维度求均值，此时同时平均了 batch_size 和 L 两个维度
    loss = tf.reduce_mean(recon_loss + prior_loss - posterior_loss)
    # 乘以 -1 将极大化 ELBO 转换为极小化损失
    return -loss

class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, latent_size : int=8, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.encoder = ConvProbEncoder(latent_size=latent_size)
        self.decoder = ConvProbDecoder(latent_size=latent_size)
    
    def call(self, x, **kwargs):
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var, L=1)
        decoded = self.sample(z)
        return decoded

    @tf.function
    def sample(self, eps=None):
        """
        eps : tf.Tensor, shape = (batch_size, latent_size)
            隐向量的随机样本
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_size))
        return self.decoder.decode(eps, apply_sigmoid=True)
    
def train_vae(X_train : tf.Tensor, X_test : tf.Tensor, model : tf.keras.Model,
              Epochs : int=10, L : int=1, batch_size : int=100, lr : float=1e-3, verbose : int=10):
    """
    Parameters
    ----------
    X_train : tf.Tensor, shape = (n_samples, w, h, num_channels)
        给定的训练图像数据集
    X_test : tf.Tensor, shape = (n_samples, w, h, num_channels)
        给定的测试图像数据集
    model : tf.keras.Model
        VAE 模型
    Epochs : int, default = 10
        训练轮数
    L : int, default = 1
        随机隐变量 z 采样次数
    """

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 创建数据迭代器
    train_set = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1000).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(X_test).shuffle(buffer_size=1000).batch(batch_size)

    # 记录训练过程
    animator = utils.Animator(xlabel='epoch', ylabel='loss', xlim=[1, Epochs], fmts=(('-',),('m--',)),
                              legend=(("train ELBO",), ("valid ELBO",)), figsize=(7, 3), ncols=2)
    
    # 训练 VAE
    for epoch in range(Epochs):
        # 计算训练集 loss
        train_loss = tf.keras.metrics.Mean()
        for x_batch in train_set:
            # 计算梯度
            with tf.GradientTape() as tape:
                loss = vae_elbo_loss(model.encoder, model.decoder, x_batch, L=L, training=True)
            train_loss(loss) # 记录 loss  
            # 更新参数
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # 绘制训练过程
        if epoch == 0 or (epoch + 1) % verbose == 0:            
            # 计算测试集 loss
            test_loss = tf.keras.metrics.Mean()
            for x_batch in test_set:
                loss = vae_elbo_loss(model.encoder, model.decoder, x_batch, L=L, training=False)
                test_loss(loss)
            
            # 添加到 animator
            animator.add(epoch + 1, (train_loss.result(),), ax=0)
            animator.add(epoch + 1, (test_loss.result(),), ax=1)
    
    return model

def plot_latent_space_transform(model : tf.keras.Model, cls, n : int=20, image_size : int=28):
    """
    Parameters
    ----------
    model : tf.keras.Model
        VAE 模型
    cls : Any
        分类器，用于获得模型的决策边界
    n : int, default = 20
        网格的大小
    image_size : int, default = 28
        图像的大小
    """
    from scipy import stats
    import seaborn as sns
    # 生成隐变量 z 的网格
    z1_grid = stats.norm.ppf(np.linspace(0.05, 0.95, n))
    z2_grid = stats.norm.ppf(np.linspace(0.05, 0.95, n))

    image_width = image_height = n * image_size
    image = np.zeros((image_height, image_width))

    for i, z1 in enumerate(z1_grid):
        for j, z2 in enumerate(z2_grid):
            # z = [z1, z2] 作为概率解码器输入
            z = tf.constant([[z1, z2]], dtype=tf.float32)
            
            # 解码得到图像
            x_recon = model.sample(z).numpy().reshape(image_size, image_size)
            # 因为 imshow 交换了 x 和 y 轴
            # 所以这里按照 [y, x] 的顺序填充图像
            image[(n - j - 1) * image_size : (n - j) * image_size,
                  i * image_size : (i + 1) * image_size] = x_recon

    # 可视化决策边界
    z_min, z_max, step = z1_grid.min(), z1_grid.max(), 0.02
    z_1, z_2 = np.meshgrid(np.arange(z_min, z_max, step), np.arange(z_min, z_max, step))
    z_grid = np.c_[z_1.ravel(), z_2.ravel()]
    prob_dist = cls.predict(z_grid)

    # 绘制从隐空间到图像空间的映射
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax = ax.flatten()
    ax[0].imshow(image, cmap='gray')
    ax[0].set_xlabel("$z_1$", fontsize=15)
    ax[0].set_ylabel("$z_2$", fontsize=15)
    ax[0].set_xticks(np.arange(0, n*image_size, image_size) + int(image_size / 2))
    ax[0].set_xticklabels(z1_grid.round(2))
    ax[0].set_yticks(np.arange(0, n*image_size, image_size) + int(image_size / 2))
    ax[0].set_yticklabels(z2_grid.round(2))
    ax[0].set_title("latent space transform", fontsize=18)

    colors = sns.color_palette("Set3", 10)
    for i in range(10):
        sns.scatterplot(x=z_grid[prob_dist==i, 0], y=z_grid[prob_dist==i, 1], 
                        label="class %d"%(i), color=colors[i], ax=ax[1])
    ax[1].legend(loc="upper right", fontsize=15)
    ax[1].set_xlabel("$z_1$", fontsize=15)
    ax[1].set_ylabel("$z_2$", fontsize=15)
    ax[1].set_title("latent space decision boundary", fontsize=18)
    plt.tight_layout()