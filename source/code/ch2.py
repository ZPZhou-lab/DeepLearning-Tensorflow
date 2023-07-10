import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from . import utils

def build_regression(n_train : int=20, n_valid : int=100, input_size : int=200, sigma : float=0.01, scale : float=0.1):
    """
    Parameters
    ----------
    n_train, n_valid : int
        训练集和验证集样本量
    input_size : int, default=200
        输入特征的维度
    sigma : float, default=0.01
        数据中高斯噪声的标准差，sigma越大，回归问题越困难
    scale : float, default=0.1
        正态特征的缩放因子，scale越大，回归问题越困难
    """
    true_w, trub_b = tf.ones([input_size,1])*0.01, 0.05
    # 生成训练集
    x_train = tf.random.normal((n_train,input_size)) * scale
    y_train = x_train @ true_w + trub_b + tf.random.normal((n_train,1),stddev=sigma)
    # 生成验证集
    x_valid = tf.random.normal((n_valid,input_size)) * scale
    y_valid = x_valid @ true_w + trub_b + tf.random.normal((n_valid,1),stddev=sigma)

    return x_train, y_train, x_valid, y_valid

class LinearRegression(tf.keras.Model):
    def __init__(self, input_dims : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = tf.Variable(tf.random.normal([input_dims,1]))
        self.b = tf.Variable(tf.zeros((1,)))
    
    def call(self, inputs, training=None, mask=None):
        return inputs @ self.W + self.b
    
def train(model, X_train : tf.Tensor, y_train : tf.Tensor, X_valid : tf.Tensor, y_valid : tf.Tensor, 
          batch_size : int=32, epochs : int=200, lr : float=0.01, verbose : int=5):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # 创建优化器
    mse = tf.keras.losses.MeanSquaredError() # 实例化损失函数

    animator = utils.Animator(xlabel="Epochs",legend=["train loss","valid loss"],
                              yscale="linear",xlim=[0,epochs]) # 创建动画实例

    for epoch in range(epochs):
        # 使用 Tensorflow API 创建批量数据的生成器
        batch_data = tf.data.Dataset.from_tensor_slices((X_train,y_train)).\
            batch(batch_size=batch_size).shuffle(buffer_size=batch_size)
        
        for x_batch, y_batch in batch_data:
            # 跟踪梯度
            with tf.GradientTape() as tape:
                y_hat = model(x_batch)
                loss = mse(y_batch,y_hat)
            
            # 选取参数，计算梯度
            weights = model.trainable_variables
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights)) # 直接调用优化器 API
        
        # 计算评估指标，添加到动画
        if epoch == 0 or (epoch+1)%verbose == 0: # 每 verbose 次更新一次
            y_train_hat = model(X_train)
            y_valid_hat = model(X_valid)

            train_loss = mse(y_train,y_train_hat).numpy()
            valid_loss = mse(y_valid,y_valid_hat).numpy()
            animator.add(epoch+1,[train_loss,valid_loss])
        
    return model

def l2_penalty(weights):
    return tf.reduce_sum(tf.pow(weights,2))

def train_weights_decay(model, X_train : tf.Tensor, y_train : tf.Tensor, X_valid : tf.Tensor, y_valid : tf.Tensor, 
                        lamda : float=1, batch_size : int=32, epochs : int=200, lr : float=0.01, verbose : int=5):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # 创建优化器
    mse = tf.keras.losses.MeanSquaredError() # 实例化损失函数

    animator = utils.Animator(xlabel="Epochs",legend=["train loss","valid loss"],
                              yscale="linear",xlim=[0,epochs]) # 创建动画实例

    for epoch in range(epochs):
        # 使用 Tensorflow API 创建批量数据的生成器
        batch_data = tf.data.Dataset.from_tensor_slices((X_train,y_train)).\
            batch(batch_size=batch_size).shuffle(buffer_size=batch_size)
        
        for x_batch, y_batch in batch_data:
            # 跟踪梯度
            with tf.GradientTape() as tape:
                y_hat = model(x_batch)
                loss = mse(y_batch,y_hat) + lamda * l2_penalty(model.W) # 增加正则化项
            
            # 选取参数，计算梯度
            weights = model.trainable_variables
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights)) # 直接调用优化器 API
        
        # 计算评估指标，添加到动画
        if epoch == 0 or (epoch+1)%verbose == 0: # 每 verbose 次更新一次
            y_train_hat = model(X_train)
            y_valid_hat = model(X_valid)

            train_loss = mse(y_train,y_train_hat).numpy()
            valid_loss = mse(y_valid,y_valid_hat).numpy()
            animator.add(epoch+1,[train_loss,valid_loss])
        
    return model

class LinearRegressionAPI(tf.keras.Model):
    def __init__(self, lamda : float=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = tf.keras.layers.Dense(
            1,kernel_regularizer=tf.keras.regularizers.l2(l2=lamda)) # 指定 kernel 的正则化器
    
    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)
    
def train_weights_decay_API(model, X_train : tf.Tensor, y_train : tf.Tensor, X_valid : tf.Tensor, y_valid : tf.Tensor, 
                            batch_size : int=32, epochs : int=200, lr : float=0.01, verbose : int=5):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # 创建优化器
    mse = tf.keras.losses.MeanSquaredError() # 实例化损失函数

    animator = utils.Animator(xlabel="Epochs",legend=["train loss","valid loss"],
                              yscale="linear",xlim=[0,epochs]) # 创建动画实例

    for epoch in range(epochs):
        # 使用 Tensorflow API 创建批量数据的生成器
        batch_data = tf.data.Dataset.from_tensor_slices((X_train,y_train)).\
            batch(batch_size=batch_size).shuffle(buffer_size=batch_size)
        
        for x_batch, y_batch in batch_data:
            # 跟踪梯度
            with tf.GradientTape() as tape:
                y_hat = model(x_batch)
                loss = mse(y_batch,y_hat) + model.losses # 增加正则化项
            
            # 选取参数，计算梯度
            weights = model.trainable_variables
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights)) # 直接调用优化器 API
        
        # 计算评估指标，添加到动画
        if epoch == 0 or (epoch+1)%verbose == 0: # 每 verbose 次更新一次
            y_train_hat = model(X_train)
            y_valid_hat = model(X_valid)

            train_loss = mse(y_train,y_train_hat).numpy()
            valid_loss = mse(y_valid,y_valid_hat).numpy()
            animator.add(epoch+1,[train_loss,valid_loss])
        
    return model

def dropout_layer(X : tf.Tensor, dropout : float=0.5):
    assert 0 <= dropout <= 1
    # 如果 dropout = 1，所有元素置 0
    if dropout == 1:
        return tf.zeros_like(X)
    # 如果 dropout = 0, 等价于不使用 Dropout
    if dropout == 0:
        return X
    
    # 创建一个 mask 掩码，通过生成 [0,1] 的均匀随机数
    # 将随机数小于 1 - dropout 的元素保留
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) < (1 - dropout)
    mask = tf.cast(mask, dtype=tf.float32) # 将 True / False 转换为 1 / 0 浮点矩阵
    return (mask * X) / (1 - dropout)

class LinearRegressionDropout(tf.keras.Model):
    def __init__(self, num_hiddens : int=64, dropout : float=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(num_hiddens,activation='tanh')
        self.dense2 = tf.keras.layersDense(1)
        self.dropout = dropout
    
    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        # 只有在训练模型时才使用dropout
        if training:
            x = dropout_layer(x,self.dropout)
        return self.dense2(x)
    
def train_dropout(model, X_train : tf.Tensor, y_train : tf.Tensor, X_valid : tf.Tensor, y_valid : tf.Tensor, 
                  batch_size : int=32, epochs : int=200, lr : float=0.01, verbose : int=5):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # 创建优化器
    mse = tf.keras.losses.MeanSquaredError() # 实例化损失函数

    animator = utils.Animator(xlabel="Epochs",legend=["train loss","valid loss"],
                              yscale="linear",xlim=[0,epochs]) # 创建动画实例

    for epoch in range(epochs):
        # 使用 Tensorflow API 创建批量数据的生成器
        batch_data = tf.data.Dataset.from_tensor_slices((X_train,y_train)).\
            batch(batch_size=batch_size).shuffle(buffer_size=batch_size)
        
        for x_batch, y_batch in batch_data:
            # 跟踪梯度
            with tf.GradientTape() as tape:
                y_hat = model(x_batch,training=True) # 训练时，使用 Dropout
                loss = mse(y_batch,y_hat)
            
            # 选取参数，计算梯度
            weights = model.trainable_variables
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights)) # 直接调用优化器 API
        
        # 计算评估指标，添加到动画
        if epoch == 0 or (epoch+1)%verbose == 0: # 每 verbose 次更新一次
            y_train_hat = model(X_train)
            y_valid_hat = model(X_valid)

            train_loss = mse(y_train,y_train_hat).numpy()
            valid_loss = mse(y_valid,y_valid_hat).numpy()
            animator.add(epoch+1,[train_loss,valid_loss])
        
    return model

class LinearRegressionDropoutAPI(tf.keras.Model):
    def __init__(self, num_hiddens : int=64, dropout : float=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(num_hiddens,activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
    
    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dropout1(x)  # API 只会在训练的时候启用 Dropout
        return self.dense2(x)
    
