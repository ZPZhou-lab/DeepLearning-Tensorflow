import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from . import utils

CIFAR100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 
                   'beaver', 'bed', 'bee', 'beetle', 'bicycle', 
                   'bottle', 'bowl', 'boy', 'bridge', 'bus', 
                   'butterfly', 'camel', 'can', 'castle', 'caterpillar', 
                   'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 
                   'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 
                   'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
                   'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
                   'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
                   'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 
                   'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
                   'plain', 'plate', 'poppy', 'porcupine', 'possum', 
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
                   'rose', 'sea', 'seal', 'shark', 'shrew', 
                   'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                   'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
                   'tank', 'telephone', 'television', 'tiger', 'tractor', 
                   'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
                   'whale', 'willow_tree', 'wolf', 'woman', 'worm']

CIFAR10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

FASHION_MNIST_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def classification_predict(model : tf.keras.Model, inputs : tuple, batch_size : int=32, label : bool=True):
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

def train(model, train_tensors : tuple, test_tensors : tuple, 
          batch_size : int=64, epochs : int=10, lr : float=0.01, verbose : int=1):
    """
    Parameters
    ----------
    train_tensors, test_tensors : tuple
        训练 / 测试数据的 `tf.Tensor` 元组，元组包含两个元素\n
        `tensors[0]` 表示输入特征，`tensors[1]` 表示标签
    """
    # 创建损失函数和优化器
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    # 度量函数
    accuracy = tf.keras.metrics.categorical_accuracy
    
    # 创建动画实例
    animator = utils.Animator(xlabel="Epochs",legend=(("train loss",),("train acc","test acc")),
                        xlim=[1,epochs],ncols=2,figsize=(10,4),fmts=(("-",),("m--","g-,"))) 

    for epoch in range(epochs):
        batch_data = tf.data.Dataset.from_tensor_slices(train_tensors).\
            batch(batch_size).shuffle(batch_size)
        for x_batch, y_batch in batch_data:
            # 跟踪梯度
            with tf.GradientTape() as tape:
                prob = model(x_batch,training=True)
                loss = loss_func(y_batch,prob)
            
            # 选择参数，计算梯度
            weights = model.trainable_variables
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads,weights))
        
        # 计算评估指标，添加到动画
        if epoch == 0 or (epoch+1)%verbose == 0: # 每 verbose 次更新一次
            # 做出预测
            y_train_prob = classification_predict(
                model,(train_tensors[0],),batch_size,label=False)
            y_test_prob = classification_predict(
                model,(test_tensors[0],),batch_size,label=False)
            
            train_loss = tf.reduce_mean(loss_func(train_tensors[1],y_train_prob)).numpy()
            train_acc = tf.reduce_mean(accuracy(train_tensors[1],y_train_prob)).numpy()
            test_acc = tf.reduce_mean(accuracy(test_tensors[1],y_test_prob)).numpy()

            animator.add(epoch+1,(train_loss,),ax=0) # 子图1
            animator.add(epoch+1,(train_acc,test_acc),ax=1) # 子图2
    
    return model

def corr2d(X : tf.Tensor, kernel : tf.Tensor):
    """
    二维矩阵的互相关运算
    """
    n_h, n_w = X.shape
    k_h, k_w = kernel.shape
    Y = tf.Variable(tf.zeros((n_h - k_h + 1,n_w - k_w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j].assign(tf.reduce_sum(X[i:i+k_h, j:j+k_w] * kernel))
    return Y

def transform_Conv2d(Conv2dLayer, X : tf.Tensor):
    # 添加批量维度和通道维度
    X = tf.reshape(X, (1,) + X.shape + (1,))
    Y = Conv2dLayer(X) # 进行卷积运算
    # 变换回原来的形状
    Y = tf.reshape(Y, Y.shape[1:3])
    return Y

def corr2d_multi_in(X : tf.Tensor, kernel : tf.Tensor):
    """
    假设 X 和 kernel 的第 0 个维度表示通道维度\n
    先遍历 X 和 kernel 的通道维度（第 0 个维度）\n
    再把结果加在一起，求和在第 0 个维度及逆行，axis = 0\n
    """
    return tf.reduce_sum([corr2d(x, k) for x, k in zip(X, kernel)], axis=0)

def corr2d_multi_in_out(X : tf.Tensor, kernel : tf.Tensor):
    """
    假设 X 的第 0 个维度表示输入通道维度\n
    假设kernel 的第 0 个维度表示输出通道维度，第 1 个维度表示输入通道数量\n
    先遍历 kernel 的输出通道维度（第 0 个维度）\n
    把每个输出通道 kernal 与 X 做 corr2d_multi_in\n
    最后把多个输出通道用 tf.stack 拼接在一起\n
    """
    return tf.stack([corr2d_multi_in(X, k) for k in kernel], axis=0)

def corr2d_multi_in_out_1x1(X : tf.Tensor, kernel : tf.Tensor):
    # X 是三维张量，维度 (c, n_h, n_w)
    # kernel 是四维张量，维度 (d, c, 1, 1)
    c, n_h, n_w = X.shape
    d = kernel.shape[0]
    X = tf.reshape(X, (c, n_h*n_w)) # 将像素所在的两个维度合并，变换为 (c, n_h*n_w)
    kernel = tf.reshape(kernel, (d, c)) # 转换维度为 (d, c)
    # 全连接层中的矩阵乘法
    Y = kernel @ X # 输出维度 (d, n_h*n_w)
    # 恢复像素所在的两个维度
    Y = tf.reshape(Y, (d, n_h, n_w))
    return Y

def pool2d(X : tf.Tensor, pool_size : tuple, mode : str="max"):
    """
    二维汇聚层运算
    """
    p_h, p_w = pool_size # 汇聚窗口大小
    Y = tf.Variable(tf.zeros(shape=(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i,j].assign(tf.reduce_max(X[i:i+p_h, j:j+p_w]))
            elif mode == "mean":
                Y[i,j].assign(tf.reduce_mean(X[i:i+p_h, j:j+p_w]))
    
    return Y

def load_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print("训练集特征形状：",x_train.shape)
    print("训练集标签形状：",y_train.shape)
    # 对数据做规范化
    x_train, x_test = x_train / 255, x_test / 255 # 转换到 [0, 1]
    # 对标签进行编码
    num_class = 100
    y_train = tf.one_hot(y_train.flatten(),depth=num_class)
    y_test = tf.one_hot(y_test.flatten(),depth=num_class)

    return (x_train,y_train), (x_test,y_test)

def LeNet(input_shape : tuple, num_class : int, activation : str="sigmoid"):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(6,kernel_size=5,padding="same",activation=activation),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(16,kernel_size=5,activation=activation),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation=activation),
        tf.keras.layers.Dense(84,activation=activation),
        tf.keras.layers.Dense(num_class),
        tf.keras.layers.Softmax()
    ])

def AlexNet(input_shape : tuple, num_class : int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.experimental.preprocessing.Resizing(height=224,width=224),
        # 第一块卷积 + 汇聚层
        tf.keras.layers.Conv2D(96,kernel_size=11,strides=4,activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=3,strides=2),
        # 第二块卷积 + 汇聚层
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 连续三个卷积层 + 汇聚层，通道数减少
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 拉直，送入全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_class),
        tf.keras.layers.Softmax()
    ])

def vgg_block(num_convs : int, num_channels : int):
    block = tf.keras.models.Sequential()
    # 添加多个卷积层
    for _ in range(num_convs):
        block.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,padding="same",activation="relu"))
    # 添加汇聚层
    block.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    return block

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu')
    ])

class Inception(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 第一条路径
        self.path1_1 = tf.keras.layers.Conv2D(c1, kernel_size=1, activation="relu")
        # 第二条路径
        self.path2_1 = tf.keras.layers.Conv2D(c2[0], kernel_size=1, activation="relu")
        self.path2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3, padding="same", activation="relu")
        # 第三条路径        
        self.path3_1 = tf.keras.layers.Conv2D(c3[0], kernel_size=1, activation="relu")
        self.path3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5, padding="same", activation="relu")
        # 第四条路径
        self.path4_1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=1,padding="same")
        self.path4_2 = tf.keras.layers.Conv2D(c4,kernel_size=1,activation="relu")
        # 拼接算子
        self.concat = tf.keras.layers.Concatenate(axis=-1)
    
    def call(self, inputs, training=None, mask=None):
        path1 = self.path1_1(inputs)
        path2 = self.path2_2(self.path2_1(inputs))
        path3 = self.path3_2(self.path3_1(inputs))
        path4 = self.path4_2(self.path4_1(inputs))
        return self.concat([path1,path2,path3,path4])

def GoogLeNetBlock1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
    ])

def GoogLeNetBlock2():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation="relu"),
        tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
    ])

def GoogLeNetBlock3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
    ])

def GoogLeNetBlock4():
    return tf.keras.models.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

def GoogLeNetBlock5():
    return tf.keras.models.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAveragePooling2D()
    ])

def GoogLeNet(input_shape : tuple, num_class : int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.experimental.preprocessing.Resizing(width=96,height=96),
        # 顺序5层计算
        GoogLeNetBlock1(),
        GoogLeNetBlock2(),
        GoogLeNetBlock3(),
        GoogLeNetBlock4(),
        GoogLeNetBlock5(),
        # 输出得到概率
        tf.keras.layers.Dense(num_class),
        tf.keras.layers.Softmax()
    ])

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # 计算移动方差的平方根倒数 1 / σ
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 缩放和移位
    inv *= gamma
    Y = (X - moving_mean) * inv + beta
    return Y

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    # 定义层结构
    def build(self, input_shape):
        # 拿取通道数量
        weight_shape = [input_shape[-1], ]
        # 添加拉伸和偏移参数
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
                                     initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
                                    initializer=tf.initializers.zeros, trainable=True)
        # 记录均值和方差的移动平均值
        # 注意移动平均值不参与训练，因此 trainalbe = False
        self.moving_mean = self.add_weight(name='moving_mean',shape=weight_shape, 
                                           initializer=tf.initializers.zeros,trainable=False)
        self.moving_var = self.add_weight(name='moving_var',shape=weight_shape,
                                          initializer=tf.initializers.ones,trainable=False)
        # 再调用父类方法
        super(BatchNorm, self).build(input_shape)
    
    # 更新均值和方差的移动平均值
    def assign_moving_average(self, variable : tf.Variable, value):
        momentum = 0.9 # 移动平均动量
        # 更新公式：v_{k+1} = m * v_k + (1-m) * v_{new}
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)
    
    # 定义 BatchNorm 层逻辑
    def call(self, inputs, training : bool, *args, **kwargs):
        # 训练模式
        if training:
            # 去掉最后一个通道维度
            axes = list(range(len(inputs.shape) - 1))
            # 计算批量均值和方差
            # 注意方差的计算需要用到均值，tf.stop_gradient 停止跟踪方差中的均值
            # 维度 (1, 1, 1, c)
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True) 
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            
            # 变换形状，将变量为 1 的维度去除
            # 维度 (c, )
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            
            # 更新均值和方差的移动平均值
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_var, batch_variance)
            # 继承自 tf.keras.layers.Layer 的 add_update 方法来更新参数
            self.add_update(mean_update)
            self.add_update(variance_update)
            # 均值和方差使用批量统计值
            mean, variance = batch_mean, batch_variance
        # 预测模式
        else:
            # 均值和方差直接使用移动平均值
            mean, variance = self.moving_mean, self.moving_var
        # 执行 BatchNorm 算子
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
                            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output

def LeNet_BatchNorm(input_shape : tuple, num_class : int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        # 第一层卷积
        tf.keras.layers.Conv2D(filters=6, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        # 第二层卷积
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        # 全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(num_class),
        tf.keras.layers.Softmax()
    ])

class Residual(tf.keras.Model):
    def __init__(self, num_channels, strides : int=1, use_1x1conv : bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 卷积层
        self.conv1 = tf.keras.layers.Conv2D(num_channels,kernel_size=3,padding="same",strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels,kernel_size=3,padding="same")
        # 是否使用 1x1 卷积核来调整通道数和分辨率
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        # BatchNorm层
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=None, mask=None):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(inputs)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            inputs = self.bn3(self.conv3(inputs))
        # 残差跳连
        Y += inputs 
        return tf.keras.activations.relu(Y)

def ResNetBlock1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels : int, num_residuals : int, first_block=False, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))
    
    def call(self, inputs, *args, **kwargs):
        for layer in self.residual_layers:
            inputs = layer(inputs)
        return inputs

def ResNet(input_shape : tuple, num_class : int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.experimental.preprocessing.Resizing(width=96,height=96),
        # 第一层
        ResNetBlock1(),
        # 后续四层残差块
        ResNetBlock(num_channels=64,num_residuals=2,first_block=True),
        ResNetBlock(num_channels=128,num_residuals=2),
        ResNetBlock(num_channels=256,num_residuals=2),
        ResNetBlock(num_channels=512,num_residuals=2),
        # 全剧平均池化，得到输出概率
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_class),
        tf.keras.layers.Softmax()
    ])