import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_regression(N : int=100, sigma : float=0.3, add_dims : bool=True):
    """
    Parameters
    ----------
    N : int, default = 200
        样本量
    sigma : float, default = 0.3
        控制噪声方差的大小
    """
    x = np.random.uniform(low=-2,high=2,size=N)
    x = np.sort(x)
    noise = np.random.randn(N) * sigma # 高斯噪声
    y = np.sin(2*x) * np.exp(-x/2) + noise # 带有噪声的标签
    func = np.sin(2*x) * np.exp(-x/2) # 真实函数

    # 调整形状
    if add_dims:
        x = x.reshape((-1,1))
        y = y.reshape((-1,1))
        func = func.reshape((-1,1))

    return x, y, func

def regression_plot(x : np.ndarray, y : np.ndarray, func : np.ndarray, func_fit_x : np.ndarray=None, func_fit : np.ndarray=None):
    fig = plt.figure(figsize=(8,4))
    plt.scatter(x,y,color="royalblue",alpha=0.2,label="Samples")
    plt.plot(x,func,color="orange",label="True Function")
    if func_fit is not None:
        plt.plot(func_fit_x,func_fit,lw=2,ls="--",color="green",label="Fitted Function")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.ylim([-2.5,2.5])
    plt.xticks([-2,-1,0,1,2])

def build_classification(N : int=400, noise : float=0.3, show : bool=False):
    X,y = make_moons(n_samples=N,noise=noise,random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    if show:
        fig = plt.figure(figsize=(6,6))
        for i in range(2):
            label = "Positive" if i == 0 else "Negative"
            color = "red" if i == 0 else "blue"
            plt.scatter(X_train[y_train==i,0],X_train[y_train==i,1],marker="s",color="none",edgecolor=color,alpha=0.5,label="%s Train"%(label))
            plt.scatter(X_test[y_test==i,0],X_test[y_test==i,1],marker="+",color=color,alpha=0.5,label="%s Test"%(label))
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()

    return X_train, X_test, y_train, y_test