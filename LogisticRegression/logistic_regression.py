import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing


def read_data(path):
    """
    读取数据
    """
    dataset = pd.read_csv(path, header=None)
    return dataset

def split_dataset(dataset, val_ratio=0.2):
    """
    划分训练集和验证集
    """
    # 打乱数据集
    dataset_shuffle = shuffle(dataset)
    num_samples = dataset_shuffle.shape[0]
    # 获得验证集
    val_dataset = dataset_shuffle.iloc[:int(num_samples*val_ratio), :]
    val_samples = val_dataset.iloc[:, :-1]
    val_labels = val_dataset.iloc[:, -1]
    # 获得训练集
    train_dataset = dataset_shuffle.iloc[int(num_samples*val_ratio):, :]
    train_samples = train_dataset.iloc[:, :-1]
    train_labels = train_dataset.iloc[:, -1]
    return train_samples, train_labels, val_samples, val_labels

def preprocess(dataset, val_dataset, test_dataset):
    """
    数据预处理，对各个特征进行归一化，使模型收敛更迅速
    """
    scaler = preprocessing.StandardScaler().fit(dataset)
    dataset = scaler.transform(dataset)
    val_dataset = scaler.transform(val_dataset)
    test_dataset = scaler.transform(test_dataset)
    return dataset, val_dataset, test_dataset

def write_result(dataset, results):
    """
    将测试集合结果写成csv形式
    """
    index = range(1, dataset.shape[0]+1)
    var_1 = np.array(dataset.iloc[:, 0])
    var_2 = np.array(dataset.iloc[:, 1])
    var_3 = np.array(dataset.iloc[:, 2])
    var_4 = np.array(dataset.iloc[:, 3])
    dataframe = pd.DataFrame({'样本序号':index,'变量名1':var_1, '变量名2':var_2, '变量名3':var_3, '变量名4':var_4, '真钞or假钞':results})
    dataframe.to_csv("test.csv", index=False, sep=',', encoding='gbk')

class LogisticRegression(object):
    """
    Logistic Regression的简易实现
    """
    def __init__(self, num_iter=500, init_lr=0.01, verbose=True, val=False):
        self.num_iter = num_iter
        self.learning_rate = init_lr
        self.verbose = verbose
        self.eval = val
        self.w = None

    def fit(self, dataset, labels):
        '''
        进行训练
        '''
        num_samples, num_attr = dataset.shape
        w = np.zeros((num_attr+1, 1)) # 初始化权重为0
        X = np.mat(dataset)
        X = np.insert(X, num_attr, values=1, axis=1)
        gt = np.mat(labels).T

        # 开始训练
        for i in range(self.num_iter):
            pred = self.__sigmoid(X.dot(w))
            # 计算梯度，并更新weight的值
            err = pred - gt
            w_grad = (X.T).dot(err)
            w -= w_grad * self.learning_rate
            # 计算损失函数的值，采用交叉熵损失函数
            y = np.where(pred>0.5, 1, 0).T
            loss = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
            # 显示训练集的损失函数值和准确率
            if self.verbose:
                if self.eval:
                    acc = (y==np.array(labels)).sum() / len(labels)
                    print('iteration{}:\tloss:{}\tacc:{}'.format(i, loss.sum()/num_samples, acc))
                else:
                    print('iteration{}:\tloss:{}'.format(i, loss.sum()/num_samples))
        # 保存训练结果
        self.w = w

    def pred(self, x):
        """
        利用训练好的权重对数据集进行预测
        """
        num_attr = x.shape[1]
        x = np.mat(x)
        x = np.insert(x, num_attr, values=1, axis=1)
        return self.__sigmoid(x.dot(self.w))

    def __sigmoid(self, x):
        """
        sigmoid 函数的实现
        """
        return 1 / (1 + np.exp(-x))

    def score(self, x, labels):
        """
        计算训练好的模型在数据集上的准确率
        """
        preds = self.pred(x)
        preds = np.array(preds).squeeze()
        preds = np.where(preds>0.5, 1, 0)
        acc = (preds==np.array(labels)).sum() / len(labels)
        print(acc)
        return acc


if __name__ == '__main__':
    # 读取数据集
    path = 'train_dataset.txt'
    dataset = read_data(path)
    # 将数据集按照4:1的比例划分为训练集和验证集
    train_samples, train_labels, val_samples, val_labels = split_dataset(dataset, val_ratio=0.2)
    # 读取测试集
    test_path = 'test_dataset.txt'
    test_dataset = read_data(test_path)
    # 对训练集，验证集，测试集进行预处理：归一化
    train_samples, val_samples, test_samples= preprocess(train_samples, val_samples, test_dataset)

    # 初始化一个LogisticRegression模型
    clf = LogisticRegression(val=True)
    # 进行训练，训练完成后在验证集上进行测试计算准确率
    clf.fit(train_samples, train_labels)
    acc = clf.score(val_samples, val_labels)

    # 对测试集进行预测，并将预测结果写入csv文件
    preds = clf.pred(test_samples)
    preds = np.array(preds).squeeze()
    preds = np.where(preds>0.5, 1, 0)
    write_result(test_dataset, preds)
    