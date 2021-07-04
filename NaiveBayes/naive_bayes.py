import numpy as np


class naive_bayes(object):
    def __init__(self):
        self.class_count = [] # 每个类别的样本数量
        self.class_prior = [] # 每个类别的先验概率
        self.classes = [] # 类别集合

    def fit(self, x, y, smooth=True):
        """
        计算每个类别的先验概率和属性的条件概率，默认会进行拉普拉斯平滑
        """
        assert len(x) == len(y)
        num_samples = len(x)
        num_attribute = len(x[0])
        dataset_split = []
        self.classes = list(set(y))
        self.classes.sort()
        num_classes = len(self.classes)

        # 按照类别划分数据集
        for label in self.classes:
            index = np.where(y == label)
            dataset_split.append(x[index])
        self.class_count = [len(dataset) for dataset in dataset_split]

        # 计算先验概率
        if smooth:
            self.class_prior = [(num + 1) / (num_samples + num_classes) for num in self.class_count]
        else:
            self.class_prior = [num / num_samples for num in self.class_count]

        # 计算条件概率
        self.conditional_prob = []
        for i in range(num_attribute):
            cnd_prob = {}
            attrs = list(set([sample[i] for sample in x])) # 获取属性i所有取值的列表
            attrs.sort()
            for attr in attrs:
                cnd_prob[attr] = []
                for dataset in dataset_split:
                    current_attr = [data[i] for data in dataset]
                    if smooth:
                        cnd_prob[attr].append((current_attr.count(attr) + 1) / (len(dataset) + len(attrs)))
                    else:
                        cnd_prob[attr].append(current_attr.count(attr) / len(dataset))
            self.conditional_prob.append(cnd_prob)
        
    def predict(self, x):
        """
        通过fit中统计得到的概率值来预测样本集合的结果
        """
        num_classes = len(self.classes)
        assert num_classes != 0
        result = []
        for sample in x:
            prob = np.zeros(num_classes)
            for j in range(num_classes):
                prob[j] = self.class_prior[j] 
                for idx, cnd_prob in enumerate(self.conditional_prob):
                    prob[j] *= cnd_prob[sample[idx]][j]
            result.append(self.classes[np.argmax(prob)])
        return result

if __name__ == '__main__':
    classify = naive_bayes()
    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 10))
    y = np.array([1, 1, 2, 2, 5, 6])
    classify.fit(X, y)
    result = classify.predict(X)
    print(result)