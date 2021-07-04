import numpy as np
import math
from graphviz import Digraph


def get_data(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            dataset.append(line)
    return dataset

class DecisionTree(object):
    def __init__(self):
        self.Tree = {'attribute_num':{}, 'child':{}}
        self.pred = None

    def fit(self, dataset):
        '''
        训练过程
        '''
        attr, attr_num = self.bulid_attr(dataset)
        self.label = list(attr['label'])
        self.label_num = attr_num[-1]
        del attr['label']
        self.attr = attr
        self.attr_num = attr_num[:-1]
        self.num_sample = len(dataset)
        self.generate_tree(dataset)

    def bulid_attr(self, dataset):
        '''
        分析数据集，获取数据的属性总数和各个属性的可取值
        '''
        attr = {}
        attr_cls = []
        num_attr = len(dataset[0])
        for i in range(num_attr):
            temp_attr = set([dataset[k][i] for k in range(len(dataset))])
            if i != (num_attr - 1):
                attr[i] = temp_attr
            else:
                attr['label'] = temp_attr
            attr_cls.append(len(temp_attr))
        return attr, attr_cls

    def generate_tree(self, dataset):
        '''
        调用grow函数生成决策树
        '''
        attr = self.attr.copy()
        self.grow(np.array(dataset), attr, self.Tree)

    def grow(self, dataset, attr, Tree, mode=1):
        '''
        递归调用grow函数，逐步生成决策树的每个结点
        '''
        if len(np.unique(dataset)) == len(self.attr_num) + 1 or len(attr)==0:
            label_data = [data[-1] for data in dataset]
            labels = list(set(label_data))
            label_count = []
            for label in labels:
                label_count.append(label_data.count(label))
            Tree = labels[np.argmax(np.array(label_count))]
            return

        if mode == 1:
            node_index, node_inform = self.cal_gain(dataset, attr)
        else:
            node_index, node_inform = self.cal_gain(dataset, attr, if_c45=True)
        Tree['attribute_num'] = node_index
        nodes = list(attr[node_index])
        del attr[node_index]
        for i, node in enumerate(nodes):
            n = node_inform[i]
            nodes_data = list(n.values())
            count_zeros = np.array([len(node_data) for node_data in nodes_data])
            zeros = len(count_zeros[count_zeros==0])
            if zeros >= len(count_zeros) - 1:
                label_index = np.where(count_zeros != 0)
                if len(label_index[0]):
                    Tree['child'][node] = self.label[label_index[0][0]]
            else:
                data = []
                for node_data in nodes_data:
                    data += node_data
                if len(data) == 0:
                    label_data = [data[-1] for data in dataset]
                    labels = list(set(label_data))
                    label_count = []
                    for label in labels:
                        label_count.append(label_data.count(label))
                    Tree['child'][node] = labels[np.argmax(np.array(label_count))]
                else:
                    nodes_data = dataset[data]
                    Tree['child'][node] = {'child':{}, 'attribute_num':{}}
                    attr_n = attr.copy()
                    self.grow(nodes_data, attr_n, Tree['child'][node])
        return

    def cal_gain(self, dataset, attri, if_c45=False):
        '''
        计算各个属性的信息熵增，获取信息熵增最大的属性和它的属性序号，帮grow函数挑选属性结点
        '''
        final_gains = {}
        at_gains = {}
        for attr_index in attri:
            attr = list(attri[attr_index])
            count = np.zeros(len(attr))
            at_gain = [{} for _ in range(len(attr))]
            for j in range(self.label_num):
                for i in range(len(attr)):
                    at_gain[i][self.label[j]] = []
            for m, sample in enumerate(dataset):
                for i in range(len(attr)):
                    if sample[attr_index] == attr[i]:
                        count[i] += 1
                        at_gain[i][sample[-1]].append(m)
            at_gains[attr_index] = at_gain
            final_gain = 0
            for j, at in enumerate(at_gain):
                gain = 0
                intrinsic_value = 0
                at = list(at.values())
                for i in at:
                    gain += len(i)/(count[j]+1e-9) * math.log2(len(i)/(count[j]+1e-9)+1e-9)
                final_gain += gain * count[j] / self.num_sample
                if if_c45:
                    intrinsic_value += ((count[j]+1e-9) / self.num_sample) * math.log2((count[j]+1e-9) / self.num_sample)
            if if_c45:
                final_gains[final_gain/intrinsic_value] = attr_index
            else:
                final_gains[-1 * final_gain] = attr_index
        print(final_gains)
        node = np.min(np.array(list(final_gains.keys())))
        print(node)
        node_index = final_gains[node]
        node_inform = at_gains[node_index]
        return node_index, node_inform

    # def cal_gini(self, dataset, attri):
    #     final_ginis = {}
    #     at_ginis = {}
    #     for attr_index in attri:
    #         attr = list(attri[attr_index])
    #         # print(attr)
    #         count = np.zeros(len(attr))
    #         at_gini = [{} for _ in range(len(attr))]
    #         for j in range(self.label_num):
    #             for i in range(len(attr)):
    #                 at_gini[i][self.label[j]] = []
    #         label_count = {}
    #         for label in self.label:
    #             label_count[label] = 0
    #         for m, sample in enumerate(dataset):
    #             label_count[sample[-1]] += 1
    #             for i in range(len(attr)):
    #                 if sample[attr_index] == attr[i]:
    #                     count[i] += 1
    #                     at_gini[i][sample[-1]].append(m)
    #         at_ginis[attr_index] = at_gini
    #         final_gini = 0
    #         for j, at in enumerate(at_gini):
    #             gini1 = 0
    #             gini2 = 0
    #             at = list(at.values())
    #             for i in at:
    #                 gini1 += math.pow((len(i)/(count[j]+1e-9)), 2)
    #                 gini2 += math.pow((count.sum()-len(i))/(count[j]+1e-9), 2)
    #             gini = 1 - gini

    #             final_gini = gini * count[j] / self.num_sample
        #         final_gains[-1 * final_gain] = attr_index
        #     # final_gains.append(-1 * final_gain)
        # # print(np.array(final_gains.keys()))
        # node = np.min(np.array(list(final_gains.keys())))
        # # print(node)
        # node_index = final_gains[node]
        # node_inform = at_gains[node_index]
        # return node_index, node_inform

    def plot_tree(self):
        '''
        调用sub_plot函数绘制决策树
        '''
        dot = Digraph(comment='Decision Tree')
        self.dot_root = 0
        dot.node(str(self.dot_root), 'Dataset')
        Tree = self.Tree.copy()
        self.sub_plot(dot, Tree['child'], '0')
        dot.render('./Tree.gv')
    
    def sub_plot(self, dot, Tree, root):
        '''
        递归调用sub_plot函数，逐步绘制决策树的每一个结点
        '''
        if isinstance(Tree, dict):
            for node in Tree:
                dot.node(str(self.dot_root+1), 'data')
                dot.edge(root, str(self.dot_root+1), label=str(node))
                self.dot_root += 1
                if isinstance(Tree[node], dict) and 'child' in Tree[node].keys():
                    self.sub_plot(dot, Tree[node]['child'], str(self.dot_root))
                else:
                    self.sub_plot(dot, Tree[node], str(self.dot_root))
        else:
            dot.node(str(self.dot_root), str(Tree))
            # dot.edge(root, str(self.dot_root+1))
            return

    def evaluate(self, test_data):
        '''
        调用predict函数，对测试集中的样本进行预测，计算决策树准确率(accuracy)
        '''
        Tree = self.Tree.copy()
        count = 0
        for data in test_data:
            attribute = data[:-1]
            gt = data[-1]
            self.predict(attribute, Tree)
            if self.pred == gt:
                count += 1
        acc = count / len(test_data)
        print('acc:{}'.format(acc))

    def predict(self, attribute, Tree):
        '''
        利用训练生成的决策树对样本进行预测
        '''
        if isinstance(Tree, dict) and 'child' in Tree.keys():
            sample_att = attribute[Tree['attribute_num']]
            Tree = Tree['child'][sample_att]
            self.predict(attribute, Tree)
        else:
            if isinstance(Tree, dict):
                self.pred = list(Tree.values())[0]
                return
            else:
                self.pred = Tree
                return


if __name__ == '__main__':
    path = 'homework.txt'
    dataset = get_data(path)
    id3 = DecisionTree()
    # id3.cal_gini(dataset, id3.attr)
    id3.fit(dataset)
    # print(id3.Tree)
    # id3.plot_tree()
    id3.evaluate(dataset)
