import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split('\t')
            dataset.append([float(data[4]), float(data[3])])
    return np.array(dataset)

class kMeans(object):
    def __init__(self, num_cluster, tol=0.0001):
        self.num_cluster = num_cluster
        self.tol = tol
    
    def rand_center(self, dataset, k):
        '''
        随机选取k个均值向量
        '''
        centers_ind = np.random.choice(dataset.shape[0], k)
        centers = dataset[centers_ind]
        return centers

    def kmeans(self, dataset, k=2):
        '''
        kmeans算法实现
        '''
        # 随机选取k个均值向量
        centers = self.rand_center(dataset, k=k)
        while True:
            # 计算样本点与均值向量的距离
            dists = []
            for center in centers:
                # dist = self.cal_dist(dataset, center)
                dist = self.distSLC(dataset, center)
                dists.append(dist)
            dists = np.array(dists)
            # 根据与每个样本点距离最小的均值向量的索引来确定样本点的簇标记
            split_clusters = np.argmin(dists, axis=0)
            # 根据计算出的簇标记划分数据集
            clusters = [dataset[split_clusters == i] for i in range(k)]
            # 计算新的均值向量
            new_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            # 判断是否达到收敛，收敛则结束迭代
            # if self.cal_dist(new_centers, centers).sum() <= self.tol:
            if self.distSLC(new_centers, centers).sum() <= self.tol:
                minimal_dist = np.amin(dists, axis=0)
                minimal_sse = [np.power(minimal_dist[split_clusters == i], 2).sum() for i in range(k)]
                break
            # 未收敛，将新的均值向量赋值给均值向量，进行下一轮迭代
            centers = new_centers

        # 保存划分的结果(包括簇划分，最终生成的均值向量，每个簇的距离平方和)
        self.final_center = new_centers
        self.clusters = clusters
        self.minimal_sse = minimal_sse

    def cal_dist(self, a, b):
        '''
        计算两个向量的欧式距离
        '''
        dist = a - b
        dist = np.power(dist, 2)
        dist = np.sqrt(dist.sum(axis=1))
        return dist

    def distSLC(self, vecA, vecB):
        '''
        根据经纬度计算地图上两点间的距离
        '''
        pi = np.pi
        if len(vecB.shape) == 1:
            vecB = np.expand_dims(vecB, 0)
        a = np.sin(vecA[:, 1] * pi / 180) * np.sin(vecB[:, 1] * pi / 180)
        b = np.cos(vecA[:, 1] * pi / 180) * np.cos(vecB[:, 1] * pi / 180) * \
                        np.cos(pi * (vecB[:, 0] - vecA[:, 0]) / 180)
        s = np.clip(a+b, -1, 1)
        return np.arccos(s) * 6371.0

    def plot(self, name=None):
        '''
        结果可视化
        '''
        fig = plt.figure()
        rect=[0.0,0.0,1.0,1.0]
        scatterMarkers=['s', 'o', '^', '8', 'p', \
                        'd', 'v', 'h', '>', '<']
        axprops = dict(xticks=[], yticks=[])
        ax0=fig.add_axes(rect, label='ax0', **axprops)
        imgP = plt.imread('Portland.png')
        ax0.imshow(imgP)
        ax1=fig.add_axes(rect, label='ax1', frameon=False)
        for i in range(self.num_cluster):
            markerStyle = scatterMarkers[i % len(scatterMarkers)]
            ax1.scatter(self.clusters_end[i][:, 0], self.clusters_end[i][:, 1], marker=markerStyle, s=90)
        ax1.scatter(self.center_end[:, 0], self.center_end[:, 1], marker='+', s=300)
        if name == None:
            plt.savefig('result.jpg')
        else:
            plt.savefig(name+'.jpg')

    def biKmeans(self, dataset):
        clusters = []
        centers = []
        SSE = []
        # 将所有点看成一个簇，计算它的均值向量和误差平方和
        clusters.append(dataset)
        centers.append(np.mean(dataset, axis=0))
        SSE.append(np.power(self.distSLC(dataset, centers[0]), 2).sum())
        
        # 当簇数目小于k时，进行划分
        while len(clusters) < self.num_cluster:
            sses = []
            clusters_possible = []
            centers_possible = []
            cluster_sse = []
            # 分别选择现有簇中的每个簇继续进行划分，计算其划分后的均值向量和误差平方和
            for i, cluster in enumerate(clusters):
                self.kmeans(cluster)
                sses.append(np.sum(self.minimal_sse) + np.sum(SSE) - SSE[i])
                centers_possible.append(self.final_center)
                clusters_possible.append(self.clusters)
                cluster_sse.append(self.minimal_sse)

            # 选择使得误差平方和最小的划分，更新簇划分C,均值向量和误差平方和
            ind = np.argmin(sses)
            del clusters[ind]
            for cluster_p in clusters_possible[ind]:
                clusters.append(cluster_p)
            del centers[ind]
            for center_p in centers_possible[ind]:
                centers.append(center_p)
            del SSE[ind]
            for sse in cluster_sse[ind]:
                SSE.append(sse)

        # 保存划分簇的结果和均值向量
        self.center_end = np.array(centers)
        self.clusters_end = clusters


if __name__ == '__main__':
    path = 'places.txt'
    dataset = read_data(path)
    clustering = kMeans(3)
    clustering.biKmeans(dataset)
    clustering.plot()
    
