#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170528
@author: JohnHuiWB
'''

import numpy as np


class kMeans(object):
    def __init__(self):
        self._assassment = 0
        self._center = 0
        self._k = 0
        self._data_set = 0


    def _cal_euclidean_distance(self, vec1, vec2):
    # 计算欧式距离
        return (np.sum((vec1 - vec2)**2))**0.5


    def _create_center(self):
    # 随机构建质心
        n = self._data_set.shape[1]
        self._center = np.zeros((self._k, n))
        for j in range(n):
        # 确保质心在数据集的范围之内
            minJ = self._data_set[:,j].min()
            rangeJ = self._data_set[:,j].max() - minJ
            self._center[:,j] = (minJ + rangeJ * np.random.rand(self._k, 1)).T


    def analyze(self, data_set, k):
    # K-means
        self._data_set = np.array(data_set)
        self._k = k
        m, n = self._data_set.shape
        self._assassment = np.zeros((m, 2))
        self._create_center() # 随机构建质心

        center_change = True
        while center_change:
            center_change = False
            for i in range(m):
            # 处理每个数据
                min_dist = np.inf # 离质心的最小距离初始化为无穷大
                min_indx = -1 # 离得最近的质心，初始化为-1
                for j in range(self._k):
                    dist = self._cal_euclidean_distance(self._data_set[i,:], self._center[j,:])
                    if dist < min_dist:
                        min_dist = dist
                        min_indx = j

                if int(self._assassment[i, 0]) != min_indx:
                    center_change = True
                self._assassment[i,:] = min_indx, min_dist**2

            for c in range(k):
                count = 0
                sum_dist = np.zeros((1, n))
                for i in range(m):
                    if self._assassment[i,0] == c:
                        sum_dist += self._data_set[i,:]
                        count += 1
                self._center[c,:] = sum_dist / count


    def assassment(self):
        return self._assassment


    def center(self):
        return self._center


    def predict(self, dataX):
        dataX = np.array(dataX)
        min_dist = np.inf # 离质心的最小距离初始化为无穷大
        min_indx = -1 # 离得最近的质心，初始化为-1
        for i in range(self._k):
            dist = self._cal_euclidean_distance(dataX, self._center[i,:])
            if dist < min_dist:
                min_dist = dist
                min_indx = i
        return min_indx



class biKMeans(kMeans):
    def __init__(self):
        super().__init__()


    def _create_center(self):
        self._center = [np.mean(self._data_set, axis = 0).tolist()]

    def analyze(self, data_set, k):
    # bisecting K-means
        self._data_set = np.array(data_set)
        self._k = k
        m, n = self._data_set.shape
        self._assassment = np.zeros((m, 2))
        self._create_center() # 构建质心集合

        center0 = np.array(self._center[0])
        for i in range(m):
            self._assassment[i, 1] = self._cal_euclidean_distance(center0, self._data_set[i,:])**2

        km = kMeans()
        while self._k > len(self._center):
            lowest_SSE = np.inf

            plot_graph(self._data_set.tolist(), self._assassment[:,0], np.array(self._center))

            for i in range(len(self._center)):
                km.analyze(self._data_set[np.nonzero(self._assassment[:,0]==i)[0], :], 2)
                new_center, new_assassment = km.center(), km.assassment()
                
                SSE_of_split = np.sum(new_assassment[:,1])
                SSE_of_not_split = np.sum(self._assassment[np.nonzero(self._assassment[:,0]!=i)[0], 1])
                new_SSE = SSE_of_split + SSE_of_not_split

                if new_SSE < lowest_SSE:
                    lowest_SSE = new_SSE
                    best_cluster_to_split = i
                    best_new_center = new_center
                    best_new_assassment = new_assassment.copy()
            best_new_assassment[np.nonzero(best_new_assassment[:,0]==1)[0], 0] = len(self._center)
            best_new_assassment[np.nonzero(best_new_assassment[:,0]==0)[0], 0] = best_cluster_to_split

            self._assassment[np.nonzero(self._assassment[:,0]==best_cluster_to_split)[0], :] = best_new_assassment
            self._center[best_cluster_to_split] = best_new_center[0, :].tolist()
            self._center.append(best_new_center[1, :].tolist())

        plot_graph(self._data_set.tolist(), self._assassment[:,0], np.array(self._center))




def plot_graph(data, labels, center):
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['b', 'c', 'm', 'r', 'y']
    data_arr = np.array(data)
    n = np.shape(data_arr)[0]
    allplot = {}
    for i in range(n):
        if labels[i] in allplot:
            allplot[int(labels[i])]['xcord'].append(data_arr[i, 0])
            allplot[int(labels[i])]['ycord'].append(data_arr[i, 1])
        else:
            allplot[int(labels[i])] = {}
            allplot[int(labels[i])]['xcord'] = [data_arr[i, 0]]

            allplot[int(labels[i])]['ycord'] = [data_arr[i, 1]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label in allplot:
        color = colors.pop()
        ax.scatter(allplot[label]['xcord'], allplot[label]['ycord'], s = 30, c = color)
    plt.scatter(center[:, 0], center[:, 1], c = 'k', marker = '*', s = 600)   
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()