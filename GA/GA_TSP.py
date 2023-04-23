#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 4:51 PM
# @Author  : charles_shen
# @File    : GA_TSP.py
# @Software: PyCharm
# GA_TSP 针对TSP问题重载了 交叉(crossover)、变异(mutation) 两个算子
"""
在该程序中，需要注意前半段的路径生成方法。
"""
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

# 地点个数
num_points = 100
# 随机生成地点
points_coordinate = np.random.rand(num_points, 2)*100  # generate coordinate of points
# print(points_coordinate)
# 该函数用于计算两个输入集合的距离，通过metric参数指定计算距离的不同方式得到不同的距离度量值
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
print(distance_matrix)

def cal_total_distance(routine):
    '''
    The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=1000, prob_mut=1)
best_points, best_distance = ga_tsp.run()

# fig, ax = plt.subplots(2, 1)
# best_points_ = np.concatenate([best_points, [best_points[0]]])
# best_points_coordinate = points_coordinate[best_points_, :]
# ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
#
# ax[1].plot(ga_tsp.generation_best_Y)
# plt.show()
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
plt.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
plt.plot(ga_tsp.generation_best_Y)
plt.show()
