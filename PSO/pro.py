#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/27/23 4:10 PM
# @Author  : charles_shen
# @File    : pro.py
# @Software: PyCharm
# 针对数模校赛的问题提出了基于粒子群算法的有效解

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# 实现数组中1的个数计数
def count_ones(x):
    count = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i, j] == 1:
                count = count+1
    return count


# 需要实现将特定的位置变为-1
def change(x, y):
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i, j] == 1:
                y[i, j] = -1
    return y


# 编写函数 输入x和对应的多维系数，再检查是否符合二重隐私保护
# z表示需要实现的多重保护方案
def check_pro(z, x):
    flag = 1  # 记录是否完成保护
    no_pro = 0  # 记录未完成保护的数目
    while len(x) != 0:
        same = 0
        for i in range(1, len(x)):
            # 记录相同数组的位置
            index_lo = np.ones(1)-1
            if (x[i] == x[0]).all():
                same = same + 1
                index_lo.append(i)
        # 删除已经形成多重保护的数组
        for i in index_lo[::-1]:
            x = np.delete(x, int(i), 0)
        if same < z:
            flag = 0
            no_pro = no_pro + len(index_lo)
        same = 0
    return flag, no_pro

# 导入数据
all_data = pd.read_excel('/Users/shenfeiyang/Documents/GitHub/mathematical-modeling/data/B_data.xlsx', sheet_name=None)
data_1 = all_data['二元1']
data_2 = all_data['二元2']
data_3 = all_data['多元1']
data_4 = all_data['多元2']

# 将data1转化为numpy形式
data_1 = np.asarray(data_1)
print(data_1)

# 设置适应度函数
def func(x):
    k = change(x, data_1)
    d, c = check_pro(2, k)
    if d == 1:
        return count_ones(x)
    else:
        return x.size+c *10

# 初始化粒子群相关参数
N = 100
D = 10
T = 200
c1 = 1.5
c2 = 1.5
w_max = 0.8
w_min = 0.8
# x_max = 9
# x_min = 0
v_max = 10
v_min = -10
x = np.random.randint(0, 2, [N, D])
v = (v_max - v_min) * np.random.rand(N, D) + v_min
vx = np.random.rand(N, D) # 这个是将速度转换成概率的矩阵
value = [89, 59, 19, 43, 100, 72, 44, 16, 7, 64]
volumn = [95, 75, 23, 73, 50, 22, 6, 57, 89, 98]

# 初始化每个粒子的适应度值
p = x  # 用来存储每个粒子的最佳位置
p_best = np.ones(N)  # 用来存储每个粒子的适应度值
for i in range(N):
    p_best[i] = func(x[i, :])
#     p[i,:] = x[j,:]


# 初始化全局最优位置与最优值
g_best = 100
x_best = np.ones(D)
for i in range(N):
    if p_best[i] > g_best:
        g_best = p_best[i]
        x_best = x[i, :].copy()

gb = np.ones(T)  # 用来存储每依次迭代的最优值
for i in range(T):
    for j in range(N):
        # 更新每个个体最优值和最优位置
        if p_best[j] < func(x[j,:]):
            p_best[j] = func(x[j, :])
            p[j, :] = x[j, :].copy()
        # 更新全局最优位置和最优值
        if p_best[j] > g_best:
            g_best = p_best[j]
            x_best = x[j, :].copy()
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        # 更新速度, 因为位置需要后面进行概率判断更新
        v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p_best[j] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
        # 边界条件处理
        for jj in range(D):
            if (v[j, jj] > v_max) or (v[j, jj] < v_min):
                v[j, jj] = v_min + np.random.rand(1) * (v_max - v_min)
        # 进行概率计算并且更新位置
        vx[j, :] = 1 / (1 + np.exp(-v[j, :]))
        for ii in range(D):
            r = np.random.rand(1)
            x[j, ii] = 1 if vx[j, ii] > r else 0
    gb[i] = g_best

print("最优值为", gb[T - 1], "最优位置为", x_best)
plt.plot(range(T), gb)
plt.xlabel("迭代次数")
plt.ylabel("适应度值")
plt.title("适应度进化曲线")
plt.show()