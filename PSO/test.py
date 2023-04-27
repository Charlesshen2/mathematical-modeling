#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 9:18 AM
# @Author  : charles_shen
# @File    : simple_GA.py
# @Software: PyCharm
""""
用于测试代码
求函数f（x，y）=3cos（xy）+x+y2的最小值，其中x的取值范围为［-4，4］，y的取值范围为［-4，4］

"""
# 将图形画出
import matplotlib.pyplot as plt
import random
import numpy as np

X = np.arange(-4, 4, 0.01)
Y = np.arange(-4, 4, 0.01)

x, y = np.meshgrid(X, Y)
z = 3*np.cos(x*y)+x+y**2

fig = plt.figure()
ax3 = plt.axes(projection="3d")
ax3.plot_surface(x, y, z, cmap='rainbow')
plt.show()

# 算法开始
# 设定参数

N = 100  # 粒子群数目
D = 2  # 维数
T = 200  # 最大迭代数
c1 = 1.5
c2 = 1.5
x_max = 4  # 粒子最大限制
x_min = -4  # 粒子最小限制
v_max = 1  # 粒子最大速度
v_min = -1  # 粒子最小速度

w_max = 0.8
w_min = 0.4


def func(x):
    return 3*np.cos(x[0]*x[1])+x[0]+x[1]**2


# 初始化粒子群位置
x = np.random.rand(N, D)*(x_max-x_min)+x_min
v = np.random.rand(N, D)*(x_max-x_min)+x_min

# 设定每个参数的个体最优位置
p_best = x
# 全局最优的x位置
p_g_best = np.ones(D)
# 每次迭代的总体最优值
value_best = np.ones(T)
# 全局最优值
value_g_best = float('inf')

for i in range(N):
    value_best[i] = func(x[i])

for i in range(T):
    for j in range(N):
        # 个体值更新
        if value_best[j] > func(x[j]):
            value_best[j] = func(x[j])
            p_best[j] = x[j].copy()

        # 全局最优值更新
        if value_g_best > func(x[j]):
            value_g_best = func(x[j])
            p_g_best = x[j].copy()

        # 计算动态w值
        w = w_max-(w_max-w_min)*i/T

        # 更新x，v值
        v[j] = w*v[j]+c1*np.random.rand(1)*(p_best[j]-x[j])+c2*np.random.rand(1)*(p_g_best-x[j])

        x[j] = x[j]+v[j]

        # 粒子位置和速度限制
        for ii in range(D):
            if v[j, ii] > v_max or v[j, ii] < v_min:
                v[j, ii] = v_min + np.random.rand(1)*(v_max-v_min)

            if x[j, ii] > x_max or x[j, ii] < x_min:
                x[j, ii] = x_min + np.random.rand(1)*(v_max-v_min)

    # 记录循环的最优值
    value_best[i] = value_g_best


print(value_best[T-1])

plt.plot(range(T), value_best)
plt.show()






