#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/21/23 3:14 PM
# @Author  : charles_shen
# @File    : PSO_scatter.py
# @Software: PyCharm
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.arange(0 ,9 ,0.01)
y = x + 6*np.sin(4*x) + 9*np.cos(5*x)
plt.figure(figsize=(15,15))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = x + 6sin(4x) + 9cos(5x)")
plt.show()

# 离散粒子群算法

# 定义适应度函数
def func2(x):
    x = str(x)
    x = x[1:-1]
    x = "".join(x.split())
    x = int(x, 2)
    x = x / int('11111111111111111111', 2) * 9
    return x + 6*np.sin(4*x) + 9*np.cos(5*x)

# 初始化粒子群相关参数
N = 100
D = 20
T = 200
c1 = 1.5
c2 = 1.5
w_max = 0.8
w_min = 0.8
x_max = 9
x_min = 0
v_max = 10
v_min = -10
x = np.random.randint(0,2,[N, D])
v = (v_max - v_min) * np.random.rand(N,D) + v_min
vx = np.zeros_like(v)
g_best = 100
# 初始化每个粒子的适应度值
p = x # 用来存储每个粒子的最佳位置
p_best = np.ones(N) # 用来存储每个粒子的适应度值
for i in range(N):
    p_best[i] = func2(x[i,:])
#     p[i,:] = x[j,:]
# 初始化全局最优位置与最优值
x_best = np.ones(D)
for i in range(N):
    if p_best[i] < g_best:
        g_best = p_best[i]
        x_best = x[i,:].copy()


gb = np.ones(T) # 用来存储每依次迭代的最优值
for i in range(T):
    for j in range(N):
        # 更新每个个体最优值和最优位置
        if p_best[j] > func2(x[j,:]):
            p_best[j] = func2(x[j,:])
            p[j,:] = x[j,:].copy()
        # 更新全局最优位置和最优值
        if p_best[j] < g_best:
            g_best = p_best[j]
            x_best = x[j, :].copy()
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        # 更新速度, 因为位置需要后面进行概率判断更新
        v[j,:] = w * v[j,:] + c1 * np.random.rand(1)*(p_best[j] - x[j,:]) + c2 * np.random.rand(1)*(x_best - x[j,:])
        # 边界条件处理
        for jj in range(D):
            if (v[j,jj] > v_max) or (v[j,jj] < v_min):
                v[j,jj] = v_min + np.random.rand(1)*(v_max - v_min)
        # 进行概率计算并且更新位置
        vx[j,:] = 1/(1 + np.exp(-v[j,:]))
        for ii in range(D):
            x[j, ii] = 1 if vx[j, ii] > np.random.rand(1) else 0
    gb[i] = g_best

    print("最优值为", gb[T - 1], "最优位置为", int("{}".format(x_best)))
    plt.plot(range(T), gb)
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.title("适应度进化曲线")
    plt.show()