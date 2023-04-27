#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 9:19 AM
# @Author  : charles_shen
# @File    : PSO_0_1_knapsack_problem.py
# @Software: PyCharm
# use PSO solve the 0-1 knapsack problem.
# 离散粒子群算法，本质就是将用二进制编码的方式进行解决，将离散问题空间映射到连续粒子运动空间中，其他速度位置更新原则还是和之前连续性粒子群
# 算法保持一致，但是不同的是粒子在状态空间的取值和变化只限于0和1两个值，而速度表示位置取值为 1 的可能性，其表达形式和logistics回归相似
# 连续问题也可以离散化，离散化后收敛速度变快，但是运行时间变长

# 粒子群算法流程
# 1.初始化粒子群，包括群体规模N，每个粒子的位置xi 和速度vi
# 2.计算每个粒子的适应度值fit[i]
# 3.通过适应度值取计算每个个体的适应度值以及全局适应度值
# 4.迭代更新每个粒子的速度和位置
# 5.进行边界条件处理
# 6、判断算法条件是否终止，否则返回步骤2

# 离散粒子群算法
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置字体和设置负号
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False

# 定义适应度函数
def func2(x):
    count = 0
    vv = 0
    for i in range(len(x)):
        if x[i] == 1:
            vv += value[i]
            count += volumn[i]
    if count <= 300:
        return vv
    else:
        return vv-1000


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
    p_best[i] = func2(x[i, :])
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
        if p_best[j] < func2(x[j,:]):
            p_best[j] = func2(x[j, :])
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