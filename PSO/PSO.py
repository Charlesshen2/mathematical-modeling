#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/21/23 9:19 AM
# @Author  : charles_shen
# @File    : PSO.py
# @Software: PyCharm
"""
代码转载：https://zhuanlan.zhihu.com/p/564819718
简单的粒子群算法案例
求函数f（x，y）=3cos（xy）+x+y2的最小值，其中x的取值范围为［-4，4］，y的取值范围为［-4，4］
"""
"""
如需要移植
需要修改Z的公式，绘制三维立体图
修改超参数
修改适应度函数
"""

import warnings
warnings.filterwarnings("ignore")
# 首先绘制这个函数的三维图像
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-4, 4, 0.01)
Y = np.arange(-4, 4, 0.01)
# meshgrid函数就是用两个坐标轴上的点在平面上画网格(当然这里传入的参数是两个的时候)。
# 当然我们可以指定多个参数，比如三个参数，那么我们的就可以用三个一维的坐标轴上的点在三维平面上画网格。
x, y = np.meshgrid(X, Y)
Z = 3*np.cos(x * y) + x + y**2
# 作图
fig = plt.figure()
ax3 = plt.axes(projection="3d")
ax3.plot_surface(x, y, Z, cmap="rainbow")
# ax3.contour(x ,y ,Z ,zdim = "z" ,offset=-2 ,cmap = "rainbow")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置字体和设置负号
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False

# 初始化种群，群体规模，每个粒子的速度和规模
N = 100 # 种群数目
D = 1 # 维度
T = 200 # 最大迭代次数
c1 = c2 = 1.5 # 个体学习因子与群体学习因子
w_max = 0.8 # 权重系数最大值
w_min = 0.4 # 权重系数最小值
x_max = 450 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
x_min = 50 # 同上
v_max = 1 # 每个维度粒子的最大速度
v_min = -1 # 每个维度粒子的最小速度


# 定义适应度函数
def func(x):
    return 64148734724-(x[0]*(x[0]-30)*(451-x[0])+19)*200000000


# 初始化种群个体
x = np.random.rand(N, D) * (x_max - x_min) + x_min # 初始化每个粒子的位置
v = np.random.rand(N, D) * (v_max - v_min) + v_min # 初始化每个粒子的速度

# 初始化个体最优位置和最优值
p = x # 用来存储每一个粒子的历史最优位置
p_best = np.ones((N, 1))  # 每行存储的是最优值,初始化都为1
for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
    p_best[i] = func(x[i, :])

# 初始化全局最优位置和全局最优值
g_best = float("inf") # 设置真的全局最优值
gb = np.ones(T) # 用于记录每一次迭代的全局最优值
x_best = np.ones(D) # 用于存储最优粒子的取值

# 按照公式依次迭代直到满足精度或者迭代次数
for i in range(T):
    for j in range(N):
        # 个更新个体最优值和全局最优值
        if p_best[j] > func(x[j,:]):
            p_best[j] = func(x[j,:])
            p[j,:] = x[j,:].copy()
        # p_best[j] = func(x[j,:]) if func(x[j,:]) < p_best[j] else p_best[j]
        # 更新全局最优值
        if g_best > p_best[j]:
            g_best = p_best[j]
            x_best = x[j,:].copy()   # 一定要加copy，否则后面x[j,:]更新也会将x_best更新
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        # 更新位置和速度
        v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
        x[j, :] = x[j, :] + v[j, :]
        # 边界条件处理
        for ii in range(D):
            if (v[j, ii] > v_max) or (v[j, ii] < v_min):
                v[j, ii] = v_min + np.random.rand(1) * (v_max - v_min)
            if (x[j, ii] > x_max) or (x[j, ii] < x_min):
                x[j, ii] = x_min + np.random.rand(1) * (x_max - x_min)
    # 记录历代全局最优值
    gb[i] = g_best
print("最优值为", gb[T - 1], "最优位置为", x_best)
plt.plot(range(T),gb)
plt.xlabel("迭代次数")
plt.ylabel("适应度值")
plt.title("适应度进化曲线")
plt.show()

