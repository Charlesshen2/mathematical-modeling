#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 3:32 PM
# @Author  : charles_shen
# @File    : simple_GA.py
# @Software: PyCharm
"""
一个封装了7种启发式算法的 Python 代码库
（差分进化算法、遗传算法、粒子群算法、模拟退火算法、蚁群算法、鱼群算法、免疫优化算法）
文章出处：https://zhuanlan.zhihu.com/p/192488077
"""

"""
本程序展示了sko包中GA算法的实例，对其中的参数做详细的解释。
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sko.GA import GA
import matplotlib.pyplot as plt
import pandas as pd


# def schaffer(p):
#     '''
#     This function has plenty of local minimum, with strong shocks
#     global minimum at (0,0) with value 0
#     '''
#     x1, x2 = p
#     x = np.square(x1) + np.square(x2)
#     return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)

def schaffer(p):
    return 3 * np.cos(p[0] * p[1]) + p[0] + p[1] ** 2

#画出该方程的三维立体图

X = np.arange(-4, 4, 0.01)
Y = np.arange(-4, 4, 0.01)
# meshgrid函数就是用两个坐标轴上的点在平面上画网格(当然这里传入的参数是两个的时候)。
# 当然我们可以指定多个参数，比如三个参数，那么我们的就可以用三个一维的坐标轴上的点在三维平面上画网格。
x, y = np.meshgrid(X, Y)
swt = x,y
Z = schaffer(swt)
# 作图
fig = plt.figure(figsize=(10, 10))
ax3 = plt.axes(projection="3d")
ax3.plot_surface(x, y, Z, cmap="rainbow")
# ax3.contour(x ,y ,Z ,zdim = "z" ,offset=-2 ,cmap = "rainbow")
plt.show()



"""
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func 每个变量的精度
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation 突变概率
        
"""

# 运行遗传算法
ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, lb=[-4, -4], ub=[4, 4], precision=1e-7)

best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

#画图


Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()

