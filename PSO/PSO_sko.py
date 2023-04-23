#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 5:03 PM
# @Author  : charles_shen
# @File    : PSO_sko.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from sko.PSO import PSO

"""
遗传算法参数：
func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
"""

def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)


plt.plot(pso.gbest_y_hist)
print(pso.gbest_y_hist)
plt.show()