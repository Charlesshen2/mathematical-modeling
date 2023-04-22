#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23 5:05 PM
# @Author  : charles_shen
# @File    : simple_SA.py
# @Software: PyCharm

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2

from sko.SA import SA

sa = SA(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y', best_y)

import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
plt.show()
#另外，scikit-opt 还提供了三种模拟退火流派: Fast, Boltzmann, Cauchy.