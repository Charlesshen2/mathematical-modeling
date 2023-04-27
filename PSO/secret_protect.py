#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/27/23 3:05 PM
# @Author  : charles_shen
# @File    : secret_protect.py
# @Software: PyCharm
"""
根据提供的数据，实现隐私的二重保护。

初步选用粒子群法作为基础算法，在此基础上对问题进行数理化
编写函数检查是否实现二重隐私保护
"""



# 需要实现将特定的位置变为-1
import numpy as np


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



if __name__ == "__main__":
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    b = np.array([[1, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0]])

    out = change(b, a)
    d,c = check_pro(2, out)
    print(d, c)








