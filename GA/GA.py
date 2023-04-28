#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/28/23 2:18 PM
# @Author  : charles_shen
# @File    : GA.py
# @Software: PyCharm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# 实现数组中1的个数计数


def count_ones(x):
    count = 0
    for i in range(len(x)):
       if int(x[i]) == 1:
           count = count + 1
    return count


# 需要实现将特定的位置变为-1
def change(x, y):
    for i in range(len(x)):
        if x[i] == 1:
            y[i] = -1
    return y


# 编写函数 输入x和对应的多维系数，再检查是否符合二重隐私保护
# z表示需要实现的多重保护方案
def check_pro(z, x):
    flag = 1  # 记录是否完成保护
    no_pro = 0  # 记录未完成保护的数目
    while len(x) > 1:
        same = 1
        index_lo = np.ones(1) - 1
        for i in range(1, len(x)):
            # 记录相同数组的位置

            if (x[i] == x[0]).all():
                # print(x[i])
                same = same + 1
                index_lo = np.append(index_lo, i)
                # print(index_lo)
        # 删除已经形成多重保护的数组
        # print(same)
        # print(index_lo)
        for i in index_lo[::-1]:
            # print(i)
            x = np.delete(x, int(i), 0)
        if same < z:
            flag = 0
            no_pro = no_pro + len(index_lo)
            # print(index_lo)
            # print(same)
            same = 0
    if len(x) == 1:
        no_pro = no_pro + 1
        flag = 0
    return flag, no_pro

# 导入数据
all_data = pd.read_excel('/Users/shenfeiyang/Documents/GitHub/mathematical-modeling/data/B_data.xlsx', sheet_name=None)
data_1 = all_data['二元1']
data_2 = all_data['二元2']
data_3 = all_data['多元1']
data_4 = all_data['多元2']

# 将data1转化为numpy形式
data_1 = np.asarray(data_1)
row_a = len(data_1)
col_a = len(data_1[0])
print(data_1)
print(row_a, col_a)
# 将二维数组转化为一维数组
data_1 = data_1 .flatten()

# 设置适应度函数
def F(x):
    k = change(x, data_1)
    d, c = check_pro(2, k.reshape(row_a, col_a))
    if d == 1:
        return count_ones(x)
    else:
        return float("inf")

DNA_SIZE = col_a
POP_SIZE = 80
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 100
X_BOUND = [-2.048, 2.048]
Y_BOUND = [-2.048, 2.048]




def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return pred
    # return pred - np.min(pred)+1e-3  # 求最大值时的适应度
    # return np.max(pred) - pred + 1e-3  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
    y_pop = pop[:, DNA_SIZE:]  # 后DNA_SIZE位表示Y

    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    print(F(x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff()
    plot_3d(ax)