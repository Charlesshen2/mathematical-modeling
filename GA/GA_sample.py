#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/28/23 2:51 PM
# @Author  : charles_shen
# @File    : GA_sample.py
# @Software: PyCharm
# procedure:GA for Knapsack Problem(KP)
# input: KP data set, GA parameters
# output: the best solution
# begin
# 	t<-- 0	                       // t:遗传代数
# 	initialize P(t) by encoding routine;         //P(t):染色体种群
# 	fitness eval(P) by decoding routine;
# 	while (not termination condition) do
# 		crossover P(t) to yield C(t);     //C(t):offspring
# 		mutation   P(t) to yield C(t);
# 		fitness eval(C) by decoding routine;
# 		select P(t+1) from P(t) and C(t);
# 		t<--t+1;
# 	end
#     output:the best solution;
# end
import numpy as np
import matplotlib.pyplot as plt
#初始化种群
def init(popsize,n):
    population=[]
    for i in range(popsize):
        pop=''
        for j in range(n):
            pop=pop+str(np.random.randint(0,2))
        population.append(pop)
    return population
#解码1
def decode1(x,n,w,c,W):
    s=[]#储存被选择物体的下标集合
    g=0
    f=0
    for i in range(n):
        if (x[i] == '1'):
            if g+w[i] <= W:
                g = g+w[i]
                f = f+c[i]
                s.append(i)
            else:
                break
    return f,s

#适应度函数1
def fitnessfun1(population,n,w,c,W):
    value=[]
    ss=[]
    for i in range(len(population)):
        [f,s]= decode1(population[i],n,w,c,W)
        value.append(f)
        ss.append(s)
    return value,ss
#解码2
def decode2(x,n,w,c):
    s=[]#储存被选择物体的下标集合
    g=0
    f=0
    for i in range(n):
        if (x[i] == '1'):
            g = g+w[i]
            f = f+c[i]
            s.append(i)
    return g,f,s

#适应度函数2
def fitnessfun2(population,n,w,c,W,M):
    value=[]
    ss=[]
    for i in range(len(population)):
        [g,f,s]= decode2(population[i],n,w,c)
        if g>W:
            f = -M*f#惩罚
        value.append(f)
        ss.append(s)
    minvalue=min(value)
    value=[(i-minvalue+1) for i in value]
    return value,ss
#轮盘赌选择
def roulettewheel(population,value,pop_num):
    fitness_sum=[]
    value_sum=sum(value)
    fitness=[i/value_sum for i in value]
    for i in range(len(population)):##
        if i==0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i-1]+fitness[i])
    population_new=[]
    for j in range(pop_num):###
        r=np.random.uniform(0,1)
        for i in range(len(fitness_sum)):###
            if i==0:
                if r>=0 and r<=fitness_sum[i]:
                    population_new.append(population[i])
            else:
                if r>=fitness_sum[i-1] and r<=fitness_sum[i]:
                    population_new.append(population[i])
    return population_new
#两点交叉
def crossover(population_new,pc,ncross):
    a=int(len(population_new)/2)
    parents_one=population_new[:a]
    parents_two=population_new[a:]
    np.random.shuffle(parents_one)
    np.random.shuffle(parents_two)
    offspring=[]
    for i in range(a):
        r=np.random.uniform(0,1)
        if r<=pc:
            point1=np.random.randint(0,(len(parents_one[i])-1))
            point2=np.random.randint(point1,len(parents_one[i]))
            off_one=parents_one[i][:point1]+parents_two[i][point1:point2]+parents_one[i][point2:]
            off_two=parents_two[i][:point1]+parents_one[i][point1:point2]+parents_two[i][point2:]
            ncross = ncross+1
        else:
            off_one=parents_one[i]
            off_two=parents_two[i]
        offspring.append(off_one)
        offspring.append(off_two)
    return offspring
#单点变异1
def mutation1(offspring,pm,nmut):
    for i in range(len(offspring)):
        r=np.random.uniform(0,1)
        if r<=pm:
            point=np.random.randint(0,len(offspring[i]))
            if point==0:
                if offspring[i][point]=='1':
                    offspring[i]='0'+offspring[i][1:]
                else:
                    offspring[i]='1'+offspring[i][1:]
            else:
                if offspring[i][point]=='1':
                    offspring[i]=offspring[i][:(point-1)]+'0'+offspring[i][point:]
                else:
                    offspring[i]=offspring[i][:(point-1)]+'1'+offspring[i][point:]
            nmut = nmut+1
    return offspring
#单点变异2
def mutation2(offspring,pm,nmut):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            r=np.random.uniform(0,1)
            if r<=pm:
                if j==0:
                    if offspring[i][j]=='1':
                        offspring[i]='0'+offspring[i][1:]
                    else:
                        offspring[i]='1'+offspring[i][1:]
                else:
                    if offspring[i][j]=='1':
                        offspring[i]=offspring[i][:(j-1)]+'0'+offspring[i][j:]
                    else:
                        offspring[i]=offspring[i][:(j-1)]+'1'+offspring[i][j:]
                nmut = nmut+1
    return offspring

#主程序----------------------------------------------------------------------------------------------------------------------------------
#参数设置-----------------------------------------------------------------------
gen=1000#迭代次数
pc=0.25#交叉概率
pm=0.02#变异概率
popsize=10#种群大小
n = 20#物品数,即染色体长度n
w = [2,5,18,3,2,5,10,4,8,12,5,10,7,15,11,2,8,10,5,9]#每个物品的重量列表
c = [5,10,12,4,3,11,13,10,7,15,8,19,1,17,12,9,15,20,2,6]#每个物品的代价列表
W = 40#背包容量
M = 5#惩罚值
fun = 1#1-第一种解码方式，2-第二种解码方式（惩罚项）
#初始化-------------------------------------------------------------------------
#初始化种群（编码）
population=init(popsize,n)
#适应度评价（解码）
if fun==1:
    value,s = fitnessfun1(population,n,w,c,W)
else:
    value,s = fitnessfun2(population,n,w,c,W,M)
#初始化交叉个数
ncross=0
#初始化变异个数
nmut=0
#储存每代种群的最优值及其对应的个体
t=[]
best_ind=[]
last=[]#储存最后一代个体的适应度值
realvalue=[]#储存最后一代解码后的值
#循环---------------------------------------------------------------------------
for i in range(gen):
    print("迭代次数：")
    print(i)
    #交叉
    offspring_c=crossover(population,pc,ncross)
    #变异
    #offspring_m=mutation1(offspring,pm,nmut)
    offspring_m=mutation2(offspring_c,pm,nmut)
    mixpopulation=population+offspring_m
    #适应度函数计算
    if fun==1:
        value,s = fitnessfun1(mixpopulation,n,w,c,W)
    else:
        value,s = fitnessfun2(mixpopulation,n,w,c,W,M)
    #轮盘赌选择
    population=roulettewheel(mixpopulation,value,popsize)
    #储存当代的最优解
    result=[]
    if i==gen-1:
        if fun==1:
            value1,s1 = fitnessfun1(population,n,w,c,W)
            realvalue=s1
            result=value1
            last=value1
        else:
            for j in range(len(population)):
                g1,f1,s1 = decode2(population[j],n,w,c)
                result.append(f1)
                realvalue.append(s1)
            last=result
    else:
        if fun==1:
            value1,s1 = fitnessfun1(population,n,w,c,W)
            result=value1
        else:
            for j in range(len(population)):
                g1,f1,s1 = decode2(population[j],n,w,c)
                result.append(f1)
    maxre=max(result)
    h=result.index(max(result))
    #将每代的最优解加入结果种群
    t.append(maxre)
    best_ind.append(population[h])

#输出结果-----------------------------------------------------------------------
if fun==1:
    best_value=max(t)
    hh=t.index(max(t))
    f2,s2 = decode1(best_ind[hh],n,w,c,W)
    print("最优组合为：")
    print(s2)
    print("最优解为：")
    print(f2)
    print("最优解出现的代数：")
    print(hh)
    #画出收敛曲线
    plt.plot(t)
    plt.title('The curve of the optimal function value of each generation with the number of iterations',color='#123456')
    plt.xlabel('the number of iterations')
    plt.ylabel('the optimal function value of each generation')
else:
    best_value=max(result)
    hh=result.index(max(result))
    s2 = realvalue[hh]
    print("最优组合为：")
    print(s2)
    print("最优解为：")
    print(f2)
import numpy as np
import matplotlib.pyplot as plt

#主程序----------------------------------------------------------------------------------------------------------------------------------
#参数设置-----------------------------------------------------------------------
gen=1000#迭代次数
pc=0.25#交叉概率
pm=0.02#变异概率
popsize=20#种群大小
n = 20#物品数,即染色体长度n
w = [2,5,18,3,2,5,10,4,8,12,5,10,7,15,11,2,8,10,5,9]#每个物品的重量列表
c = [5,10,12,4,3,11,13,10,7,15,8,19,1,17,12,9,15,20,2,6]#每个物品的代价列表
W = 40#背包容量
M = 5#惩罚值
fun = 1#1-第一种解码方式，2-第二种解码方式（惩罚项）
#初始化-------------------------------------------------------------------------
#初始化种群（编码）
population=init(popsize,n)
#适应度评价（解码）
if fun==1:
    value,s = fitnessfun1(population,n,w,c,W)
else:
    value,s = fitnessfun2(population,n,w,c,W,M)
#初始化交叉个数
ncross=0
#初始化变异个数
nmut=0
#储存每代种群的最优值及其对应的个体
t=[]
best_ind=[]
last=[]#储存最后一代个体的适应度值
realvalue=[]#储存最后一代解码后的值
#循环---------------------------------------------------------------------------
for i in range(gen):
    print("迭代次数：")
    print(i)
    #适应度函数计算
    if fun==1:
        value,s = fitnessfun1(population,n,w,c,W)
    else:
        value,s = fitnessfun2(population,n,w,c,W,M)
    #轮盘赌选择
    population=roulettewheel(population,value,popsize)
    #交叉
    offspring_c=crossover(population,pc,ncross)
    #变异
    #offspring_m=mutation1(offspring,pm,nmut)
    offspring_m=mutation2(offspring_c,pm,nmut)
    population=offspring_m

    #储存当代的最优解
    result=[]
    if i==gen-1:
        if fun==1:
            value1,s1 = fitnessfun1(population,n,w,c,W)
            realvalue=s1
            result=value1
            last=value1
        else:
            for j in range(len(population)):
                g1,f1,s1 = decode2(population[j],n,w,c)
                result.append(f1)
                realvalue.append(s1)
            last=result
    else:
        if fun==1:
            value1,s1 = fitnessfun1(population,n,w,c,W)
            result=value1
        else:
            for j in range(len(population)):
                g1,f1,s1 = decode2(population[j],n,w,c)
                result.append(f1)
    maxre=max(result)
    h=result.index(max(result))
    #将每代的最优解加入结果种群
    t.append(maxre)
    best_ind.append(population[h])

#输出结果-----------------------------------------------------------------------
if fun==1:
    best_value=max(t)
    hh=t.index(max(t))
    f2,s2 = decode1(best_ind[hh],n,w,c,W)
    print("最有组合为：")
    print(s2)
    print("最优解为：")
    print(f2)
    print("最优解出现的代数：")
    print(hh)
    #画出收敛曲线
    plt.plot(t)
    plt.title('The curve of the optimal function value of each generation with the number of iterations',color='#123456')
    plt.xlabel('the number of iterations')
    plt.ylabel('the optimal function value of each generation')
else:
    best_value=max(result)
    hh=result.index(max(result))
    s2 = realvalue[hh]
    print("最优组合为：")
    print(s2)
    print("最优解为：")
    print(f2)
