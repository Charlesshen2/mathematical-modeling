#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/1/23 10:12 PM
# @Author  : charles_shen
# @File    : 1.py
# @Software: PyCharm
from cvxpy import *
# Create two scalar optimization variables.
# 在CVXPY中变量有标量(只有数值大小)，向量，矩阵。
# 在CVXPY中有常量(见下文的Parameter)

x = Variable() //定义变量x,定义变量y。两个都是标量
y = Variable()
# Create two constraints.
//定义两个约束式
constraints = [x + y == 1,
              x - y >= 1]
//优化的目标函数
obj = Minimize(square(x - y))
//把目标函数与约束传进Problem函数中
prob = Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print "status:", prob.status
print "optimal value", prob.value //最优值
print "optimal var", x.value, y.value //x与y的解
status: optimal
optimal value 0.999999999761
optimal var 1.00000000001 -1.19961841702e-11
//状态域被赋予'optimal'，说明这个问题被成功解决。
//最优值是针对所有满足约束条件的变量x,y中目标函数的最小值
//prob.solve()返回最优值，同时更新prob.status,prob.value,和所有变量的值。