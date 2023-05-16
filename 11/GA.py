#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/11/23 6:40 PM
# @Author  : charles_shen
# @File    : GA.py
# @Software: PyCharm
import random

class Box:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
    def __str__(self):
        return f'Box({self.width}, {self.height}, {self.depth})'

class Individual:
    def __init__(self, boxes, max_box_size, num_boxes):
        self.boxes = boxes
        self.max_box_size = max_box_size
        self.num_boxes = num_boxes
        self.fitness = 0
        self.volume = 0

    def calculate_fitness(self):
        self.fitness = -self.volume  #要最小化箱子使用量
        for box in self.boxes:
            if box.width > self.max_box_size or \
               box.height > self.max_box_size or \
               box.depth > self.max_box_size:
                self.fitness -= self.num_boxes * self.max_box_size ** 3  # 惩罚超过最大尺寸的箱子
                continue
            # 计算总体积
            self.volume += box.width * box.height * box.depth

    def __str__(self):
        return f'Individual(Volume: {self.volume}, Fitness: {self.fitness})'

def create_individual(boxes, max_box_size):
    num_boxes = len(boxes)
    individual = Individual(boxes, max_box_size, num_boxes)
    # 随机分配箱子到盒子里面
    boxes_in_boxes = [[]]
    for box in boxes:
        added = False
        for box_group in boxes_in_boxes:
            box_group_volume = sum([b.width * b.height * b.depth for b in box_group])
            if (box_group_volume + box.width * box.height * box.depth) <= max_box_size ** 3:
                box_group.append(box)
                added = True
                break
        if not added:
            boxes_in_boxes.append([box])
    # 随机排序盒子中的箱子
    for i in range(len(boxes_in_boxes)):
        random.shuffle(boxes_in_boxes[i])
    # 将盒子列表随机排序，以增加遗传算法的多样性
    random.shuffle(boxes_in_boxes)
    individual.boxes = sum(boxes_in_boxes, [])  # 把盒子合并到一个列表中
    return individual

def crossover(parent_1, parent_2):
    num_boxes = len(parent_1.boxes)
    # 随机选择交叉点
    crossover_point = random.randint(1, num_boxes - 1)
    child_1_boxes = parent_1.boxes[:crossover_point] + parent_2.boxes[crossover_point:]
    child_2_boxes = parent_2.boxes[:crossover_point] + parent_1.boxes[crossover_point:]
    # 创建新的个体并返回
    child_1 = Individual(child_1_boxes, parent_1.max_box_size, num_boxes)
    child_2 = Individual(child_2_boxes, parent_1.max_box_size, num_boxes)
    return child_1, child_2

def mutate(individual, mutation_rate):
    for i in range(len(individual.boxes)):
        if random.random() < mutation_rate:
            # 移动箱子到不同的盒子
            target_box_index = random.randint(0, individual.num_boxes - 1)
            target_box = individual.boxes[target_box_index]
            source_box = individual.boxes[i]
            if source_box.width > target_box.width or \
               source_box.height > target_box.height or \
               source_box.depth > target_box.depth:
                # 如果移动箱子后目标盒子不能容纳该箱子，那么不做任何操作
                continue
            individual.boxes.remove(source_box)
            individual.boxes.insert(target_box_index, source_box)

def select_parents(population):
    # 按适应度函数的值升序排序个体列表
    population.sort(key=lambda ind: ind.fitness, reverse=False)
    # 选择前10%的个体作为父母
    num_parents = int(len(population) * 0.1)
    parents = population[:num_parents]
    return parents

def generate_population(num_individuals, boxes, max_box_size):
    population = []
    for i in range(num_individuals):
        individual = create_individual(boxes, max_box_size)
        individual.calculate_fitness()
        population.append(individual)
    return population

def evolve(population, mutation_rate):
    parents = select_parents(population)
    num_children = len(population) - len(parents)
    children = []
    while len(children) < num_children:
        parent_1 = random.choice(parents)
        parent_2 = random.choice(parents)
        if parent_1 == parent_2:
            continue
        child_1, child_2 = crossover(parent_1, parent_2)
        mutate(child_1, mutation_rate)
        mutate(child_2, mutation_rate)
        child_1.calculate_fitness()
        child_2.calculate_fitness()
        children.append(child_1)
        children.append(child_2)
    # 合并父代和子代，返回新一代群体
    new_population = parents + children
    return new_population

# 示例用法
boxes = [Box(2, 3, 4), Box(3, 4, 5), Box(4, 5, 6), Box(5, 6, 7)]
max_box_size = 10
num_individuals = 100
mutation_rate = 0.1
num_generations = 50

population = generate_population(num_individuals, boxes, max_box_size)
for i in range(num_generations):
    population = evolve(population, mutation_rate)
    best_individual = min(population, key=lambda ind: ind.fitness)
    print(f'Generation {i}, Best individual: {best_individual}')
