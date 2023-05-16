#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/11/23 6:46 PM
# @Author  : charles_shen
# @File    : GA_2.py
# @Software: PyCharm
import random

class Item:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
    def __str__(self):
        return f'Item({self.width}, {self.height}, {self.depth})'

class Box:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.items = []
    def __str__(self):
        return f'Box({self.width}, {self.height}, {self.depth})'

class Individual:
    def __init__(self, items, boxes):
        self.items = items
        self.boxes = boxes
        self.fitness = 0
        self.volume = 0

    def calculate_fitness(self):
        self.fitness = -self.volume  #要最小化箱子使用量
        for box in self.boxes:
            item_volume = sum([item.width * item.height * item.depth for item in box.items])
            if item_volume <= box.width * box.height * box.depth:
                # 如果该盒子中没有任何物品或者所有物品可以全部放入该盒子中
                self.volume += item_volume
            else:
                # 盒子不够大，惩罚个体
                self.fitness -= (item_volume - box.width * box.height * box.depth) ** 2

    def __str__(self):
        return f'Individual(Volume: {self.volume}, Fitness: {self.fitness})'

def create_individual(items, boxes):
    individual = Individual(items, boxes)
    # 随机分配物品到盒子里面
    for item in items:
        added = False
        for box in boxes:
            if (item.width <= box.width and item.height <= box.height and item.depth <= box.depth):
                box.items.append(item)
                added = True
                break
        if not added:
            # 创建新的盒子来存放物品
            new_box = Box(item.width, item.height, item.depth)
            new_box.items.append(item)
            boxes.append(new_box)
    # 随机排序盒子中的物品
    for box in boxes:
        random.shuffle(box.items)
    # 将盒子列表随机排序，以增加遗传算法的多样性
    random.shuffle(boxes)
    individual.boxes = boxes
    return individual

def crossover(parent_1, parent_2):
    # 随机选择交叉点
    num_boxes = len(parent_1.boxes)
    crossover_point = random.randint(1, num_boxes - 1)
    child_1_boxes = parent_1.boxes[:crossover_point] + parent_2.boxes[crossover_point:]
    child_2_boxes = parent_2.boxes[:crossover_point] + parent_1.boxes[crossover_point:]
    # 创建新的个体并返回
    child_1 = Individual(parent_1.items, child_1_boxes)
    child_2 = Individual(parent_2.items, child_2_boxes)
    return child_1, child_2

def mutate(individual, mutation_rate):
    for box in individual.boxes:
        if random.random() < mutation_rate:
            # 将物品从一个盒子移动到另一个盒子
            source_box = box
            target_box_index = random.randint(0, len(individual.boxes) - 1)
            target_box = individual.boxes[target_box_index]
            if target_box.index == source_box.index or \
               sum([item.width * item.height * item.depth for item in target_box.items]) + \
               sum([item.width * item.height * item.depth for item in source_box.items]) > \
               target_box.width * target_box.height * target_box.depth:
                # 如果目标盒子与源箱相同，或者目标盒子不足以容纳源箱的所有物品，则不操作。
                continue
            item = random.choice(source_box.items)
            source_box.items.remove(item)
            target_box.items.append(item)

def select_parents(population):
    # 按适应度函数的值升序排序个体列表
    population.sort(key=lambda ind: ind.fitness, reverse=False)
    # 选择前10%的个体作为父母
    num_parents = int(len(population) * 0.1)
    parents = population[:num_parents]
    return parents

def generate_population(num_individuals, items, max_box_size):
    population = []
    for i in range(num_individuals):
        boxes = [Box(max_box_size, max_box_size, max_box_size)]  # 最初只有一个盒子
        individual = create_individual(items, boxes)
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
items = [Item(2, 3, 4), Item(3, 4, 5), Item(4, 5, 6), Item(5, 6, 7)]
max_box_size = 10
num_individuals = 100
mutation_rate = 0.1
num_generations = 50

population = generate_population(num_individuals, items, max_box_size)
for i in range(num_generations):
    population = evolve(population, mutation_rate)
    best_individual = min(population, key=lambda ind: ind.fitness)
    print(f'Generation {i}, Best individual: {best_individual}. Num boxes: {len(best_individual.boxes)}')
