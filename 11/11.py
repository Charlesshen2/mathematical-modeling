#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/11/23 10:52 AM
# @Author  : charles_shen
# @File    : 11.py
# @Software: PyCharm
from vtk import *
import vtk
import random as rd
import functools
import random

# 数据生成，输入为箱子种类数，箱子最大长、最小长、最大宽、最小宽、商品个数、商品最大长、最小长、最大宽、最小宽
# 输出为生成的箱子列表和商品列表
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkWindowToImageFilter


def data_generate_3d(nums_box, max_edge_box, min_edge_box, nums_good, max_edge_good, min_edge_good):

    #随机生成箱子
    boxs = []
    for i in range(nums_box):
        boxs.append([int((max_edge_box-min_edge_box) * rd.random() + min_edge_box),
                     int((max_edge_box-min_edge_box) * rd.random() + min_edge_box),
                     int((max_edge_box-min_edge_box) * rd.random() + min_edge_box)])

    #随机生成商品
    goods = []
    for i in range(nums_good):
        goods.append([int((max_edge_good-min_edge_good) * rd.random() + min_edge_good),
                      int((max_edge_good-min_edge_good) * rd.random() + min_edge_good),
                      int((max_edge_good-min_edge_good) * rd.random() + min_edge_good)])

    return boxs, goods

# 二维商品排序用
def cmp_2d(x,y):
    if x[0] < y[0]:
        return -1
    elif x[0] > y[0]:
        return 1
    elif x[1] < y[1]:
        return -1
    elif x[1] > y[1]:
        return 1
    else:
        return 0


# 三维商品排序用
def cmp_3d(x, y):
    if x[0] < y[0]:
        return -1
    elif x[0] > y[0]:
        return 1
    elif x[1] < y[1]:
        return -1
    elif x[1] > y[1]:
        return 1
    elif x[2] < y[2]:
        return -1
    elif x[2] > y[2]:
        return 1
    else:
        return 0


# 检验是否每个商品都能有一个箱子放下它
def check_3d(boxs, goods):
    for good in goods:
        can_put = False
        for box in boxs:
            if good[0] <= box[0] and good[1] <= box[1] and good[2] <= box[2]:
                can_put = True
                break
        if not can_put:
            print(good,"太大，无合适箱子")
            return False
    return True


def can_put_3d(l, w, h, goods):
    L = max(l, w, h)
    H = min(l, w, h)
    W = l + w + h - L - H

    for good in goods:
        lg, wg, hg = good[0], good[1], good[2]
        # lg = max(good[0], good[1], good[2])
        # hg = min(good[0], good[1], good[2])
        # wg = good[0] + good[1] + good[2] - lg - hg
        if lg > L or wg > W or hg > H:
            return False
    return True

# 先以w为限制码垛，再以l为限制码垛
# 输入为长、宽、商品集，输出为箱子个数
def packing_simple(l, w, h, goods):

    # 先检查是否每一个商品在此规则下都能放下
    if not can_put_3d(l, w, h, goods):
        return -1

    #以h为限制码垛成条，商品排序，大的放前面
    goods1 = []
    for good in goods:
        if good[0] <= l and good[1] <= w and good[2] <= h:
            goods1.append([good[0], good[1], good[2]])
        elif good[0] <= l and good[2] <= w and good[1] <= h:
            goods1.append([good[0], good[2], good[1]])
        elif good[1] <= l and good[0] <= w and good[2] <= h:
            goods1.append([good[1], good[0], good[2]])
        elif good[1] <= l and good[2] <= w and good[0] <= h:
            goods1.append([good[1], good[2], good[0]])
        elif good[2] <= l and good[0] <= w and good[1] <= h:
            goods1.append([good[2], good[0], good[1]])
        else:
            goods1.append([good[2], good[1], good[0]])

    goods1 = sorted(goods1, key=functools.cmp_to_key(cmp_3d), reverse=True)

    strips = []
    goods1_used = [0 for i in range(len(goods1))]

    while sum(goods1_used) < len(goods1_used):
        l_used = 0
        w_used = 0
        h_used = 0
        for i in range(len(goods1_used)):
            if goods1_used[i] == 0 and h_used + goods1[i][2] <= h:
                l_used = max(l_used, goods1[i][0])
                w_used = max(w_used, goods1[i][1])
                h_used += goods1[i][2]
                goods1_used[i] = 1
        strips.append([l_used, w_used])

    strips = sorted(strips, key=functools.cmp_to_key(cmp_2d), reverse=True)

    #以w为限制码垛成层
    levels = []
    strip_used = [0 for i in range(len(strips))]

    while sum(strip_used) < len(strip_used):
        l_used = 0
        w_used = 0
        for i in range(len(strips)):
            if strip_used[i] == 0 and w_used + strips[i][1] <= w:
                l_used = max(l_used, strips[i][0])
                w_used += strips[i][1]
                strip_used[i] = 1
        levels.append(l_used)

    #再以l为限制码垛
    levels = sorted(levels, reverse=True)
    L_box_unused = [l]

    for level in levels:
        flag = -1
        for i in range(len(L_box_unused)):
            if L_box_unused[i] >= level:
                if flag == -1:
                    flag = i
                elif L_box_unused[i] < L_box_unused[flag]:
                    flag = i
        if flag == -1:
            L_box_unused.append(l - level)
        else:
            L_box_unused[flag] -= level

    return len(L_box_unused)


# 选择合适的主箱子
def box_choose_3d(boxs, nums_simplePacking_1, nums_simplePacking_2,nums_simplePacking_3, nums_simplePacking_4,nums_simplePacking_5, nums_simplePacking_6):
    l = -1
    w = -1
    h = -1
    nums = -1
    for i in range(len(boxs)):
        if nums_simplePacking_1[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_1[i]) or (nums != -1 and nums == nums_simplePacking_1[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][0]
                w = boxs[i][1]
                h = boxs[i][2]
                nums = nums_simplePacking_1[i]
        if nums_simplePacking_2[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_2[i]) or (nums != -1 and nums == nums_simplePacking_2[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][0]
                w = boxs[i][2]
                h = boxs[i][1]
                nums = nums_simplePacking_2[i]
        if nums_simplePacking_3[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_3[i]) or (nums != -1 and nums == nums_simplePacking_3[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][1]
                w = boxs[i][0]
                h = boxs[i][2]
                nums = nums_simplePacking_3[i]
        if nums_simplePacking_4[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_4[i]) or (nums != -1 and nums == nums_simplePacking_4[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][1]
                w = boxs[i][2]
                h = boxs[i][0]
                nums = nums_simplePacking_4[i]
        if nums_simplePacking_5[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_5[i]) or (nums != -1 and nums == nums_simplePacking_5[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][2]
                w = boxs[i][0]
                h = boxs[i][1]
                nums = nums_simplePacking_5[i]
        if nums_simplePacking_6[i] != -1:
            if nums == -1 or (nums != -1 and nums > nums_simplePacking_6[i]) or (nums != -1 and nums == nums_simplePacking_6[i] and l * w * h > boxs[i][0] * boxs[i][1] * boxs[i][2]):
                l = boxs[i][2]
                w = boxs[i][1]
                h = boxs[i][0]
                nums = nums_simplePacking_6[i]
    return l, w, h


def packing_3d(l, w, h, goods):
    # 先检查是否每一个商品在此规则下都能放下
    if not can_put_3d(l, w, h, goods):
        return -1

    # 以h为限制码垛成条，商品排序，大的放前面
    goods1 = []
    # for good in goods:
    #     if good[3] is not min(good):
    #         good.append(min(good))
    #
    #     goods1.append([good[0], good[1], good[3]])

    for good in goods:
        if good[0] <= l and good[1] <= w and good[2] <= h:
            goods1.append([good[0], good[1], good[2]])
        elif good[0] <= l and good[2] <= w and good[1] <= h:
            goods1.append([good[0], good[2], good[1]])
        elif good[1] <= l and good[0] <= w and good[2] <= h:
            goods1.append([good[1], good[0], good[2]])
        elif good[1] <= l and good[2] <= w and good[0] <= h:
            goods1.append([good[1], good[2], good[0]])
        elif good[2] <= l and good[0] <= w and good[1] <= h:
            goods1.append([good[2], good[0], good[1]])
        else:
            goods1.append([good[2], good[1], good[0]])

    # 排序
    goods1 = sorted(goods1, key=functools.cmp_to_key(cmp_3d), reverse=True)

    strips = []
    strips_goods = []
    goods1_used = [0 for _ in range(len(goods1))]

    # 先把物品堆成一条条顶天柱子
    while sum(goods1_used) < len(goods1_used):
        l_used = 0
        w_used = 0
        h_used = 0
        strip_goods = []
        for i in range(len(goods1_used)):
            if goods1_used[i] == 0 and h_used + goods1[i][2] <= h:
                l_used = max(l_used, goods1[i][0])
                w_used = max(w_used, goods1[i][1])
                strip_goods.append([goods1[i][0], goods1[i][1], goods1[i][2], 0, 0, h_used])
                h_used += goods1[i][2]
                goods1_used[i] = 1
        strips.append([l_used, w_used])
        strips_goods.append(strip_goods)

    strips = sorted(strips, key=functools.cmp_to_key(cmp_2d), reverse=True)

    # 以w为限制码垛成层
    # 排序
    for i in range(len(strips)-1):    # 前面参照柱子
        for j in range(i+1, len(strips)):   # 当前需放置柱子
            # 前面柱子长度小于当前柱子长度 or (前面柱子长度等于当前柱子长度 and 前面柱子宽度小于当前柱子宽度)
            if strips[i][0] < strips[j][0] or (strips[i][0] == strips[j][0] and strips[i][1] < strips[j][1]):
                temp = strips[i]  # 交换前后柱子
                strips[i] = strips[j]
                strips[j] = temp
                temp1 = strips_goods[i]
                strips_goods[i] = strips_goods[j]
                strips_goods[j] = temp1

    levels = []
    levels_goods = []
    strip_used = [0 for i in range(len(strips))]

    # 把柱子按箱子宽度方向排
    while sum(strip_used) < len(strip_used):
        l_used = 0
        w_used = 0
        level_goods = []
        for i in range(len(strips)):
            if strip_used[i] == 0 and w_used + strips[i][1] <= w:
                l_used = max(l_used, strips[i][0])
                for g in strips_goods[i]:
                    level_goods.append([g[0], g[1], g[2], 0, w_used, g[5]])
                w_used += strips[i][1]
                strip_used[i] = 1
        levels.append(l_used)
        levels_goods.append(level_goods)

    # 再以l为限制码垛
    for i in range(len(levels)-1):
        for j in range(i+1, len(levels)):
            if levels[i] < levels[j]:
                temp = levels[i]
                levels[i] = levels[j]
                levels[j] = temp
                temp1 = levels_goods[i]
                levels_goods[i] = levels_goods[j]
                levels_goods[j] = temp1

    L_box_unused = [l]
    L_goods = []
    L_coordinates = []

    L_goods.append([])
    L_coordinates.append([])

    for i in range(len(levels)):
        flag = -1
        for j in range(len(L_box_unused)):
            if L_box_unused[j] >= levels[j]:
                if flag == -1 or (flag != -1 and L_box_unused[j] < L_box_unused[flag]):
                    flag = j
        if flag == -1:
            L_box_unused.append(l - levels[i])
            L_goods.append([levels_goods[i][j][:3] for j in range(len(levels_goods[i]))])
            L_coordinates.append([levels_goods[i]])
        else:
            L_box_unused[flag] -= levels[i]
            L_goods[flag] += [levels_goods[i][j][:3] for j in range(len(levels_goods[i]))]
            if len(L_coordinates[flag]) == 0:
                L_coordinates[flag] += [levels_goods[i]]
            else:
                L_coordinates[flag] += [[[levels_goods[i][j][0], levels_goods[i][j][1], levels_goods[i][j][2], L_coordinates[flag][-1][0][0] + L_coordinates[flag][-1][0][3], levels_goods[i][j][4], levels_goods[i][j][5]] for j in range(len(levels_goods[i]))]]

    L_coordinates_merge = []

    for i in range(len(L_coordinates)):
        L_coordinates_i = []
        for j in range(len(L_coordinates[i])):
            L_coordinates_i += L_coordinates[i][j]
        L_coordinates_merge.append(L_coordinates_i)

    L_box = [[l, w, h] for i in range(len(L_box_unused))]

    return L_box, L_goods, L_coordinates_merge


# 正交二叉树启发式，试每一种箱子装下所有的商品需要的个数，取最少的，再去缩减最后一个箱子
def OBT_3d(boxs, goods):

    # 分别以长宽作为限制，依次码垛成层
    nums_simplePacking_1 = []
    nums_simplePacking_2 = []
    nums_simplePacking_3 = []
    nums_simplePacking_4 = []
    nums_simplePacking_5 = []
    nums_simplePacking_6 = []

    for box in boxs:
        nums_simplePacking_1.append(packing_simple(box[0], box[1], box[2], goods))
        nums_simplePacking_2.append(packing_simple(box[0], box[2], box[1], goods))
        nums_simplePacking_3.append(packing_simple(box[1], box[0], box[2], goods))
        nums_simplePacking_4.append(packing_simple(box[1], box[2], box[0], goods))
        nums_simplePacking_5.append(packing_simple(box[2], box[0], box[1], goods))
        nums_simplePacking_6.append(packing_simple(box[2], box[1], box[0], goods))

    # 找箱子数最少的箱子
    l, w, h = box_choose_3d(boxs, nums_simplePacking_1, nums_simplePacking_2, nums_simplePacking_3,
                            nums_simplePacking_4, nums_simplePacking_5, nums_simplePacking_6)

    # 装载
    L_box, L_goods, L_coordinates = packing_3d(l, w, h, goods)

    return L_box, L_goods, L_coordinates


# 检验结果中的商品集是否和原始的商品集一致
def goods_check(goods, L_goods):

    nums = 0
    for gs in L_goods:
        nums += len(gs)

    if len(goods) == nums:
        return True

    return False


#添加商品图形
def Addcube_3d(ren, coordinate, edge_max, x_re, y_re, z_re):
    cube = vtk.vtkCubeSource()
    cube.SetXLength(coordinate[0]/edge_max)
    cube.SetYLength(coordinate[1]/edge_max)
    cube.SetZLength(coordinate[2]/edge_max)
    cube.Update()

    translation = vtk.vtkTransform()
    translation.Translate((coordinate[3] + coordinate[0]/2.0)/edge_max + x_re, (coordinate[4] + coordinate[1]/2.0)/edge_max + y_re, (coordinate[5] + coordinate[2]/2.0)/edge_max + z_re)
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(cube.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()

    transformedMapper = vtkPolyDataMapper()
    transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
    transformedActor = vtkActor()
    transformedActor.SetMapper(transformedMapper)
    transformedActor.GetProperty().SetColor((rd.uniform(0, 1), rd.uniform(0, 1), rd.uniform(0, 1)))

    ren.AddActor(transformedActor)

def png_save(renWin, name):
    windowToImageFilter = vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(name)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()


# 三维展示，输入为箱子集和商品集，包裹的箱子和商品集一一对应
def show_3d(L_box, L_coordinates):

    nums = len(L_box)
    edge_max = max([max(L_box[i]) for i in range(len(L_box))]) if max([max(L_box[i]) for i in range(len(L_box))]) > 0 else 1

    #预设参数
    gap = 0.01
    CL_p = 1.1
    CW_p = nums + gap * (nums - 1)
    CH_p = 0.01
    gap = 0.25

    x_re = 0
    y_re = 0
    z_re = 0

    #渲染及渲染窗口，并根据捕捉的鼠标事件执行相应的操作
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(1200, 600)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)


    """画容器"""
    for i in range(nums):

        cube = vtk.vtkCubeSource()
        cube.SetXLength(L_box[i][0]/edge_max)
        cube.SetYLength(L_box[i][1]/edge_max)
        cube.SetZLength(L_box[i][2]/edge_max)
        cube.Update()

        translation = vtkTransform()
        translation.Translate(L_box[i][0]/edge_max/2.0 + x_re, L_box[i][1]/edge_max/2.0 + i + gap*i + y_re, L_box[i][2]/edge_max/2.0 + z_re)
        transformFilter = vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(cube.GetOutputPort())
        transformFilter.SetTransform(translation)
        transformFilter.Update()

        transformedMapper = vtkPolyDataMapper()
        transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
        transformedActor = vtkActor()
        transformedActor.SetMapper(transformedMapper)
        transformedActor.GetProperty().SetColor((1, 1, 1))
        transformedActor.GetProperty().SetRepresentationToWireframe()

        ren.AddActor(transformedActor)

    """画托盘"""
    cube = vtk.vtkCubeSource()
    cube.SetXLength(CL_p)
    cube.SetYLength(CW_p)
    cube.SetZLength(CH_p)
    cube.Update()

    translation = vtkTransform()
    translation.Translate(CL_p/2.0 + x_re, CW_p/2.0 + y_re, -CH_p/2.0 + z_re)
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(cube.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()

    transformedMapper = vtkPolyDataMapper()
    transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
    transformedActor = vtkActor()
    transformedActor.SetMapper(transformedMapper)
    transformedActor.GetProperty().SetColor((0.2, 0.4, 0.8))

    ren.AddActor(transformedActor)

    for i in range(len(L_coordinates)):
        for j in range(len(L_coordinates[i])):
            Addcube_3d(ren, L_coordinates[i][j], edge_max, x_re, i + gap*i + y_re, z_re)

    camera = vtk.vtkCamera()
    camera.SetViewUp(0, 0, 1)  # 设置相机的“上”方向
    camera.SetPosition(10, 10, 1)  # 位置：世界坐标系，设置相机位置
    camera.SetFocalPoint(0, 4, 0)
    camera.ComputeViewPlaneNormal()
    ren.SetActiveCamera(camera)

    iren.Initialize()
    renWin.Render()
    # 保存过程
    png_save(renWin, "result_D3.png")
    # 展示
    iren.Start()


def exchange_item(items):  # 第一类邻域选择，随机交换两个物品
    s1, s2 = random.randint(0, len(items) - 1), random.randint(0, len(items) - 1)
    while s1 == s2:
        s2 = random.randint(0, len(goods) - 1)
    items[s1], items[s2], = items[s2], items[s1]
    return items

def exchange_direction(items):  # 第二类邻域选择，随机交换某个物品的方向
    s = random.randint(0, len(items) - 1)
    item = items[s]
    s_1, s_2 = random.randint(0, len(item) - 1), random.randint(0, len(item) - 1)
    while s_1 == s_2:
        s_2 = random.randint(0, len(item) - 1)
    item[s_1], item[s_2], = item[s_2], item[s_1]
    items[s] = item
    return items


def Search(alpha, t_set, goods_se, boxs_se, markovlen):
    # alpha = 0.99
    # t = (1, 100)
    # m = 100
    min_t = t_set[0]
    t = t_set[1]
    L_box, L_goods, L_coordinates = OBT_3d(boxs_se, goods_se)
    show_3d(L_box, L_coordinates)
    print(L_coordinates[3])
    print(L_coordinates[-1])
    print(len(L_goods[-1]))
    goods_se_last = L_goods[-1]
    boxs_se_last = [[25, 60, 60] for _ in range(3)]
    L_box_last, L_goods_last, L_coordinates_last = OBT_3d(boxs_se_last, goods_se_last)
    # show_3d(L_box_last, L_coordinates_last)
    for L_list in L_coordinates_last[0]:
        L_list[3] += 55
    for L_list in L_coordinates_last[1]:
        L_list[3] += 55
    L_coordinates[0].extend(L_coordinates_last[0])
    L_coordinates[1].extend(L_coordinates_last[1])
    L_coordinates.pop()
    print(L_coordinates[0])
    print(L_box, L_coordinates)
    show_3d(L_box, L_coordinates)


if __name__ == "__main__":
    space_ratio = []
    # 生成箱子集和商品集，计算并展示
    # boxs, goods = data_generate_3d(nums_box=2, max_edge_box=20, min_edge_box=10, nums_good=10, max_edge_good=10, min_edge_good=1)
    boxs = [[80, 60, 60] for _ in range(6)]
    # 第一种方案
    goods = [[21, 14, 9] for _ in range(2 + 5 + 7 + 15)]  # 选4号
    goods.extend([[25, 15, 9.5] for _ in range(25 + 25 + 25 + 25 + 25)])  # 选7号
    goods.extend([[27.5, 17.5, 10.5] for _ in range(25 + 25 + 22 + 20 + 10)])  # 选9号
    # 第二种方案
    # goods = [[21, 14, 9] for _ in range(2 + 5 + 7 + 15)]  # 选4号
    # goods.extend([[23.5, 16.5, 9.2] for _ in range(25 + 25 + 25 + 25 + 25)])  # 选5号
    # goods.extend([[27.5, 17.5, 10.5] for _ in range(25 + 25 + 22 + 20 + 10)])  # 选9号
    # 第三种方案
    # goods = [[21, 14, 9] for _ in range(2 + 5 + 7 + 15)]  # 选4号
    # goods.extend([[23.5, 16.5, 9.2] for _ in range(25 + 25 + 25 + 25 + 25 + 25)])  # 选7号
    # goods.extend([[27.5, 17.5, 10.5] for _ in range(25 + 22 + 20 + 10)])  # 选9号

    Search(alpha=0.99, t_set=(0.001, 1), goods_se=goods, boxs_se=boxs, markovlen=1)