import math
import random
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Dijkstra(G, start):
    # 输入是从 0 开始，所以起始点减 1
    start = start - 1
    inf = float('inf')
    node_num = len(G)
    # visited 代表哪些顶点加入过
    visited = [0] * node_num
    # 初始顶点到其余顶点的距离
    dis = {node: G[start][node] for node in range(node_num)}
    # parents 代表最终求出最短路径后，每个顶点的上一个顶点是谁，初始化为 -1，代表无上一个顶点
    parents = {node: -1 for node in range(node_num)}
    # 起始点加入进 visited 数组
    visited[start] = 1
    # 最开始的上一个顶点为初始顶点
    last_point = start

    for i in range(node_num - 1):
        # 求出 dis 中未加入 visited 数组的最短距离和顶点
        min_dis = inf
        for j in range(node_num):
            if visited[j] == 0 and dis[j] < min_dis:
                min_dis = dis[j]
                # 把该顶点做为下次遍历的上一个顶点
                last_point = j
        # 最短顶点假加入 visited 数组
        visited[last_point] = 1
        # 对首次循环做特殊处理，不然在首次循环时会没法求出该点的上一个顶点
        if i == 0:
            parents[last_point] = start + 1
        for k in range(node_num):
            if G[last_point][k] < inf and dis[
                    k] > dis[last_point] + G[last_point][k]:
                # 如果有更短的路径，更新 dis 和 记录 parents
                dis[k] = dis[last_point] + G[last_point][k]
                parents[k] = last_point + 1

    # 因为从 0 开始，最后把顶点都加 1
    return {key + 1: values
            for key, values in dis.items()
            }, {key + 1: values
                for key, values in parents.items()}


def get_each_house(house_file_path):
    house = pd.read_excel(house_file_path)
    house_kuancheng = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][:168]
    house_erdao = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][168:337]
    house_zhaoyang = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][337:537]
    house_lvyuan = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][537:721]
    house_nanguan = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][721:923]
    house_jingkai = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][923:1037]
    house_changchun = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][1037:1122]
    house_jingyue = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][1122:1268]
    house_qikai = house[['小区人口数（人）', '小区横坐标', '小区纵坐标']][1268:1409]
    return house_kuancheng, house_erdao, house_zhaoyang, house_lvyuan, house_nanguan, house_jingkai, house_changchun, house_jingyue, house_qikai


def get_locations(location, house_area):
    house_area_locations = []
    house_area_locations_index = []
    x_min = house_area['小区横坐标'].min()
    x_max = house_area['小区横坐标'].max()
    y_min = house_area['小区纵坐标'].min()
    y_max = house_area['小区纵坐标'].max()
    for loc in location.iterrows():
        if x_min <= loc[1]['路口横坐标'] <= x_max and y_min <= loc[1][
                '路口纵坐标'] <= y_max:
            house_area_locations.append(
                np.array([loc[1]['路口横坐标'], loc[1]['路口纵坐标']]))
            house_area_locations_index.append(loc[1]['节点编号'])
    return house_area_locations, house_area_locations_index


def get_adjacent(locations, route_dict):
    locations_index = locations[1]
    # print(locations_index)
    dim = len(locations_index)
    S = np.zeros((dim, dim))
    inf = float('inf')
    for i in range(dim):
        for j in range(i + 1, dim):
            distance = route_dict[locations_index[i]].get(
                locations_index[j], 0)
            if distance:
                S[i][j] = distance
            else:
                S[i][j] = inf
            S[j][i] = S[i][j]
    return S


def get_min_dis(adjacent_matrix):
    dim = adjacent_matrix.shape[1]
    min_dis = {}
    for i in range(dim):
        dis, parents = Dijkstra(adjacent_matrix, i + 1)
        min_dis[i + 1] = dis
    return min_dis


def get_min_dis_each_area(location, area_names, area_list):
    route_dict = collections.defaultdict(dict)
    route = pd.read_csv('route.csv')
    for row in route.iterrows():
        route_dict[row[1]['start']][row[1]['end']] = row[1]['dis']
    np.save('route_dict.npy', route_dict)
    locations_each_area = {}
    for area_name, area in zip(area_names, area_list):
        locations_each_area[area_name] = get_locations(location, area)
    np.save('locations_each_area.npy', locations_each_area)
    for area_name in area_names[1:]:
        adjacent_matrix = get_adjacent(locations_each_area[area_name],
                                       route_dict)
        min_dis = get_min_dis(adjacent_matrix)
        np.save(f'min_dis_{area_name}.npy', min_dis)


# location = pd.read_excel('location.xlsx')
# area_list = list(get_each_house('house1.xlsx'))
# area_names = [
#     'kuancheng', 'erdao', 'zhaoyang', 'lvyuan', 'nanguan', 'jingkai',
#     'changchun', 'jingyue', 'qikai'
# ]
# get_min_dis_each_area(location, area_names, area_list)
