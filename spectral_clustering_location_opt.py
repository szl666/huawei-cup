from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from shapely.geometry import LinearRing, Polygon
import random
from scipy.optimize import minimize


def euclidDistance(x1, x2):
    res = np.sum((x1 - x2)**2)
    return res


def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N, N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1]
                         for m in range(k + 1)]  # xi's k nearest neighbours

        for j in neighbours_id:  # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
            A[j][i] = A[i][j]  # mutually

    return A


def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix**(0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


def perform_sp_kmeans(num_clusters, X, ori_kmeans=False):
    Similarity = calEuclidDistanceMatrix(X)
    Adjacent = myKNN(Similarity, k=9)
    Laplacian = calLaplacianMatrix(Adjacent)
    x, V = np.linalg.eig(Laplacian)
    dictx = dict(zip(x, range(len(X))))
    kEig = np.sort(x)[0:num_clusters]
    ix = [dictx[k] for k in kEig]
    H = V[:, ix]
    sp_kmeans = KMeans(n_clusters=num_clusters).fit(H)
    if ori_kmeans:
        pure_kmeans = KMeans(n_clusters=num_clusters).fit(X)
        return X, H, sp_kmeans, pure_kmeans
    else:
        return X, H, sp_kmeans


def draw_clustering_compare(X, H, sp_kmeans, pure_kmeans):
    plt.figure(figsize=(20, 8))
    plt.rcParams['font.size'] = 32
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=sp_kmeans.labels_, cmap='Set1')
    plt.title("Spectral Clustering")
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=pure_kmeans.labels_, cmap='Set1')
    plt.title("Kmeans Clustering")
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    print("silhouette_score", silhouette_score(H, sp_kmeans.labels_))
    print("silhouette_score", silhouette_score(X, pure_kmeans.labels_))
    print("davies_bouldin_score", davies_bouldin_score(H, sp_kmeans.labels_))
    print("davies_bouldin_score", davies_bouldin_score(X, pure_kmeans.labels_))


# house = pd.read_excel('house1.xlsx')
# X = np.array(house[['小区横坐标', '小区纵坐标']])[:168]
# X, H, sp_kmeans, pure_kmeans = perform_sp_kmeans(9, X, ori_kmeans=True)
# draw_clustering_compare(X,H, sp_kmeans, pure_kmeans)


def get_X_dict(X, sp_kmeans):
    X_dict = collections.defaultdict(list)
    for index, label in enumerate(sp_kmeans.labels_):
        X_dict[label].append(X[index])
    return X_dict


# get X_dict considering population
def get_X_dict_pop(X, sp_kmeans):
    X_dict = collections.defaultdict(list)
    for index, label in enumerate(sp_kmeans.labels_):
        X_dict[label].append(X[index])
    return X_dict


def cal_opt_pos(X_dict_ele):
    dis_list = []
    for pos_test in X_dict_ele:
        dis = 0
        for pos in X_dict_ele:
            dis += np.sqrt(np.sum((pos - pos_test)**2))
        dis_list.append(dis / len(X_dict_ele))
    min_index = np.argmin(dis_list)
    return min_index, X_dict_ele[min_index]


def get_opt_pos(X_dict):
    opt_pos_list = []
    for value in X_dict.values():
        _, opt_pos = cal_opt_pos(value)
        opt_pos_list.append(opt_pos)
    return opt_pos_list


def dist(a, b):
    """Computes the distance between two points"""
    return (sum([(a[i] - b[i])**2 for i in range(len(a))])**.5)


def cost(coord_, nodes):
    cost = 0
    cost = sum([dist(coord_, nodes[i]) for i in range(len(nodes))])
    return (cost)


def get_nearest_location(coords, locations):
    dis_house_location = [dist(coords,location) for location in locations]
    location_index = np.argmin(dis_house_location)
    dis = np.min(dis_house_location)
    location_pos = locations[location_index]
    return location_index, dis, location_pos

def dist_path(a, b, locations_area,min_dis_area):
    locations_area = np.array(locations_area)
    location_index_a, dis_to_loc_a, location_pos_a = get_nearest_location(a, locations_area)
    location_index_b, dis_to_loc_b, location_pos_b = get_nearest_location(b, locations_area)
    dis_between_locations = min_dis_area[location_index_a][location_index_b]
    return dis_between_locations+dis_to_loc_a+dis_to_loc_b


# cost considering population and path
def cost_path(coord_, nodes, locations_area,min_dis_area):
    nodes_coord = nodes[:, 0:2]
    cost = 0
    cost = sum([dist_path(coord_,nodes_coord[i], locations_area,min_dis_area) for i in range(len(nodes_coord))])
    return (cost)


def get_fermat_point(nodes, cost):
    coord = [
        random.uniform(min([i[0] for i in nodes]), max([i[0] for i in nodes])),
        random.uniform(min([i[1] for i in nodes]), max([i[1] for i in nodes]))
    ]
    xmin = min([i[0] for i in nodes])
    xmax = max([i[0] for i in nodes])
    ymin = min([i[1] for i in nodes])
    ymax = max([i[1] for i in nodes])
    bounds = ((xmin, xmax), (ymin, ymax))
    Iter = 10
    min_dist = [0, 10000]
    for i in range(Iter):
        ret = minimize(cost,
                       coord,
                       args=(nodes),
                       method='Nelder-Mead',
                       jac=None,
                       bounds=bounds,
                       tol=None,
                       callback=None,
                       options={'maxiter': 100})
        if ret.fun < min_dist[1]:
            min_dist[0] = min_dist[1]
            min_dist[1] = ret.fun
            min_coord = []
            min_coord.append(ret.x)
    return min_dist[1], min_coord[0]


# get fermat point considering population and path
def get_fermat_point_pop(nodes, cost_path):
    nodes_coord = nodes[:, 0:2]
    coord = [
        random.uniform(min([i[0] for i in nodes_coord]),
                       max([i[0] for i in nodes_coord])),
        random.uniform(min([i[1] for i in nodes_coord]),
                       max([i[1] for i in nodes_coord]))
    ]
    xmin = min([i[0] for i in nodes_coord])
    xmax = max([i[0] for i in nodes_coord])
    ymin = min([i[1] for i in nodes_coord])
    ymax = max([i[1] for i in nodes_coord])
    bounds = ((xmin, xmax), (ymin, ymax))
    Iter = 10
    min_dist = [0, 2e18]
    for i in range(Iter):
        ret = minimize(cost_path,
                       coord,
                       args=(nodes),
                       method='Nelder-Mead',
                       jac=None,
                       bounds=bounds,
                       tol=None,
                       callback=None,
                       options={'maxiter': 100})
        if ret.fun < min_dist[1]:
            min_dist[0] = min_dist[1]
            min_dist[1] = ret.fun
            min_coord = []
            min_coord.append(ret.x)
    return min_dist[1], min_coord[0]


def get_fermat_point_list(X_dict):
    fermat_point_list = []
    dist_list = []
    for value in X_dict.values():
        nodes = value
        min_dist, fermat_point = get_fermat_point(nodes, cost)
        fermat_point_list.append(fermat_point)
        dist_list.append(min_dist)
    return fermat_point_list, dist_list


def draw_fermat_point(nodes, fermat_point):
    nodes = Polygon(nodes)
    x, y = nodes.exterior.xy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    # ax.plot(x, y, color='#6699cc', alpha=0.7,linewidth=3, solid_capstyle='round', zorder=2)
    for i in range(len(x)):
        ax.plot([x[i], fermat_point[0]], [y[i], fermat_point[1]],
                'ro-',
                color='#6699cc',
                zorder=0)
    ax.scatter(fermat_point[0],
               fermat_point[1],
               marker='o',
               color='#DF0101',
               s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


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


def get_shangyou_point(area_list, area_names):
    shangyou_pos = {}
    for area_name, area in zip(area_names, area_list):
        nodes = np.array(area[['小区横坐标', '小区纵坐标', '小区人口数（人）']])
        shangyou_pos[area_name] = get_fermat_point(nodes, cost)
    shangyou_pos_df = pd.DataFrame(shangyou_pos).T
    shangyou_pos_df.to_csv('shangyou_pos.csv')
    return shangyou_pos


# area_list = list(get_each_house('house1.xlsx'))
# area_names = [
#     'kuancheng', 'erdao', 'zhaoyang', 'lvyuan', 'nanguan', 'jingkai',
#     'changchun', 'jingyue', 'qikai'
# ]
# shangyou_pos = get_shangyou_point('house1.xlsx')


def opt_total_zhongyou_pos(shangyou_pos, house_path, area_list, area_names):
    house = pd.read_excel(house_path)
    random.seed(42)
    overall_dis = []
    fermat_point_listss = []
    overall_dis_to_xiayou = []
    overall_dis_to_shangyou = []
    for zhongyou_nums_all in range(100, 500):
        distance_xiayou = []
        fermat_point_lists = []
        distance_shangyou = []
        population_all = house['小区人口数（人）'].sum()
        ratio_list = [
            area['小区人口数（人）'].sum() / population_all for area in area_list
        ]
        zhongyou_nums = [ratio * zhongyou_nums_all for ratio in ratio_list]
        for zhongyou_num, area, area_name in zip(zhongyou_nums, area_list,
                                                 area_names):
            X = np.array(area[['小区横坐标', '小区纵坐标']])
            _, H, sp_kmeans = perform_sp_kmeans(round(zhongyou_num),
                                                X,
                                                ori_kmeans=False)
            X_dict = get_X_dict(X, sp_kmeans)
            fermat_point_list, dist_list = get_fermat_point_list(X_dict)
            dis_to_shangyou = sum([
                dist(shangyou_pos[area_name][1], fermat_point)
                for fermat_point in fermat_point_list
            ])
            dis_to_xiayou = sum(dist_list)
            distance_xiayou.append(dis_to_xiayou)
            fermat_point_lists.append(fermat_point_list)
            distance_shangyou.append(dis_to_shangyou)
        fermat_point_listss.append(fermat_point_lists)
        overall_dis_to_xiayou.append(sum(distance_xiayou))
        overall_dis_to_shangyou.append(sum(distance_shangyou))
        overall_dis.append(sum(distance_xiayou) + sum(distance_shangyou))
    return fermat_point_listss, overall_dis_to_xiayou, overall_dis_to_shangyou, overall_dis


# area_list = list(get_each_house('house1.xlsx'))
# area_names = [
#     'kuancheng', 'erdao', 'zhaoyang', 'lvyuan', 'nanguan', 'jingkai',
#     'changchun', 'jingyue', 'qikai'
# ]
# shangyou_pos = get_shangyou_point('house1.xlsx')
# fermat_point_listss, overall_dis_to_xiayou, overall_dis_to_shangyou, overall_dis = opt_total_zhongyou_pos(shangyou_pos, 'house1.xlsx', area_list, area_names)


def get_house_index(coords_x, coords_y, house):
    for row in house.iterrows():
        if row[1]['小区横坐标'] == coords_x and row[1]['小区纵坐标'] == coords_y:
            index = row[1]['小区编号']
            code = row[1]['街道编号']
            pop = row[1]['小区人口数（人）']
    return index, code, pop


# get_house_index(X_dicts[0][8][0][0], X_dicts[0][8][0][1])


def get_clustering_result(shangyou_pos, house, area_list, area_names):
    random.seed(42)
    X_dicts = []
    distance_xiayou = []
    fermat_point_lists = []
    distance_shangyou = []
    population_all = house['小区人口数（人）'].sum()
    ratio_list = [
        area['小区人口数（人）'].sum() / population_all for area in area_list
    ]
    zhongyou_nums = [ratio * 199 for ratio in ratio_list]
    for zhongyou_num, area, area_name in zip(zhongyou_nums, area_list,
                                             area_names):
        pop = np.array(area[['小区人口数（人）']])
        X = np.array(area[['小区横坐标', '小区纵坐标']])
        _, H, sp_kmeans = perform_sp_kmeans(round(zhongyou_num),
                                            X,
                                            ori_kmeans=False)
        X_dict = get_X_dict(X, pop, sp_kmeans)
        X_dicts.append(X_dict)
        fermat_point_list, dist_list = get_fermat_point_list(X_dict)
        dis_to_shangyou = sum([
            dist(shangyou_pos[area_name][1], fermat_point) * pop.sum()
            for fermat_point in fermat_point_list
        ])
        dis_to_xiayou = sum(dist_list)
        distance_xiayou.append(dis_to_xiayou)
        fermat_point_lists.append(fermat_point_list)
        distance_shangyou.append(dis_to_shangyou)
    return X_dicts, dis_to_xiayou, distance_xiayou, distance_shangyou, fermat_point_lists


def get_info_each_zhongyou(area_names, X_dicts, fermat_point_lists):
    for index, area_name in enumerate(area_names):
        info = {}
        indexss = []
        codess = []
        popss = []
        for key, values in list(X_dicts[index].items()):
            indexs = []
            codes = []
            pops = []
            for value in values:
                index, code, pop = get_house_index(value[0], value[1])
                indexs.append(index)
                codes.append(code)
                pops.append(pop)
            indexss.append(indexs)
            codess.append(codes)
            popss.append(pops)
        info['小区数'] = [len(i) for i in indexss]
        info['人口数'] = [sum(i) for i in popss]
        info['坐标x'] = np.array(fermat_point_lists[index])[:, 0]
        info['坐标y'] = np.array(fermat_point_lists[index])[:, 1]
        info_df = pd.DataFrame(info)
        info_df.to_csv(f'{area_name}.csv')


# area_list = list(get_each_house('house1.xlsx'))
# area_names = [
#     'kuancheng', 'erdao', 'zhaoyang', 'lvyuan', 'nanguan', 'jingkai',
#     'changchun', 'jingyue', 'qikai'
# ]
# shangyou_pos = get_shangyou_point('house1.xlsx')
# house = pd.read_excel('house1.xlsx')
# X_dicts, dis_to_xiayou, distance_xiayou, distance_shangyou, fermat_point_lists = get_clustering_result(
#     shangyou_pos, house, area_list, area_names)
# get_info_each_zhongyou(area_names, X_dicts, fermat_point_lists)


def draw_opt_step(list1,
                  list2,
                  list3,
                  name,
                  ylabel,
                  scale=1,
                  markersize=10,
                  overall_dis):
    plt.figure(figsize=(18, 12))
    plt.rcParams['font.size'] = 32
    plt.rcParams['font.sans-serif'] = ['simhei']
    steps = np.array(list(range(len(list1)))) * scale + 100
    plt.plot(steps,
             list1,
             marker='o',
             linewidth=2.0,
             linestyle='-',
             markersize=markersize,
             markerfacecolor='#377eb8',
             markeredgewidth=1.2,
             alpha=0.9,
             markeredgecolor='k')
    plt.plot(steps,
             list2,
             marker='o',
             linewidth=2.0,
             linestyle='-',
             markersize=markersize,
             markerfacecolor='#ff7f00',
             markeredgewidth=1.2,
             alpha=0.9,
             markeredgecolor='k')
    plt.plot(steps,
             list3,
             marker='o',
             linewidth=2.0,
             linestyle='-',
             markersize=markersize,
             markerfacecolor='#4daf4a',
             markeredgewidth=1.2,
             alpha=0.9,
             markeredgecolor='k')
    plt.scatter(np.argmin(overall_dis) + 100,
                np.min(overall_dis),
                c='orangered',
                alpha=0.9,
                marker='*',
                edgecolors='black',
                s=1500,
                linewidths=2,
                zorder=20000)
    plt.legend(labels=['总运输里程', '上游到中游总运输里程', '中游到下游总运输里程'], loc='best')
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.ylabel(ylabel)
    plt.xlabel("各项物资的集散地总数(中游)")
    plt.savefig(name, bbox_inches='tight', dpi=1200)


# fermat_point_listss, overall_dis_to_xiayou, overall_dis_to_shangyou, overall_dis = opt_total_zhongyou_pos(shangyou_pos, 'house1.xlsx', area_list, area_names)
# draw_opt_step(overall_dis,
#               overall_dis_to_shangyou,
#               overall_dis_to_xiayou,
#               'opt_process.pdf',
#               '人力数',
#               scale=1,
#               markersize=10)

def coefficient_vs_best_zhongyounum(overall_dis_to_shangyou, overall_dis_to_xiayou, up)
    best_zhongyou = []
    for i in range(up):
        overall_dis = overall_dis_to_shangyou/i +overall_dis_to_xiayou
        best_zhongyou.append(np.argmin(overall_dis)+100)
    return best_zhongyou

def draw_opt_step_coefficient(list1, name, ylabel, scale=1, markersize=10):
    plt.figure(figsize=(18, 12))
    plt.rcParams['font.size'] = 32
    plt.rcParams['font.sans-serif'] = ['simhei']
    steps = np.array(list(range(len(list1)))) * scale
    plt.plot(steps,
             list1,
             marker='o',
             linewidth=2.0,
             linestyle='-',
             markersize=markersize,
             markerfacecolor='#377eb8',
             markeredgewidth=1.2,
             alpha=0.9,
             markeredgecolor='dimgrey')
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.ylabel(ylabel)
    plt.xlabel("系数")
    plt.savefig(name, bbox_inches='tight', dpi=1200)

# best_zhongyou = coefficient_vs_best_zhongyounum(overall_dis_to_shangyou, overall_dis_to_xiayou, 1000)
# draw_opt_step(best_zhongyou,
#               'opt_process_coefficient.pdf',
#               '最优各项物资的集散地总数(中游)',
#               scale=1,
#               markersize=10)


