import xgboost as xgb
from os import listdir
import os
import csv
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn import ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
import json
from tqdm import tqdm
import random
from matplotlib.font_manager import FontProperties
from bird_swarm_opt import Bird_swarm_opt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

parameter_space_gbr = {
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "subsample": hp.uniform("subsample", 0.9, 1),
    "min_child_weight": hp.uniform("min_child_weight", 0.5, 10),
    "gamma": hp.uniform("gamma", 0.01, 0.5)
}


def model_training_with_kfold_cv(X, Y, k, ite, rs):
    def function(argsDict):
        colsample_bytree = argsDict["colsample_bytree"]
        max_depth = argsDict["max_depth"]
        n_estimators = argsDict['n_estimators']
        learning_rate = argsDict["learning_rate"]
        subsample = argsDict["subsample"]
        min_child_weight = argsDict["min_child_weight"]
        gamma = argsDict["gamma"]
        model = xgb.XGBRegressor(nthread=4,
                                 colsample_bytree=colsample_bytree,
                                 max_depth=int(max_depth),
                                 n_estimators=int(n_estimators),
                                 learning_rate=learning_rate,
                                 subsample=subsample,
                                 min_child_weight=min_child_weight,
                                 gamma=gamma,
                                 random_state=int(42),
                                 objective="reg:squarederror")

        kf = KFold(n_splits=k, shuffle=True,
                   random_state=rs)  #k-fold cross validation
        T_y = []
        P_y = []
        for train_index, test_index in kf.split(X, Y):
            trainX, testX, trainY, testY = [], [], [], []
            for i in range(len(Y)):
                if i in train_index:
                    trainX.append(X[i])
                    trainY.append(Y[i])
                else:
                    testX.append(X[i])
                    testY.append(Y[i])
            trainX = np.array(trainX)
            testX = np.array(testX)
            model.fit(trainX, trainY)
            prediction = model.predict(testX)
            T_y.extend(testY)
            P_y.extend(prediction)
            #R2 = r2_score(testY, prediction)
        RMSE = mean_squared_error(T_y, P_y)**(1 / 2)
        R2 = r2_score(T_y, P_y)
        return -R2

    trials = Trials()
    best = fmin(function,
                parameter_space_gbr,
                algo=tpe.suggest,
                max_evals=ite,
                trials=trials)
    parameters = [
        'colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate',
        'gamma', 'min_child_weight'
    ]
    #hyper_parameters_plot(parameters, trials, Loop_Step, screen_step)
    colsample_bytree = best["colsample_bytree"]
    max_depth = best["max_depth"]
    n_estimators = best['n_estimators']
    learning_rate = best["learning_rate"]
    subsample = best["subsample"]
    min_child_weight = best["min_child_weight"]
    gamma = best["gamma"]
    print("The_best_parameter：", best)
    gbr = xgb.XGBRegressor(
        nthread=4,  #进程数
        colsample_bytree=colsample_bytree,
        max_depth=int(max_depth),  #最大深度
        n_estimators=int(n_estimators),  #树的数量
        learning_rate=learning_rate,  #学习率
        subsample=subsample,  #采样数
        min_child_weight=min_child_weight,  #子数
        gamma=gamma,
        random_state=int(42),
        objective="reg:squarederror")
    kf = KFold(n_splits=k, shuffle=True,
               random_state=rs)  #k-fold cross validation
    True_y = []
    Pred_y = []
    for train_index, test_index in kf.split(X, Y):
        trainX, testX, trainY, testY = [], [], [], []
        for i in range(len(Y)):
            if i in train_index:
                trainX.append(X[i])
                trainY.append(Y[i])
            else:
                testX.append(X[i])
                testY.append(Y[i])
        trainX = np.array(trainX)
        testX = np.array(testX)
        gbr.fit(trainX, trainY)
        p = gbr.predict(testX)
        True_y.extend(testY)
        Pred_y.extend(p)
    return (best, True_y, Pred_y)


# district_name = ['朝阳区', '南关区', '二道区', '宽城区', '绿园区', '长春新', '经开区', '净月区', '汽开区', '莲花山', '九台区', '中韩示']
# district_people = [57.8,48.9,32.6,38.5,42.6,36.8,20.3,22.8,21.7,5.1,0,0]
def generate_X(list_rsvsnd, X_og):
    list_rsvsnd = list(list_rsvsnd)
    list_rsvsnd.extend([
        1 - sum([
            list_rsvsnd[0], list_rsvsnd[2], list_rsvsnd[4], list_rsvsnd[6],
            list_rsvsnd[8]
        ]), 1 - sum([
            list_rsvsnd[1], list_rsvsnd[3], list_rsvsnd[5], list_rsvsnd[7],
            list_rsvsnd[9]
        ])
    ])
    sum_rsvandsnd = [0 for i in range(4)]
    for i in range(len(district_name)):
        for j in range(len(X_og[i])):
            sum_rsvandsnd[0] += X_og[i][j][0]
            sum_rsvandsnd[1] += X_og[i][j][1]
            sum_rsvandsnd[2] += X_og[i][j][2]
            sum_rsvandsnd[3] += X_og[i][j][3]
    sum_rsvandsnd_perdst = []
    for i in range(len(district_name)):
        sum_rsvandsnd_perdst.append(district_people[i] / sum(district_people) *
                                    np.array(sum_rsvandsnd))
    X_out = []
    #print(sum_rsvandsnd)
    #print(sum_rsvandsnd_perdst)
    #print(list_rsvsnd)
    for i in range(len(district_name)):
        temp = []
        for j in range(len(X_og[i])):
            temp.append([
                sum_rsvandsnd_perdst[i][0] * list_rsvsnd[2 * j],
                sum_rsvandsnd_perdst[i][1] * list_rsvsnd[2 * j],
                sum_rsvandsnd_perdst[i][2] * list_rsvsnd[2 * j + 1],
                sum_rsvandsnd_perdst[i][3] * list_rsvsnd[2 * j + 1],
                X_og[i][j][4]
            ])
        X_out.append(temp)
    #print(np.array(X_out).shape)
    return (np.array(X_out))


def get_XY(y1, y2, x1, x2, x3, x4):
    X, Y = [], []
    for i in range(len(y1)):
        Y.append(y1[i] + y2[i])
        if i == 0:
            X.append([x1[i], x2[i], x3[i], x4[i], 0])
            #Y.append(y1[i]+y2[i]-554-403)
        else:
            X.append([x1[i], x2[i], x3[i], x4[i], y1[i - 1] + y2[i - 1]])
            #Y.append(y1[i]+y2[i]-y1[i-1]-y2[i-1])
    print(len(X), len(X[0]), len(Y))
    return (X, Y)


def get_model(hyperparameter_dict, X,
              Y):  #get explainer for a set of hyperparameters
    model = xgb.XGBRegressor(
        nthread=4,
        colsample_bytree=hyperparameter_dict['colsample_bytree'],
        max_depth=int(hyperparameter_dict['max_depth']),
        n_estimators=int(hyperparameter_dict['n_estimators']),
        learning_rate=hyperparameter_dict['learning_rate'],
        subsample=hyperparameter_dict['subsample'],
        min_child_weight=hyperparameter_dict['min_child_weight'],
        gamma=hyperparameter_dict['gamma'],
        random_state=int(42),
        objective="reg:squarederror")
    model.fit(X, Y)
    return (model)


def get_hyper_X_dis_Y_dis(file_name, district_name):
    file_name = 'data1.csv'
    data = pd.read_csv(file_name)
    date = data['Date'].values.tolist()
    X_dis, Y_dis = [], []
    data1 = pd.read_csv('illed.csv', encoding='gbk')
    data2 = pd.read_csv('infect.csv', encoding='gbk')
    list_inf = list(data1)
    for i in range(len(list_inf)):
        list_inf[i] = list_inf[i][:3]
    data1.columns = list_inf
    data2.columns = list_inf
    print('Check_datasize')
    for i in range(len(district_name)):
        print(district_name[i])
        if district_name[i] in list_inf:
            y1 = data1[district_name[i]].values.tolist()
            y2 = data2[district_name[i]].values.tolist()
            x1 = data[district_name[i] + 'y1'].values.tolist()
            x2 = data[district_name[i] + 'y2'].values.tolist()
            x3 = data[district_name[i] + 'y3'].values.tolist()
            x4 = data[district_name[i] + 'y4'].values.tolist()
        else:
            y1, y2, x1, x2, x3, x4 = [0 for i in range(37)], [
                0 for i in range(37)
            ], [0 for i in range(37)], [0 for i in range(37)
                                        ], [0 for i in range(37)
                                            ], [0 for i in range(37)]
        dx, dy = get_XY(y1, y2, x1, x2, x3, x4)
        X_dis.append(dx)
        Y_dis.append(dy)

    models = pd.read_csv('model_district.csv')
    hyper = models['B'].values.tolist()
    test = eval(hyper[0])
    print(type(test))
    return hyper, X_dis, Y_dis


# file_name =  'data1.csv'
# district_name = ['朝阳区', '南关区', '二道区', '宽城区', '绿园区', '长春新', '经开区', '净月区', '汽开区', '莲花山', '九台区', '中韩示']
# hyper, X_dis, Y_dis = get_hyper_X_dis_Y_dis(file_name, district_name)


def cal_RMSE_infected(X_11_dis):
    RMSE = 0
    Infect = 0
    for i in range(len(hyper)):
        #print(district_name[i])
        model = get_model(eval(hyper[i]), np.array(X_dis[i]),
                          np.array(Y_dis[i]))
        X_dis_new = np.array(X_11_dis[i])
        for j in range(len(X_dis_new)):
            if j > 0:
                X_dis_new[j][4] = model.predict(X_dis_new)[j - 1]
        Infect += model.predict(X_dis_new)
        RMSE += mean_squared_error(Y_dis[i][15:21],
                                   model.predict(X_dis_new))**(1 / 2) / 6
    #print(model)
    return (RMSE, Infect)


# # test random_search
# X_og = []
# for i in range(len(X_dis)):
#     X_og.append(X_dis[i][15:21])
# #print(np.array(X_og).shape)#12个区，5天，5个样本每天
# RMSE_og,Infected_og = cal_RMSE_infected(X_og)
# print(RMSE_og,sum(Infected_og)/5)
# Infect = [[864.7133422851563]]
# sum_Infect = []
# plan = ['og']
# for i in tqdm(range(100)):
#     inlist = []
#     for j in range(5):
#         rnd1 = random.random() - 0.5
#         rnd2 = random.random() - 0.5
#         inlist.append(0.2 + rnd1 * 0.4)
#         inlist.append(0.2 + rnd1 * 0.4 + rnd2 * 0.2)
#     if inlist[0] + inlist[2] + inlist[4] + inlist[6] + inlist[8] < 1 and inlist[
#             1] + inlist[3] + inlist[5] + inlist[7] + inlist[9] < 1:
#         X = generate_X(inlist, X_og)
#         RMSE, Infected = cal_RMSE_infected(X)
#         if sum(Infected) / 6 < min(Infect):
#             Infect.append(sum(Infected / 6))
#             plan.append(inlist)
#             print(RMSE, sum(Infected) / 6, Infected, inlist)

# test_bird_swarm_optimization
# X_og = []
# for i in range(len(X_dis)):
#     X_og.append(X_dis[i][15:21])
# #print(np.array(X_og).shape)#12个区，5天，5个样本每天
# RMSE_og,Infected_og = cal_RMSE_infected(X_og)
# print(RMSE_og,sum(Infected_og)/5)
# bsa = Bird_swarm_opt(dim = 10, X_og=X_og, min_value = 0, max_value = 0.5)
# fMin, bestIndex, bestX, b2 = bsa.search(M=1000)
# np.save('fMin.npy',fMin)
# np.save('bestIndex.npy',bestIndex)
# np.save('bestX.npy',bestX)
# np.save('b2.npy',b2)

def draw_recieve_send(date, inlist):
    font = FontProperties(fname='SimHei.ttf',size=14)
    xr,xs = [],[]
    for i in range(5):
        xr.append(inlist[2*i])
        xs.append(inlist[2*i+1])
        print(xr,xs)
    xr.extend([1-sum(xr)])
    xs.extend([1-sum(xs)])
    plt.ylabel('接收/发送',fontproperties=font)
    plt.xlabel('日期',fontproperties=font)
    plt.plot(date, xr, c = 'firebrick',alpha=0.8,label = '接收')
    plt.plot(date, xs, c = 'royalblue',alpha=0.8, label = '发送')
    plt.scatter(date, xr, c = 'black',alpha=0.5,linewidths=1)
    plt.scatter(date, xs, c = 'black',alpha=0.5,linewidths=1)
    plt.legend(prop=font)
    plt.show()

# date = ['4/10','4/11','4/12','4/13','4/14','4/15']
# inlist = [0.009692070810964154, 0.07115274301118318, 0.19049756848547392, 0.1809725228993294, 0.11827059625149411, 0.04332998170558683, 0.12844478408075335, 0.07655661285750662, 0.16780249601909786, 0.14348243780435077, 0.3852924843522165, 0.4845057017220432]
# draw_recieve_send(date, inlist)

def draw_compare(date, inf_og, inf_opt):
    plt.ylabel('感染人数',fontproperties=font)
    plt.xlabel('日期',fontproperties=font)
    plt.plot(date, inf_og, c = 'firebrick',alpha=1,label = '原方案')
    plt.plot(date, inf_opt, c = 'royalblue',alpha=0.8, label = '调整方案')
    plt.scatter(date, inf_og, c = 'black',alpha=0.5,linewidths=1)
    plt.scatter(date, inf_opt, c = 'black',alpha=0.5,linewidths=1)
    plt.legend(prop=font)
    plt.show()

# date = ['4/10','4/11','4/12','4/13','4/14','4/15']
# inf_og = [845, 651, 974, 906, 436,564]
# inf_opt = [598.6063, 717.8098, 511.02844, 320.64313, 417.3214 ,408.86835]
# font = FontProperties(fname='SimHei.ttf',size=14)
draw_compare(date, inf_og, inf_opt)
