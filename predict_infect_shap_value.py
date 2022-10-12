import pandas as pd
import numpy as np
from os import listdir
import math
#导入数据
file_name =  'data1.csv'
data = pd.read_csv(file_name)
date = data['Date'].values.tolist()
y1 = data['illed'].values.tolist()
y2 = data['infected'].values.tolist()
x1 = data['y1'].values.tolist()
x2 = data['y2'].values.tolist()
x3 = data['y3'].values.tolist()
x4 = data['y4'].values.tolist()
X,Y = [],[]
for i in range(len(y1)):   
    Y.append(y1[i]+y2[i])
    if i == 0:
        X.append([x1[i],x2[i],x3[i],x4[i],403+554])
        #Y.append(y1[i]+y2[i]-554-403)
    else:
        X.append([x1[i],x2[i],x3[i],x4[i],y1[i-1]+y2[i-1]])
        #Y.append(y1[i]+y2[i]-y1[i-1]-y2[i-1])
print(len(X),len(X[0]),len(Y))
print(date)
#for i in range(len(X)):
 #   print(date[i],Y[i],X[i][0],X[i][1],X[i][2],X[i][3],X[i][4])
from os import listdir
import os
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
from sklearn import ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
#train with LASSO
def view_accuracy(testings, predictions):
    print("RMSE:",mean_squared_error(testings, predictions)**(1/2))
    print("R2:",r2_score(testings, predictions))
    plt.figure(figsize=(6,6))
    plt.ylabel('Predict Infected (eV)',fontsize=25)
    plt.xlabel('True Infected (eV)',fontsize=25)
    x1 = np.linspace(min([min(testings),min(predictions)])-1,max([max(testings),max(predictions)])+1,500)#从(-1,1)均匀取50个点
    y1 = x1
    plt.plot(x1,y1,color = 'black',linewidth = 1,dashes=[6, 2])
    plt.scatter(testings, predictions, c = 'salmon',alpha=0.8,label='10-fold_cross_validation')
    plt.legend(fontsize=15)
    plt.show()


rs = 42
R2 = []
RMSE = []
for alp in [1*i-10 for i in range(20)]:
    overfitting = []
    testings = []
    predictions = []
    index = []
    kf = KFold(n_splits=5,shuffle=True,random_state=rs)#k-fold cross validation
    for train_index,test_index in kf.split(X,Y):
        trainX,testX,trainY,testY = [],[],[],[]
        for i in range(len(Y)):
            if i in train_index:
                trainX.append(X[i])
                trainY.append(Y[i])
            else:
                testX.append(X[i])
                testY.append(Y[i])
        model = Lasso(alpha=10**alp,normalize=True, precompute=True, warm_start=True)
        model.fit(trainX, trainY) 
        #print(model.coef_,model)
        predictY_test = model.predict(testX)
        predictY_train = model.predict(trainX)
        testings.extend(testY)
        predictions.extend(predictY_test)
        index.extend(test_index)
        #print('RMSE_testing:',mean_squared_error(testY, predictY_test)**(1/2))
        #print('r2_testing:',r2_score(testY, predictY_test))
        #print('RMSE_training:',mean_squared_error(trainY, predictY_train)**(1/2))
        #print('r2_training:',r2_score(trainY, predictY_train))
        #overfitting.append(mean_squared_error(trainY, predictY_train)**(1/2)-mean_squared_error(testY, predictY_test)**(1/2))
    R2.append(r2_score(testings, predictions))
    RMSE.append(mean_squared_error(testings, predictions)**(1/2))
    print(alp)
    view_accuracy(testings, predictions)
print(max(R2),min(RMSE))
model = Lasso(alpha=10**0.75,normalize=True, precompute=True, warm_start=True)
model.fit(X,Y)
#train with xgb
import shap
import xgboost
import xgboost as xgb
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
parameter_space_gbr = {"colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                           "max_depth": hp.quniform("max_depth", 1, 10, 1),
                           "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
                           "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                           "subsample": hp.uniform("subsample", 0.9, 1),
                           "min_child_weight": hp.uniform("min_child_weight", 0.5, 10),
                           "gamma": hp.uniform("gamma", 0.01, 0.5)
                           }

def function(argsDict):
    colsample_bytree = argsDict["colsample_bytree"]
    max_depth = argsDict["max_depth"]
    n_estimators = argsDict['n_estimators']
    learning_rate = argsDict["learning_rate"]
    subsample = argsDict["subsample"]
    min_child_weight = argsDict["min_child_weight"]
    gamma = argsDict["gamma"]

    clf = xgb.XGBRegressor(nthread=4,    #进程数
                            colsample_bytree=colsample_bytree,
                            max_depth=int(max_depth),  #最大深度
                            n_estimators=int(n_estimators),   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            gamma=gamma,
                            random_state=int(42),
                            objective="reg:squarederror"
                            )
    clf.fit(trainX, trainY)
    prediction = clf.predict(testX)
    R2 = r2_score(testY, prediction)
    RMSE = mean_squared_error(testY, prediction)**(1/2)
    return RMSE


X = X
ynew1 = Y
RMSE = 0
r2 = 0
for rs in ([42]):
    trainX, testX, trainY, testY = train_test_split(X, ynew1,train_size=0.8, test_size=0.2, random_state=rs)
    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = trainY
    trials = Trials()
    best = fmin(function, parameter_space_gbr, algo=tpe.suggest, max_evals=200, trials=trials)
    parameters = ['colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate', 'gamma', 'min_child_weight']
    #hyper_parameters_plot(parameters, trials, Loop_Step, screen_step)
    colsample_bytree = best["colsample_bytree"]
    max_depth = best["max_depth"]
    n_estimators = best['n_estimators']
    learning_rate = best["learning_rate"]
    subsample = best["subsample"]
    min_child_weight = best["min_child_weight"]
    gamma = best["gamma"]
    print("The_best_parameter：", best)

    gbr = xgb.XGBRegressor(nthread=4,    #进程数
                                colsample_bytree=colsample_bytree,
                                max_depth=int(max_depth),  #最大深度
                                n_estimators=int(n_estimators),   #树的数量
                                learning_rate=learning_rate, #学习率
                                subsample=subsample,      #采样数
                                min_child_weight=min_child_weight,   #子数
                                gamma=gamma,
                                random_state=int(42),
                                objective="reg:squarederror"
                                )
    gbr.fit(np.array(trainX), trainY)
    predY = gbr.predict(np.array(testX))
    print('RMSE:',mean_squared_error(testY, predY)**(1/2))
    print('r2:',r2_score(testY,predY))
    RMSE += mean_squared_error(testY, predY)**(1/2)
    r2 += r2_score(testY,predY)
print(gbr.feature_importances_,len(gbr.feature_importances_))
print(X,Y)
best,True_y,Pred_y = model_training_with_kfold_cv(X,Y,5,500,32)
#view_accuracy(True_y,Pred_y)
#SHAP基线
hyperparameter_dict = dict({'colsample_bytree': 0.9490648712766054, 'gamma': 0.05419789221052884, 'learning_rate': 0.35900789445158116, 'max_depth': 5.0, 'min_child_weight': 3.3249426022527295, 'n_estimators': 11.0, 'subsample': 0.9789421913034576})
model = xgb.XGBRegressor(nthread=4,    
                                colsample_bytree=hyperparameter_dict['colsample_bytree'],
                                max_depth=int(hyperparameter_dict['max_depth']),  
                                n_estimators=int(hyperp2\arameter_dict['n_estimators']),   
                                learning_rate=hyperparameter_dict['learning_rate'], 
                                subsample=hyperparameter_dict['subsample'],      
                                min_child_weight=hyperparameter_dict['min_child_weight'],   
                                gamma=hyperparameter_dict['gamma'],
                                random_state=int(42),
                                objective="reg:squarederror"
                                )
model.fit(X,Y)
Pred_y = model.predict(X)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(np.array(X)) # 获取训练集data各个样本各个特征的SHAP值
y_base = explainer.expected_value
print(y_base)
#可视化预测结果
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#P = [0 for i in range(len(date))]
#for i in range(len(Pred_y)):
#    P[Y.index(True_y[i])] = Pred_y[i]
print('R2:',r2_score(Y,Pred_y))
print('RMSE:',mean_squared_error(Y,Pred_y)**0.5)
plt.figure(figsize=(6,6))
font = FontProperties(fname='SimHei.ttf',size=16)
plt.ylabel('感染人数',fontproperties=font)
plt.xlabel('日期',fontproperties=font)
y_major_locator=MultipleLocator(500)
x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
ax=plt.gca()
#plt.plot(x1,y1,color = 'black',linewidth = 1)
plt.plot(date, Pred_y, c = 'firebrick',alpha=0.8,label='预测')
plt.plot(date, Y, c = 'royalblue',alpha=0.8,label='实际')
plt.legend(prop=font)
plt.savefig('/mnt/c/Users/azere/Desktop/data/infection_XGB.png',dpi = 600,bbox_inches='tight')
plt.show()
#
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
font = FontProperties(fname='SimHei.ttf',size=16)
SISSO = []
for i in range(len(X)):
    SISSO.append(0.8306*(X[i][2]+X[i][0]-X[i][1])+0.2638241397E+02)
print('R2:',r2_score(Y,SISSO))
print('RMSE:',mean_squared_error(Y,SISSO)**0.5)
plt.figure(figsize=(6,6))
plt.ylabel('感染人数',fontproperties=font)
plt.xlabel('日期',fontproperties=font)
y_major_locator=MultipleLocator(500)
x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
ax=plt.gca()
#plt.plot(x1,y1,color = 'black',linewidth = 1)
plt.plot(date, SISSO, c = 'firebrick',alpha=0.8,label='预测')
plt.plot(date, Y, c = 'royalblue',alpha=0.8,label='实际')
plt.legend(prop=font)
plt.savefig('/mnt/c/Users/azere/Desktop/data/infection_SISSO.png',dpi = 600,bbox_inches='tight')
plt.show()
print(Y[15:21])
#SHAP分析
import pandas as pd
shap.initjs()
player_explainer = pd.DataFrame()
player_explainer['feature'] = ['purchase items','purchase weight','distribute items','distribute weight','infected(Day-1)']
player_explainer['feature_value'] = np.array(X[i])
player_explainer['shap_value'] = shap_values[i]
player_explainer
#help(shap.plots.force)
i = 20
print(date[i])
shap.plots.force(explainer.expected_value,shap_values[i], feature_names=['自采箱数','自采重量','发放箱数','发放重量','前一天感染人数'])
print(shap_values.T)
print(shap_vasp)
cols = player_explainer['feature'].values.tolist()
data = pd.DataFrame()
print(len(cols))
temp = []
for i in range(len(cols)):
    temp.append([])
newfv1 = X
for i in range(len(newfv1)):
    for j in range(len(cols)):
        temp[j].append(newfv1[i][j])
plt.rcParams['font.sans-serif'] = ['simhei']      
for i in range(len(cols)):
    data[cols[i]] = np.array(temp[i])
data[cols]
shap_values = explainer.shap_values(data[cols])
#help(shap.summary_plot)
print(len(shap_values[0]))
shap.summary_plot(shap_values, data[cols],plot_type='bar',show = True)
#plt.savefig('/mnt/c/Users/azere/Desktop/SHAP.png',dpi = 600,bbox_inches='tight')
#shap.summary_plot(shap_values, data[cols],plot_type='violin',show = True)
shap.summary_plot(shap_values, data[cols],show = True)




