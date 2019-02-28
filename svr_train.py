import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.externals import joblib

path = os.path.dirname(os.path.abspath(__file__))

# 读取原始数据，将.csv文件放在该脚本的同级目录下
raw_data = pd.read_csv(path + os.sep + 'data.csv')

# 将原始数据转化为dataframe
df = pd.DataFrame(data=raw_data)

# 设定data及其对应的label
data = df.loc[:, ['X轴电流', 'X轴负载', 'Y轴电流', 'Y轴负载', 'Y轴坐标', 'Y轴功率', 'Z轴电流', 'Z轴负载', '主轴负载', '主轴功率',
              '主轴转速']]
labels = df['平均磨损值']

# 划分训练集与测试集
x_train, x_test, y_train, y_test = \
    train_test_split(data.values, labels.values, test_size=0.01, random_state=3)

# 进行数据的标准化，使各输入参数的值在同一区间内
min_max_scaler = preprocessing.StandardScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# # 调用GridSearchCV，寻找使用SVR算法的最优参数进行回归
# C_range = np.logspace(3, 10, 8)
# gamma_range = np.logspace(-9, -5, 5)
# epsilon_range = np.linspace(0.001, 0.01, 10)
# param_grid = dict(C=C_range, epsilon=epsilon_range)
# svr = GridSearchCV(SVR(), param_grid=param_grid)
# print(svr.best_params_)
# print(svr.best_score_)
# print(svr.best_estimator_)

# 利用上面获得的最优参数，调用SVR算法，使用train dataset进行训练
svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.008, gamma='auto',
          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svr.fit(x_train, y_train)

# # 简单的测试
# temp = 0
# for i in range(len(y_test)):
#     pred_i = svr.predict(x_test[i].reshape(1, -1))[0]
#     print('第{0}组测试数据的预测值为{1}\t实际值为{2}'
#           .format(i + 1, round(pred_i, 4), round(y_test[i], 4),))
#     dis = (pred_i - y_test[i]) ** 2
#     temp += dis
# print('均方误差值约为{}'.format(np.sqrt(temp)/len(y_test)))

# 保存训练好的SVR模型
joblib.dump(svr, 'SVR_model.pkl')



