from sklearn.externals import joblib
from sklearn import preprocessing
from xml.dom import minidom
import numpy as np
import os
import datetime

path = os.path.dirname(os.path.abspath(__file__))

# 设定不同工况的输入阈值
rand_input_0 = [[-0.009, 0.039], [0, 0.023], [-0.549, -0.449], [2.619, 2.899], [78.759, 99.459], [1.470, 1.471],
                [-2.799, -2.749], [14.989, 15.489], [1.729, 1.769], [228.516, 228.517], [5307.599, 5308.109]]

rand_input_1 = [[0, 0.049], [0, 0.229], [-0.539, -0.469], [2.469, 3.339], [43.769, 78.009], [1.470, 1.471],
                [-2.739, -2.679], [14.829, 15.439], [1.729, 1.779], [228.519, 228.521], [5307.549, 5308.009]]

data_list_eng = ['X-axis-current', 'X-axis-load', 'Y-axis-current', 'Y-axis-load', 'Y-axis-pos', 'Y-axis-power',
                 'Z-axis-current', 'Z-axis-load', 'spindle-load', 'spindle-power', 'spindle-speed']

test_data_0 = []
test_data_1 = []


def generate_input_data():
    for i in range(10000):
        temp_0, temp_1 = [], []
        for j in range(len(data_list_eng)):
            temp_0.append(np.random.uniform(rand_input_0[j][0], rand_input_0[j][1]))
            temp_1.append(np.random.uniform(rand_input_1[j][0], rand_input_1[j][1]))
        temp_0 = np.array(temp_0)
        temp_1 = np.array(temp_1)
        test_data_0.append(temp_0)
        test_data_1.append(temp_1)


def get_time():
    now_time_str = str(datetime.datetime.now())
    temp = now_time_str.split(' ')
    temp_0 = temp[0].split('-')
    temp_0[0], temp_0[1], temp_0[2] = temp_0[1], temp_0[2], temp_0[0]
    temp_new = '/'.join(temp_0)
    now_time_str = temp_new+' '+temp[1]
    return now_time_str


def save_to_xml(mode, name, test_data_scale, test_data):
    impl = minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'Data', None)
    Data = dom.documentElement

    for i in range(10000):
        # 创建一个stock_data节点
        stock_data = dom.createElement('StockData')
        # 添加工况Name(0或1)
        stock_data.setAttribute('Name', mode)
        # 添加时间戳
        stock_data.setAttribute('Date', get_time())
        # 添加输入数据
        for j in range(len(data_list_eng)):
            stock_data.setAttribute(data_list_eng[j], str(round(test_data[i][j], 4)))
        # 添加预测结果
        data = test_data_scale[i].reshape(1, -1)
        pred = svr.predict(data)[0]
        pred = pred if pred > 0 else 0
        stock_data.setAttribute('average-wear', str(round(pred, 4)))

        # 在xml中添加该stock_data节点
        Data.appendChild(stock_data)

    f = open(name+'.xml', 'w')
    dom.writexml(f, addindent=' ', newl='\n', encoding='utf-8')
    f.close()


if __name__ == "__main__":
    # 获取预训练文件
    svr = joblib.load(path + os.sep + 'SVR_model.pkl')

    # 生成输入数据
    generate_input_data()

    # 数据标准化
    standard_scaler = preprocessing.StandardScaler()
    test_data_0_scale = standard_scaler.fit_transform(test_data_0)
    test_data_1_scale = standard_scaler.fit_transform(test_data_1)

    # 将测试结果导出为.xml文件
    save_to_xml(mode='0', name='show_0', test_data_scale=test_data_0_scale, test_data=test_data_0)
    save_to_xml(mode='1', name='show_1', test_data_scale=test_data_1_scale, test_data=test_data_1)

    print('Done, .xml files are saved.')
