from sklearn import svm
from skimage import feature as ft
from sklearn.model_selection import GridSearchCV
import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

root = r'D:\zhang_work\svm\sklearn'#训练数据
shape_width = 100
shape_height = 140
reshape_len = shape_width * shape_height * 3
model_path = r'D:\zhang_work\svm\model'
test_path = r'D:\zhang_work\svm\check'

def fit():
    data_x = []
    data_y = []
    for char_no in os.listdir(root):
        char_path = os.path.join(root, char_no)
        # time_log(f'reading {char_path}')
        for file in os.listdir(char_path):
            image_path = os.path.join(char_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 灰度图片
            image = cv2.resize(image, (shape_width, shape_height))
            blur = cv2.bilateralFilter(image,9,75,75) # 双边滤波
            features = ft.hog(blur,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True, feature_vector= True) #HOG特征
            data_x.append(features[0])
            data_y.append(char_no)
    data = np.row_stack(data_x)
    svr = svm.SVC()
    parameters = {'kernel':('linear', 'rbf'), 
                'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 
                'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                'verbose': [True, False],
                'decision_function_shape':['ovo', 'ovr']}
    clf = GridSearchCV(svr, parameters, cv=5, n_jobs=8) # 网格搜索，自动调参， cv=5表示5折交叉验证， n_jobs=8表示用8核cpu
    clf.fit(data, data_y)
    print(clf.return_train_score)
    print(clf.best_params_)#打印出最好的结果
    best_model = clf.best_estimator_
    print("SVM Model save...")
    save_path = model_path + "\\" + "svm_efd_" + "train_model.pkl"
    joblib.dump(best_model,save_path)#保存最好的模型

def predict():
    model_file = os.path.join(model_path,os.listdir(model_path)[0])
    svc = joblib.load(model_file)
    errorcount = 0
    amounts = 0
    for check in os.listdir(test_path):
        check_path = os.path.join(test_path, check)
        for file in os.listdir(check_path):
            amounts+=1
            file_path = os.path.join(check_path, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (shape_width, shape_height))
            blur = cv2.bilateralFilter(image,9,75,75)
            features = ft.hog(blur,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True, feature_vector= True)
            vector = features[0].reshape(1,6336)#HOG特征
            val = svc.predict(vector)
            GT = file_path.split('\\')[4]#用文件名提取真实值
            if val != GT:
                errorcount+=1
            print("预测值：{}------真实值:{}".format(val,GT))
    print("共错了{}个测试数据，正确率为{}".format(errorcount,(amounts - errorcount)/amounts))