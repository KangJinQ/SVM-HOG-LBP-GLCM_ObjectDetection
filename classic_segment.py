import numpy as np
from src.segmentThreshold import *
from src.edgeDetection import *
from src.segmentRegion import *
from src.segmentWithDetection import segment_WithDetection
from src.svmDetection import *


# 固定阈值分割
def segmentFixedThreshold(self, t):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = segmentFixed_Threshold(img_y, t)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# OTSU大津法分割
def segmentOTSU(self):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = segmentOTSU_Threshold(img_y)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# 自适应阈值分割
def segmentAdaptive(self):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = segmentAdaptive_Threshold(img_y)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# Canny算子边缘检测
def edgeDetectionCanny(self):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = edgeDetection_Canny(img_y)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# Sobel算子边缘检测
def edgeDetectionSobel(self):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    # img = self.input_image
    img_y = img_y.astype(np.uint8)
    img1 = edgeDetection_Sobel(img_y)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# 区域分割算法——分水岭算法
def waterShed(self):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    # img = self.input_image
    img_y = img_y.astype(np.uint8)
    img1 = water_Shed(img_y)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# svm检测缺陷位置
def svmDetection(self, model_version):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = svm_Detection(img_y, model_version)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)


# 基于检测-分割的数字图像处理方法
def segmentWithDetection(self, method, t):
    img_y = self.input_array.transpose(1, 2, 0).copy()
    img_y = img_y.astype(np.uint8)
    img1 = segment_WithDetection(img_y, method, t)
    photo = img1.copy()
    self.output_array = photo.transpose(2, 0, 1)
