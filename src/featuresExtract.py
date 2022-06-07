import math

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def get_lbp(self):
    if self.ndim != 2:
        gray = cv2.cvtColor(self, cv2.COLOR_RGB2GRAY)
    else:
        gray = self
    n_points = 24
    radius = 8
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # 使用LBP方法提取图像的纹理特征.
    lbp = local_binary_pattern(gray, n_points, radius, 'uniform')
    # 统计图像的直方图
    max_bins = int(lbp.max() + 1)
    # hist size:256
    hist, _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins), density=True)
    # print("lbp:", hist.shape)
    return hist


orientations = 9
visualize = False


def get_hog2(self, model_version=None):
    if model_version == "162":
        my_pixels_per_cell = (10, 10)
        my_cells_per_block = (3, 6)
    elif model_version == "270":
        my_pixels_per_cell = (10, 10)
        my_cells_per_block = (3, 5)
    elif model_version == "324":
        my_pixels_per_cell = (10, 10)
        my_cells_per_block = (3, 3)
    else:
        my_pixels_per_cell = (10, 10)
        my_cells_per_block = (3, 5)
    # hog_fd = hog(self, orientations, pixels_per_cell=(8, 8),
    #              cells_per_block=(3, 3), visualize=visualize, channel_axis=2)
    hog_fd = hog(self, orientations, pixels_per_cell=my_pixels_per_cell,
                 cells_per_block=my_cells_per_block, visualize=visualize, channel_axis=2)
    return hog_fd


# 定义最大灰度级数
gray_level = 16


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print("图像的高宽分别为：height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    # print("max_gray_level:", max_gray_level)
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    # con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # agm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def get_glcm(img):
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return

    # 这里如果用‘/’会报错TypeError: integer argument expected, got float
    # 其实主要的错误是因为 因为cv2.resize内的参数是要求为整数
    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    if img.ndim != 2:
        # img = img.astype(np.float32)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)
    # print(glcm_0)

    asm, con, eng, idm = feature_computer(glcm_0)

    return np.array([asm, con, eng, idm])


def get_features_pre(self, model_version=None):
    return np.append(np.append(get_hog2(self, model_version), get_glcm(self)), get_lbp(self))
