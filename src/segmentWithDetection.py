import os

import cv2
import joblib

from src.svmDetection import sliding_window
from src.nms import nms
from src.featuresExtract import *


def segment_WithDetection(im, method, t):
    print(im.shape)
    min_wdw_sz = (60, 30)

    step_size = (20, 10)
    downscale = 1

    path = os.getcwd()  # 获取当前路径
    print(path)
    model_path = path + "\\src\\svm270lbp.model"
    visualize_det = False
    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # This list contains detections at the current scale
    cd = []

    for (x, y, im_window) in sliding_window(im, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the features
        fd = get_features_pre(im_window)  # 原始
        fd = fd.reshape(1, -1)
        pred = clf.predict(fd)
        if pred == 1:
            print("Detection:: Location -> ({}, {})".format(x, y))
            print("Scale ->  {} | Confidence Score {} ".format(scale, clf.decision_function(fd)))
            detections.append((x, y, clf.decision_function(fd),
                               int(min_wdw_sz[0] * (downscale ** scale)),
                               int(min_wdw_sz[1] * (downscale ** scale))))
            print(detections[-1])
            print("\n")
            cd.append(detections[-1])

    print("the len of detections before nms is: ", len(detections))

    threshold = .3
    detections = nms(detections, threshold)
    print("the len of detections after nms is: ", len(detections))

    blank = np.zeros(im.shape)
    blank[:, :, :] = 255
    print(blank.shape)
    for (x, y, _, w, h) in detections:
        if method == "adaptive":
            blank[y:y + h, x:x + w, :] = segmentAdaptive_Threshold(im[y:y + h, x:x + w, :])
        elif method == "fixed":
            blank[y:y + h, x:x + w, :] = segmentFixed_Threshold(im[y:y + h, x:x + w, :], t)
        elif method == "OTSU":
            t, blank[y:y + h, x:x + w, :] = segmentOTSU_Threshold(im[y:y + h, x:x + w, :])
        elif method == "Canny":
            blank[y:y + h, x:x + w, :] = edgeDetection_Canny(im[y:y + h, x:x + w, :])
            t = "canny"
        else:
            blank[y:y + h, x:x + w, :] = segmentAdaptive_Threshold(im[y:y + h, x:x + w, :])
    print("分割阈值为: ", t)
    print("共检测出缺陷部分{}处".format(len(detections)))
    # Display the results after performing NMS
    clone = im.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    #  = np.array(clone, dtype=np.uint8)
    OUT_IMG = np.array(blank, dtype=np.uint8)
    return OUT_IMG


def segmentFixed_Threshold(src, t):
    # src = np.asarray(src, dtype="float32")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src, t, 255, cv2.THRESH_BINARY)  # 阈值t
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(img_close, cv2.COLOR_GRAY2BGR)
    return IMG_OUT


def segmentOTSU_Threshold(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(img_close, cv2.COLOR_GRAY2BGR)
    return ret, IMG_OUT


def segmentAdaptive_Threshold(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, 6)
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(img_close, cv2.COLOR_GRAY2BGR)
    return IMG_OUT


def edgeDetection_Canny(self):
    src = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    # 50 is the low threshold value
    # if the pixel is lower than the low value, the pixel is considered not to be on the edge

    # 150 is the high threshold value
    # if the pixel is greater than the high value, the pixel is considered to be on the edge
    # if the pixel is lower than the high value and the pixel is connected the edge, the pixel
    # could be considered to be on the edge

    canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
    tmp = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY_INV)
    # tmp = 255 * np.array(tmp).astype('uint8')
    IMG_OUT = cv2.cvtColor(tmp[1], cv2.COLOR_GRAY2BGR)
    return IMG_OUT
