import cv2
import numpy as np


def segmentFixed_Threshold(src, t):
    # src = np.asarray(src, dtype="float32")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src, t, 255, cv2.THRESH_BINARY)  # 阈值t
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    print(ret)
    return IMG_OUT


def segmentOTSU_Threshold(src):
    # src = np.asarray(src, dtype="float32")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    print(ret)
    return IMG_OUT


def segmentAdaptive_Threshold(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, 6)
    kernel = np.ones((5, 5), dtype=np.uint8)
    # img_open = cv2.morphologyEx(img_gray,cv.MORPH_OPEN,kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    IMG_OUT = cv2.cvtColor(img_close, cv2.COLOR_GRAY2BGR)
    return IMG_OUT
