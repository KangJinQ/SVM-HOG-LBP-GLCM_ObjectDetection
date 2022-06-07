# Import the required modules
from skimage.io import imread
import joblib
import cv2
import numpy as np
import argparse as ap
from nms import nms
from config import *
from image_processing import *
import time
import datetime

def sliding_window(image, window_size, step_size):
    res = []
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
           res.append((x, y, image[y:y + window_size[1], x:x + window_size[0]]))
    return res

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def exe_det(img_path):
    print("开始检测..................")
    # img_path = r'F:\code\Labelimg\labelImg\data\test\16_remap.png'
    print(img_path)

    cur_time = get_datetime_str()
    res_path = r"F:\code\Labelimg\labelImg\data\res\res"+cur_time+".png"
    im = imread(img_path, as_gray=False)
    im = adjust_gamma(im, gamma=1.6)
    # print(im.shape)
    min_wdw_sz = (60, 30)
    step_size = (20, 10)
    downscale = 1
    model_version = "svm162lbp"
    model_path = "F:\code\Labelimg\labelImg\ddet\object_detector_new\data\models\\" + model_version + ".model"
    visualize_det = False

    clf = joblib.load(model_path)
    detections = []
    scale = 0
    cd = []
    # print(123)
    for (x, y, im_window) in sliding_window(im, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        fd = get_features(im_window)    #原始
        fd = fd.reshape(1, -1)
        pred = clf.predict(fd)
        if pred == 1:
            print("Detection:: Location -> ({}, {})".format(x, y))
            print("Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd)))
            detections.append((x, y, clf.decision_function(fd),
                int(min_wdw_sz[0]*(downscale**scale)),
                int(min_wdw_sz[1]*(downscale**scale))))
            cd.append(detections[-1])
        if visualize_det:
            clone = im.copy()
            for x1, y1, _, _, _ in cd:
                cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                    im_window.shape[0]), (0, 0, 0), thickness=2)
            cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                im_window.shape[0]), (255, 255, 255), thickness=2)
            cv2.imshow("Sliding Window in Progress", clone)
            cv2.waitKey(30)
    scale += 1
    clone = im.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
    print(im.shape)
    detections = nms(detections, threshold)
    for (x_tl, y_tl, cs, w, h) in detections:
        # Draw the detections
        if cs[0] < 0.04:
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 48, 48), thickness=2)
        else:
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
        text = 'Score:'+str(round(cs[0],2))
        cv2.putText(clone,text,(x_tl,y_tl-5),0,0.4,(255,0,0),1)

    print(detections)
    cv2.imwrite(res_path, clone)
    return res_path
    # cv2.waitKey(0)
#



def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y%m%d')
    time_str = cur_time.strftime('%H%M%S')

    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + '_' + time_str

if __name__ == '__main__':
    # exe_det(r"F:\code\Labelimg\labelImg\data\test\16_remap.png")
    exe_det(r"F:\code\Labelimg\labelImg\ddet\object_detector_new\data\images\test\13_remap.png")
