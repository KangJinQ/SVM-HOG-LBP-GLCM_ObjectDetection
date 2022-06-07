# Import the required modules
import os

import joblib
from src.nms import nms
from src.featuresExtract import *


def sliding_window(image, window_size, step_size):
    """
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    """

    res = []
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            res.append((x, y, image[y:y + window_size[1], x:x + window_size[0]]))
    return res


def svm_Detection(im, model_version):
    print(im.shape)
    min_wdw_sz = (60, 30)

    step_size = (20, 10)
    downscale = 1

    path = os.getcwd()  # 获取当前路径
    print(path)
    model_path = path + "\\src\\svm" + model_version + "lbp.model"
    visualize_det = False
    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate
    # for im_scaled in pyramid_gaussian(im, downscale=downscale):
    # This list contains detections at the current scale
    cd = []
    # If the width or height of the scaled image is less than
    # the width or height of the window, then end the iterations.
    # if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
    # break
    # for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
    for (x, y, im_window) in sliding_window(im, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the features

        fd = get_features_pre(im_window, model_version)  # 原始
        fd = fd.reshape(1, -1)
        pred = clf.predict(fd)
        if pred == 1:
            print("Detection:: Location -> ({}, {})".format(x, y))
            print("Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd)))
            detections.append((x, y, clf.decision_function(fd),
                               int(min_wdw_sz[0] * (downscale ** scale)),
                               int(min_wdw_sz[1] * (downscale ** scale))))
            # print(detections[-1])
            cd.append(detections[-1])
        # # If visualize is set to true, display the working
        # # of the sliding window
        # if visualize_det:
        #     # clone = im_scaled.copy()
        #     clone = im.copy()
        #     for x1, y1, _, _, _ in cd:
        #         # Draw the detections at this scale
        #         cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
        #                                         im_window.shape[0]), (0, 0, 0), thickness=2)
        #     cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
        #                                   im_window.shape[0]), (255, 255, 255), thickness=2)
        #     cv2.imshow("Sliding Window in Progress", clone)
        #     cv2.waitKey(30)
    # Move the next scale
    # scale += 1
    print(len(detections))

    threshold = .3
    detections = nms(detections, threshold)
    print("共检测出缺陷部分{}处".format(len(detections)))
    # Display the results after performing NMS
    clone = im.copy()
    for (x_tl, y_tl, cs, w, h) in detections:
        if cs[0] < 0.04:
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 48, 48), thickness=2)
            text = 'Score:' + str(round(cs[0], 2))
            cv2.putText(clone, text, (x_tl, y_tl - 5), 0, 0.4, (255, 0, 0), 1)
        else:
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    OUT_IMG = np.array(clone, dtype=np.uint8)
    return OUT_IMG
