import time
import cv2


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
    IMG_OUT = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    print("canny")
    return IMG_OUT


def edgeDetection_Sobel(src):
    start = time.time()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(src, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(src, cv2.CV_16S, 0, 1)
    # 对图像进行自动拉伸
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    IMG_OUT = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    end = time.time()
    print("Execution Time:", end - start)
    return IMG_OUT
