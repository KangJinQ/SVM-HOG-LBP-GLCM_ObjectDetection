import cv2
import numpy as np


def water_Shed(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 阈值分割，将图像分为黑白两部分
    # 使用OTSU算法找寻分割阈值，并且像素值反转（白色背景变为黑色背景）
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV
                                + cv2.THRESH_OTSU)

    # 对图像进行开运算，先腐蚀再膨胀，消除孤立点或毛刺
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                               kernel, iterations=2)

    # 对开运算的结果进行膨胀，得到大部分都是背景的区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step5.通过distanceTransform获取前景区域
    # 距离变换的处理图像通常都是二值图像，
    # 而二值图像其实就是把图像分为两部分，即背景和物体
    # 通常把前景目标的灰度值设为255，即白色，背景的灰度值设为0，即黑色
    # 所以定义中的非零像素点即为前景目标，零像素点即为背景
    # 图像中前景目标中的像素点距离背景越远，那么距离就越大
    # 如果用这个距离值替换像素值，那么新生成的图像中这个点越亮

    # DIST_L2,欧式距离 距离变换掩码矩阵大小，5
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 转成uint8
    dist_transformImg = cv2.convertScaleAbs(10 * dist_transform)
    # 固定阈值：0.1 * dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域
    # 此区域和轮廓区域的关系未知
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)

    # 设置标记的作用是为了防止分水岭的过分割
    # 连通区域处理
    # 对连通区域进行标号  序号为 0 ~ N-1
    # 8邻接

    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    # 分水岭算法对物体做的标注必须都 大于1 ，背景为标号为0
    # 对所有markers 加1  变成了  1 ~ N
    print("markers")
    # ret 返回连通区域数量
    print(ret)
    print(markers)
    print("markers+1")
    markers = markers + 1
    print(markers)
    # 去掉属于背景区域的部分（即让其变为0，成为背景）
    # 此语句的Python语法 类似于if ，“unknow==255” 返回的是图像矩阵的真值表。
    markers[unknow == 255] = 0
    print(markers)

    # Step8.分水岭算法
    # 分水岭算法后，所有轮廓的像素点被标注为  -1
    markers = cv2.watershed(src, markers)
    # print(markers)

    # 标注为-1 的像素点标红
    src[markers == -1] = [255, 0, 0]
    return src
