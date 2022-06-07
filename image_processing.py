import math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

import numpy as np


def open_image_file(self):
    """
    Load image from image file path.
    """
    # 1. 读入图片名字
    filename = QFileDialog.getOpenFileName()
    print(filename)
    if filename[0] is '':
        self.label_message.setText("请打开图片")
        print("no picture")
        return

    # 居中显示图片
    self.labelimg.setAlignment(Qt.AlignCenter)
    self.labelimg.setPixmap(QPixmap(filename[0]).scaled(self.labelimg.width(), self.labelimg.height(),
                                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
    # 2. 图像 -> QPixmap
    self.input_pixmap = QPixmap(filename[0])
    nCol = width = self.input_pixmap.width()
    nRow = height = self.input_pixmap.height()
    # 3. QPixmap -> QImage : 获取像素值
    self.input_image = self.input_pixmap.toImage()
    # 4. 作为numpy数组放入input数组
    channel = 3
    self.input_array = np.zeros((channel, nRow, nCol)).copy()

    for r in range(nRow):
        for c in range(nCol):
            val = self.input_image.pixel(c, r)
            # pixmap -> image -> pixel -> rgb
            colors = QColor(val).getRgbF()
            for RGB in range(channel):
                self.input_array[RGB, r, c] = int(colors[RGB] * 255)
    # if self.input_image.width() > self.viewCol or self.input_image.height() > self.viewRow:
    #     # pm = cv_image.get_fixed_pixmap(self, self.input_array)
    #     pm = get_fixed_pixmap(self, self.input_array)
    #     self.labelimg.setPixmap(pm)
    # else:
    #     self.labelimg.setPixmap(self.input_pixmap)
    # self.setStatusTip("输入图像大小: " + str(nCol) + " x " + str(nRow))  # 输入图像大小
    self.label_message.setText("成功打开图片, 输入图像大小: " + str(nCol) + " x " + str(nRow))


def get_fixed_pixmap(self, inputImageArray):
    """
    Resize displayed image.
    """
    # promise me that being my son

    # 将Array类型的图像输出到Label：Array->QImage->QPixmap->输出
    (RGB, input_image_h, input_image_w) = inputImageArray.shape
    color = QColor()  # 进入QImage的值
    image_ratio_w = self.viewCol / input_image_w
    image_ratio_h = self.viewRow / input_image_h
    if image_ratio_w < image_ratio_h:
        image_ratio_h = image_ratio_w
    else:
        image_ratio_w = image_ratio_h
    new_c = int(input_image_w * image_ratio_w)
    new_r = int(input_image_h * image_ratio_h)
    viewImage = QImage(new_c, new_r, QImage.Format_RGB888)

    # 预先计算好的self。将outputArray转换为QImage，即PyQt的对象。
    for r in range(new_r):  # r -> x , c -> y
        for c in range(new_c):
            # rgb -> pixel -> image -> pixmap
            r_ = math.floor(r / image_ratio_h)
            c_ = math.floor(c / image_ratio_w)
            try:
                color.setRgbF(inputImageArray[0][r_][c_] / 255,
                              inputImageArray[1][r_][c_] / 255,
                              inputImageArray[2][r_][c_] / 255)
                viewImage.setPixel(c, r, color.rgb())
            except:
                continue
    viewPixmap = QPixmap.fromImage(viewImage)
    return viewPixmap


def open_dir(self):
    filename = QFileDialog.getOpenFileName()
    print(filename)
    self.labelimg.setPixmap(QPixmap(filename[0]).scaled(self.labelimg.width(), self.labelimg.height(),
                                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
    # self.input_image =
    # self.labelimg.repaint()
    # self.labelimg.setScaledContents(True)  # 图片大小与label适应，否则图片可能显示不全
    print("success open")


def displayOutputImage(self, nRow=0, nCol=0):
    """
    Display output image.
    """
    (RGB, nRow, nCol) = self.output_array.shape
    color = QColor()  # pixel
    self.output_image = QImage(nCol, nRow, QImage.Format_RGB888)

    # self.labelimg.setPixmap(self.output_image.scaled(self.labelimg.width(), self.labelimg.height(),
    #                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))
    for r in range(nRow):  # r -> x , c -> y
        for c in range(nCol):
            # rgb -> pixel -> image -> pixmap
            color.setRgbF(self.output_array[0][r][c] / 255,
                          self.output_array[1][r][c] / 255,
                          self.output_array[2][r][c] / 255)  # rgb -> pixel
            # color Pixel -> Image[x,y]
            self.output_image.setPixel(c, r, color.rgb())
    self.output_pixmap = QPixmap.fromImage(self.output_image)
    self.labelimg.setPixmap(self.output_pixmap.scaled(self.labelimg.width(), self.labelimg.height(),
                                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))
    # if (self.output_image.width() > self.viewCol or
    #         self.output_image.height() > self.viewRow):
    #     pm = get_fixed_pixmap(self, self.output_array)
    #     self.labelimg.setPixmap(pm)
    # else:
    #     self.labelimg.setPixmap(self.output_pixmap)
    # self.output_array = self.output_array.astype(np.int16)
    # self.label_message.setText("成功打开图片, 输入图像大小: " + str(nCol) + " x " + str(nRow))
    print("输出图像大小: " + str(nCol) + " x " + str(nRow))
