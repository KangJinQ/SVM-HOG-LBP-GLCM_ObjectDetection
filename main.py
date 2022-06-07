from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtWidgets import QSlider

import classic_segment
from MyWindow import Ui_MainWindow
import image_processing

'''
控制台输出定向到Qtextedit中
'''


class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    input_pixmap = None  # QPixmap Object Image
    input_image = None  # QImage Object (0~1 사이 값)
    input_array = None  # npArray Image (0~255 사이 값)
    output_pixmap = None
    output_image = None
    output_array = None
    viewRow = 421
    viewCol = 821

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self) \
            # 此处编写业务逻辑代码

        # Custom output stream
        sys.stdout = Stream(newText=self.onUpdateText)

        self.labelimg.setStyleSheet("QLabel{border:2px solid rgb(200, 200, 200);}");
        # 显示时间
        # timer = QtCore.QTimer()
        self.label_message = QtWidgets.QLabel()
        self.label_message.setStyleSheet("color:rgb(255,255,255)")
        self.statusBar.addWidget(self.label_message)
        self.label_permanent_message = QtWidgets.QLabel()
        self.label_permanent_message.setText('版权所有：')
        self.label_permanent_message.setStyleSheet("color:rgb(255,255,255)")
        self.statusBar.addPermanentWidget(self.label_permanent_message)
        # 为每个菜单项绑定triggered信号并链接槽函数
        self.actionOpenFile.triggered.connect(self.openImage)
        self.actionOpenDir.triggered.connect(self.openDir)
        self.actionSegmentFixedThreshold.triggered.connect(self.segmentFixedThreshold)
        self.actionSegmentOTSU.triggered.connect(self.segmentOTSU)
        self.actionSegmentAdaptive.triggered.connect(self.segmentAdaptive)
        self.actionCanny.triggered.connect(self.edgeDetectionCanny)
        self.actionSobel.triggered.connect(self.edgeDetectionSobel)
        self.actionWatershed.triggered.connect(self.watershed)
        # self.actionSVM.triggered.connect(self.svm)
        self.action162HOG.triggered.connect(lambda: self.svm(model_version="162"))
        self.action270HOG_4GLCM_26LBP.triggered.connect(lambda: self.svm(model_version="270"))
        self.action324HOG_4GLCM_26LBP.triggered.connect(lambda: self.svm(model_version="324"))
        # self.actionMy.triggered.connect(self.segmentWithDetection)
        self.action_svm_adaptive.triggered.connect(lambda: self.segmentWithDetection(method="adaptive"))
        self.action_svm_fixed.triggered.connect(lambda: self.segmentWithDetection(method="fixed"))
        self.action_svm_OTSU.triggered.connect(lambda: self.segmentWithDetection(method="OTSU"))
        self.action_svm_Canny.triggered.connect(lambda: self.segmentWithDetection(method="Canny"))
        # 阈值滑块以及微调窗口
        self.verticalSlider.setMinimum(1)  # 最小值
        self.verticalSlider.setMaximum(254)  # 最大值
        self.verticalSlider.setSingleStep(1)  # 步长
        self.verticalSlider.setTickPosition(QSlider.TicksAbove)  # 设置刻度位置，在上方
        self.verticalSlider.setTickInterval(5)  # 设置刻度间隔
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(254)
        self.spinBox.setSingleStep(1)
        # self.spinBox.setStyleSheet("background-color:white;color:blue;font-size:14px;")
        self.verticalSlider.valueChanged.connect(lambda: self._splider_change())  # 滑块的connect
        self.spinBox.valueChanged.connect(lambda: self._spinbox_change())  # 微调框的connect

    def _splider_change(self):
        self.spinBox.setValue(self.verticalSlider.value())

    def _spinbox_change(self):
        self.verticalSlider.setValue(self.spinBox.value())

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def openImage(self):
        print("open")
        image_processing.open_image_file(self)

    def openDir(self):
        print("open dir")
        image_processing.open_dir(self)
        # self.labelimg.setPixmap(QPixmap())

    def displayOutputImage(self):
        image_processing.displayOutputImage(self)

    def segmentFixedThreshold(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        t = int(self.spinBox.value())
        print("阈值为:", t)
        classic_segment.segmentFixedThreshold(self, t)
        print("finish segment")
        self.label_message.setText("finish segment!")
        self.displayOutputImage()
        print("fixed")

    def segmentOTSU(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.segmentOTSU(self)
        print("finish OTSU segment")
        self.label_message.setText("finish OTSU segment!")
        self.displayOutputImage()

    def segmentAdaptive(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.segmentAdaptive(self)
        print("finish adaptive segment")
        self.label_message.setText("finish adaptive segment!")
        self.displayOutputImage()

    def edgeDetectionCanny(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.edgeDetectionCanny(self)
        print("finish Canny detection")
        self.label_message.setText("finish canny segment!")
        self.displayOutputImage()

    def edgeDetectionSobel(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.edgeDetectionSobel(self)
        print("finish Sobel detection")
        self.label_message.setText("finish sobel segment!")
        self.displayOutputImage()

    def watershed(self):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.waterShed(self)
        print("finish watershed")
        self.label_message.setText("finish watershed!")
        self.displayOutputImage()

    def svm(self, model_version):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        classic_segment.svmDetection(self, model_version)
        print("finish svm162 detection")
        self.label_message.setText("finish svm detection!")
        self.displayOutputImage()

    def segmentWithDetection(self, method):
        if self.input_array is None:
            print("please input a image!")
            self.label_message.setText("please input a image!")
            return
        t = int(self.spinBox.value())
        classic_segment.segmentWithDetection(self, method, t)
        print("finish segment with detection")
        self.label_message.setText("finish segment with detection!")
        self.displayOutputImage()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
