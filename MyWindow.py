# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MyWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(960, 960)
        font = QtGui.QFont()
        font.setKerning(True)
        MainWindow.setFont(font)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setMouseTracking(False)
        MainWindow.setTabletTracking(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("F:/Pictures/NIKE.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("QMainWindow\n"
                                 "{background-image: url(:/jpg/image/android.jpg);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(410, 510, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 10, 591, 71))
        font = QtGui.QFont()
        font.setPointSize(28)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setMouseTracking(False)
        self.label.setStyleSheet("QLabel{color:white}")
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 510, 321, 31))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setGeometry(QtCore.QRect(890, 170, 22, 160))
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.labelimg = QtWidgets.QLabel(self.centralwidget)
        self.labelimg.setGeometry(QtCore.QRect(40, 80, 821, 421))
        self.labelimg.setObjectName("labelimg")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(880, 140, 42, 22))
        self.spinBox.setObjectName("spinBox")
        self.process = QtWidgets.QTextEdit(self.centralwidget)
        self.process.setGeometry(QtCore.QRect(30, 600, 900, 300))
        self.process.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.process.setAutoFillBackground(True)
        self.process.setStyleSheet("styleSheet{color:rgb(0, 170, 255)}")
        self.process.setReadOnly(True)
        self.process.setObjectName("process")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 960, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        self.menuSVM = QtWidgets.QMenu(self.menu_5)
        self.menuSVM.setObjectName("menuSVM")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        self.menu_6.setObjectName("menu_6")
        self.menu_7 = QtWidgets.QMenu(self.menu_6)
        self.menu_7.setObjectName("menu_7")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setAutoFillBackground(False)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionOpenFile = QtWidgets.QAction(MainWindow)
        self.actionOpenFile.setObjectName("actionOpenFile")
        self.actionOpenDir = QtWidgets.QAction(MainWindow)
        self.actionOpenDir.setObjectName("actionOpenDir")
        self.actionSegmentFixedThreshold = QtWidgets.QAction(MainWindow)
        self.actionSegmentFixedThreshold.setObjectName("actionSegmentFixedThreshold")
        self.actionSegmentOTSU = QtWidgets.QAction(MainWindow)
        self.actionSegmentOTSU.setObjectName("actionSegmentOTSU")
        self.actionSegmentAdaptive = QtWidgets.QAction(MainWindow)
        self.actionSegmentAdaptive.setObjectName("actionSegmentAdaptive")
        self.actionCanny = QtWidgets.QAction(MainWindow)
        self.actionCanny.setObjectName("actionCanny")
        self.actionWatershed = QtWidgets.QAction(MainWindow)
        self.actionWatershed.setObjectName("actionWatershed")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.action162HOG = QtWidgets.QAction(MainWindow)
        self.action162HOG.setObjectName("action162HOG")
        self.action270HOG_4GLCM_26LBP = QtWidgets.QAction(MainWindow)
        self.action270HOG_4GLCM_26LBP.setObjectName("action270HOG_4GLCM_26LBP")
        self.action324HOG_4GLCM_26LBP = QtWidgets.QAction(MainWindow)
        self.action324HOG_4GLCM_26LBP.setObjectName("action324HOG_4GLCM_26LBP")
        self.action_svm_adaptive = QtWidgets.QAction(MainWindow)
        self.action_svm_adaptive.setObjectName("action_svm_adaptive")
        self.action_svm_fixed = QtWidgets.QAction(MainWindow)
        self.action_svm_fixed.setObjectName("action_svm_fixed")
        self.action_svm_OTSU = QtWidgets.QAction(MainWindow)
        self.action_svm_OTSU.setObjectName("action_svm_OTSU")
        self.action_svm_Canny = QtWidgets.QAction(MainWindow)
        self.action_svm_Canny.setObjectName("action_svm_Canny")
        self.menu.addAction(self.actionOpenFile)
        self.menu.addAction(self.actionOpenDir)
        self.menu_2.addAction(self.actionSegmentFixedThreshold)
        self.menu_2.addAction(self.actionSegmentOTSU)
        self.menu_2.addAction(self.actionSegmentAdaptive)
        self.menu_3.addAction(self.actionWatershed)
        self.menu_4.addAction(self.actionCanny)
        self.menu_4.addAction(self.actionSobel)
        self.menuSVM.addAction(self.action162HOG)
        self.menuSVM.addAction(self.action270HOG_4GLCM_26LBP)
        self.menuSVM.addAction(self.action324HOG_4GLCM_26LBP)
        self.menu_5.addAction(self.menuSVM.menuAction())
        self.menu_7.addAction(self.action_svm_adaptive)
        self.menu_7.addAction(self.action_svm_fixed)
        self.menu_7.addAction(self.action_svm_OTSU)
        self.menu_7.addAction(self.action_svm_Canny)
        self.menu_6.addAction(self.menu_7.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "myWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "??????"))
        self.label.setText(_translate("MainWindow", "????????????????????????????????????"))
        self.labelimg.setText(_translate("MainWindow", "TextLabel"))
        self.menu.setTitle(_translate("MainWindow", "??????"))
        self.menu_2.setTitle(_translate("MainWindow", "????????????"))
        self.menu_3.setTitle(_translate("MainWindow", "????????????"))
        self.menu_4.setTitle(_translate("MainWindow", "????????????"))
        self.menu_5.setTitle(_translate("MainWindow", "?????????"))
        self.menuSVM.setTitle(_translate("MainWindow", "SVM"))
        self.menu_6.setTitle(_translate("MainWindow", "????????????"))
        self.menu_7.setTitle(_translate("MainWindow", "??????-??????"))
        self.actionOpenFile.setText(_translate("MainWindow", "open file(&F)"))
        self.actionOpenDir.setText(_translate("MainWindow", "open dir(&D)"))
        self.actionSegmentFixedThreshold.setText(_translate("MainWindow", "??????????????????"))
        self.actionSegmentOTSU.setText(_translate("MainWindow", "?????????"))
        self.actionSegmentAdaptive.setText(_translate("MainWindow", "?????????????????????"))
        self.actionCanny.setText(_translate("MainWindow", "Canny"))
        self.actionWatershed.setText(_translate("MainWindow", "???????????????"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.action162HOG.setText(_translate("MainWindow", "162HOG+4GLCM+26LBP"))
        self.action270HOG_4GLCM_26LBP.setText(_translate("MainWindow", "270HOG+4GLCM+26LBP"))
        self.action324HOG_4GLCM_26LBP.setText(_translate("MainWindow", "324HOG+4GLCM+26LBP"))
        self.action_svm_adaptive.setText(_translate("MainWindow", "?????????????????????"))
        self.action_svm_fixed.setText(_translate("MainWindow", "??????????????????"))
        self.action_svm_OTSU.setText(_translate("MainWindow", "OTSU?????????"))
        self.action_svm_Canny.setText(_translate("MainWindow", "Canny????????????"))


import img_rc
