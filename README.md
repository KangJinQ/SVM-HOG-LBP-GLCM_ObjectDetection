这是一次大作业。
浅浅完成了连铸板坯低倍组织缺陷分割技术研究。
其中用到了SVM支持向量机、HOG方向梯度直方图、LBP局部二值模式和GLCM灰度共生矩阵特征提取，
使用滑动窗口的办法进行逐行检测。
在SVM识别出矩形框后，会进一步使用nms非极大值抑制提升精度。
还有很多经典的图像处理方法，在src目录中。
其中配备了GUI界面。

其中，你要首先配置好环境
这里有你需要的Python包：
numpy,opencv-python 4.2.0.32,PyQt5,pyqt5-plugins,pyqt5-Qt5,pyqt5-tools,
PyQt5Designer,qt5-tools,qt5-applications,scikit-learn,scikit-image等等。
需要哪个就添加哪个好了。