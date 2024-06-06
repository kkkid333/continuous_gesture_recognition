from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPainter, QColor, QFont

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import random
import numpy as np
import time
from sklearn.cluster import kmeans_plusplus

#from oob_parser import uartParserSDK
from graphUtilities import *

class parseUartThread(QThread):
    fin = pyqtSignal('PyQt_PyObject')

    def __init__(self, uParser):
        QThread.__init__(self)
        self.parser = uParser

    def run(self):
        pointCloud = self.parser.readAndParseUart()
        #print(pointCloud)
        self.fin.emit(pointCloud)

class updateQTTargetThread3D(QThread):
    done = pyqtSignal('PyQt_PyObject')

    def __init__(self, pointCloud, scatter):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        #.targets = targets
        #self.indexes = indexes
        self.scatter = scatter
        self.colorArray = ('r','g','b','w')



    def run(self):
        #print("pointcloud",self.pointCloud)
        #graph the points with colors
        try:
            if(1):
                toPlot = self.pointCloud[0:3, :].transpose()#N*3に変更
                # print("toPlot",toPlot,toPlot.shape)
                #size = np.log2(self.pointCloud[4, :].transpose())
                colors = np.zeros((np.shape(self.pointCloud)[1], 4))
                for i in range(np.shape(self.pointCloud)[1]):
                    xs = self.pointCloud[0, i]  # x座標
                    ys = self.pointCloud[1, i]  # y座標
                    zs = self.pointCloud[2, i]  # z座標
                    dp = self.pointCloud[3, i]
                    #print(dp)
                    if abs(dp) <= 0.03:
                        colors[i] = pg.glColor("blue")
                    elif (abs(dp) >= 0.07) and (abs(dp) <= 1):
                        colors[i] = pg.glColor("y")
                    elif (abs(dp) >= 1) and (abs(dp) <= 2):
                        colors[i] = pg.glColor("m")
                    elif (abs(dp) >= 2):
                        colors[i] = pg.glColor("r")
                    elif (abs(dp) >= 0.03) and (abs(dp) <= 0.07):
                        colors[i] = pg.glColor("g")

                    """
                    if (zs <= 1.5) and (zs >= -1):
                        # print(zs)
                        colors[i] = pg.glColor("g")
                        if (ys >= 2.0):  # 1.5とすると手だけになる
                            colors[i] = pg.glColor("w")
                    else:
                        colors[i] = pg.glColor("w")
                    """
            else:
                co_li = ["b","g","r","c","m","y","orange","pink","cyan","brown"]
                colors = np.zeros((np.shape(self.pointCloud)[1], 4))
                toPlot = self.pointCloud[0:3, :].transpose()
                km = kmeans_plusplus(n_clusters=10).fit(toPlot)
                l = km.labels_
                for i in range(np.shape(self.pointCloud)[1]):

                    if(l == 0):
                        colors[i] = pg.glColor(co_li[l])
                    elif(l == 1):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 2):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 3):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 4):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 5):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 6):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 7):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 8):
                        colors[i] = pg.glColor(co_li[l])
                    elif (l == 9):
                        colors[i] = pg.glColor(co_li[l])


            self.scatter.setData(pos=toPlot, color=colors, size=15)
        except Exception as e:
            print(e)



        #graph the targets
        targetInfor = [0,0,0,0,0,0]
        self.done.emit(targetInfor)