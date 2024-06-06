import sys
from sklearn.cluster import kmeans_plusplus
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QFileDialog, QButtonGroup)

from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
# from pyqtgraph.pgcollections import OrderedDict
from collections import OrderedDict
import random
import numpy as np
import time
import math
import struct
import os
# from oob_parser import uartParserSDK
from gui_threads3 import *
from graphUtilities import *
from gl_classes import GLTextItem
from data_load2 import Dataload
# from config import *
# from new_function import *
from config import PersistentFrames


class Window(QDialog):
    def __init__(self, parent=None, size=[]):
        super(Window, self).__init__(parent)
        # set window toolbar options, and title.
        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("Hlab replay demo")

        if (0):  # set to 1 to save terminal output to logFile, set 0 to show terminal output
            ts = time.localtime()
            terminalFileName = str(
                'logData/logfile_' + str(ts[2]) + str(ts[1]) + str(ts[0]) + '_' + str(ts[3]) + str(
                    ts[4]) + '.txt')
            sys.stdout = open(terminalFileName, 'w')
        #
        #   参数初始化
        self.frameTime = 50  # 这个可以控制时钟速度 来控制画图速度
        self.profile = {'startFreq': 60.25, 'numLoops': 64, 'numTx': 3, 'sensorHeight': 2,
                        'maxRange': 10, 'az_tilt': 0, 'elev_tilt': 0}
        self.Gradients = OrderedDict([
            ('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
                               (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)),
                               (0.11348314606741573, (0, 68, 255, 255)),
                               (0.6797752808988764, (255, 255, 0, 255)),
                               (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)),
                               (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
            ('summer',
             {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'}),
            ('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)),
                                 (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)),
                                 (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))],
                       'mode': 'rgb'}),
            (
                'winter',
                {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'}),
            ('spectrum2',
             {'ticks': [(1.0, (255, 0, 0, 255)), (0.0, (255, 0, 255, 255))], 'mode': 'hsv'}),
        ])
        cmap = 'spectrum2'
        self.configSent = 0
        self.graphFin = 1
        # self.frameTime = 50
        self.threeD = 1
        self.previousFirstZ = -1
        self.frameNUm = 0
        # gui size
        if (size):
            left = 50
            top = 50
            width = math.ceil(size.width() * 0.8)
            height = math.ceil(size.height() * 0.8)
            self.setGeometry(left, top, width, height)  # 左上角坐标，窗口大小

        self.previousCloud = np.zeros((6, 1150, 10))  # 6行 1150列 高10的矩阵
        self.previousPointCount = np.zeros((10, 1))  # 10行1列
        self.bbox = [-1000, 1000, -1000, 1000, -1000, 1000]
        #
        if (cmap in self.Gradients):
            self.gradientMode = self.Gradients[cmap]
        self.zRange = [-3, 3]
        #
        #   调用模块函数
        #

        self.plot3DQTGraph()
        self.colorGradient()
        self.setSaveFileLayout()
        self.setStatsLayout()
        # Setup graph pyqtgraph
        self.graphTabs = QTabWidget()
        self.graphTabs.addTab(self.pcplot, '3D Plot')
        self.graphTabs.currentChanged.connect(self.whoVisible)
        #   布局
        gridlay = QGridLayout()

        gridlay.addWidget(self.saveFileBox, 0, 0, 1, 1)
        gridlay.addWidget(self.statBox, 1, 0, 1, 1)
        gridlay.addWidget(self.graphTabs, 0, 1, 6, 1)
        gridlay.addWidget(self.gw, 0, 2, 6, 1)

        self.setLayout(gridlay)

    # def selectSaveFile:
    def setSaveFileLayout(self):

        self.saveFileBox = QGroupBox('Save file config')

        self.selectFile = QPushButton('Select file')  # 继承自abstract button 可以设置text
        self.selectFile.clicked.connect(self.selectSaveFile)  # 连接点击事件

        self.startReplay = QPushButton('Start replay')
        self.startReplay.clicked.connect(self.startApp)

        # 再生時間を変えている
        self.slowPlay = QPushButton('Slow Play')  # 继承自abstract button 可以设置text
        self.slowPlay.clicked.connect(self.setFramTime)  # 连接点击事件

        self.SaveFileLayout = QVBoxLayout()
        self.SaveFileLayout.addWidget(self.slowPlay)
        self.SaveFileLayout.addWidget(self.selectFile)
        self.SaveFileLayout.addWidget(self.startReplay)
        # self.configLayout.addStretch(1)

        self.saveFileBox.setLayout(self.SaveFileLayout)
        print("11111111")

    def setStatsLayout(self):
        # 在 画图完成后通过settext（）更新
        self.statBox = QGroupBox('Statistics')
        self.frameNumDisplay = QLabel('Frame: 0')
        self.plotTimeDisplay = QLabel('Average Plot Time: 0 ms')
        self.numPointsDisplay = QLabel('Points: 0')
        self.numTargetsDisplay = QLabel('Targets: 0')
        self.replayTime = QLabel('ReplayTime: 0:0:0:0')
        self.frameNUmt = QLabel("frame_number: "+ str(self.frameNUm))

        self.statsLayout = QVBoxLayout()

        self.statsLayout.addWidget(self.frameNumDisplay)
        self.statsLayout.addWidget(self.plotTimeDisplay)
        self.statsLayout.addWidget(self.numPointsDisplay)
        self.statsLayout.addWidget(self.numTargetsDisplay)
        self.statsLayout.addWidget(self.replayTime)
        self.statsLayout.addWidget(self.frameNUmt)

        self.statBox.setLayout(self.statsLayout)

    def selectSaveFile(self):
        try:
            self.parser = Dataload()
            self.parser.frameTime = self.frameTime
            self.uart_thread = parseUartThread(self.parser)
            self.uart_thread.fin.connect(self.updateGraph)
            self.parseTimer = QTimer()
            self.parseTimer.setSingleShot(False)
            self.parseTimer.timeout.connect(self.parseData)
            # self.parseCfg(self.selectFile())
            self.FileSelect = 1
            print("select file")

        except Exception as e:
            print(e)
            print('No cfg file selected!')

    def startApp(self):
        print("start replay")
        # self.configSent = 1
        # self.parseTimer.start(self.frameTime)  # need this line
        self.frameNUm = 0
        self.FileSelect = 1
        self.parseTimer.start(self.frameTime)  # need this line

    def setFramTime(self):
        self.frameTime = 600

    def parseData(self):
        self.uart_thread.start(priority=QThread.HighestPriority)

    # 先实现功能 再考虑布局
    def plot3DQTGraph(self):
        dummy = np.zeros((1, 3))  # 一行三列
        self.pcplot = gl.GLViewWidget()

        self.gz = gl.GLGridItem()  # 显示线框网格 实例化
        self.gz.translate(0, 0, -1 * self.profile['sensorHeight'])  # 在其父方的坐标系中通过（*dx*, *dy*, *dz*）对对象进行平移。

        self.boundaryBoxViz = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
        self.bottomSquare = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]

        for box in self.boundaryBoxViz:
            box.setVisible(False)
        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=dummy)
        self.pcplot.addItem(self.gz)
        self.pcplot.addItem(self.boundaryBoxViz[0])
        self.pcplot.addItem(self.boundaryBoxViz[1])
        self.pcplot.addItem(self.bottomSquare[0])
        self.pcplot.addItem(self.bottomSquare[1])
        self.pcplot.addItem(self.scatter)

        # create box to represent device
        verX = 0.0625
        verY = 0.05
        verZ = 0.125
        offsetZ = 0  # self.profile['sensorHeight'] #TODO
        verts = np.empty((2, 3, 3))
        verts[0, 0, :] = [-verX, 0, verZ + offsetZ]
        verts[0, 1, :] = [-verX, 0, -verZ + offsetZ]
        verts[0, 2, :] = [verX, 0, -verZ + offsetZ]
        verts[1, 0, :] = [-verX, 0, verZ + offsetZ]
        verts[1, 1, :] = [verX, 0, verZ + offsetZ]
        verts[1, 2, :] = [verX, 0, -verZ + offsetZ]
        self.evmBox = gl.GLMeshItem(vertexes=verts, smooth=False, drawEdges=True,
                                    edgeColor=pg.glColor('y'), drawFaces=False)  # 设置画device的图
        self.pcplot.addItem(self.evmBox)
        # add text items for tracks
        self.coordStr = []
        # coordinateString = GLTextItem()
        # coordinateString.setGLViewWidget(self.pcplot)
        # self.pcplot.addItem(coordinateString)
        # coordinateString.setPosition(1,1,1)
        # add mesh objects for ellipsoids
        self.ellipsoids = []
        # self.dataImages = []
        edgeColor = pg.glColor('k')
        for m in range(0, 20):
            # add track object
            mesh = gl.GLLinePlotItem()
            mesh.setVisible(False)
            self.pcplot.addItem(mesh)
            self.ellipsoids.append(mesh)
            # add track coordinate string
            text = GLTextItem()
            text.setGLViewWidget(self.pcplot)
            text.setVisible(False)
            self.pcplot.addItem(text)
            self.coordStr.append(text)



    def updateGraph(self, parsedData):
        def remove(a, b):
            w = []
            for i in range(len(a)):
                idx = 0
                if (i != b[idx]):
                    w.append(a[i])
                else:
                    idx += 1
            return w

        print('------------------------updateGraph--------------------')
        # self.useFilter = 0
        # classifierOutput = []
        # print("parsedata", parsedData)

        pointCloud = np.array(parsedData)

        """
        print("shape",pointCloud.shape)
        pointCloud = pointCloud.tolist()
        #print(pointCloud)
        #print("pointCloud",pointCloud.shape,pointCloud)
        #print(pointCloud[0,0])
        num = len(pointCloud[0])
        dl = []
        for i in range(num):
            xs = pointCloud[0][i]  # x座標
            ys = pointCloud[1][i]  # y座標
            zs = pointCloud[2][i]  # z座標
            #print(pointCloud[0])

            if (zs >= 1.5) or (zs <= -1) or (ys >= 1.7):# 1.5とすると手だけになる
                # print(zs)
                dl.append(i)
                #print("call dl", dl)

        for i in range(3):
            pointCloud[i].clear()


        pointCloud[0] = remove(pointCloud[0],dl)
        pointCloud[1] = remove(pointCloud[1], dl)
        pointCloud[2] = remove(pointCloud[2], dl)

        pointCloud2 = np.array(pointCloud)
        print("shape2",pointCloud2.shape)

        """

        if (self.graphFin):
            self.plotstart = int(round(time.time() * 1000))
            self.graphFin = 0
            if (self.threeD):
                indicesIn = []
                print("call_thread")
                self.get_thread = updateQTTargetThread3D(pointCloud, self.scatter)
                self.get_thread.done.connect(self.graphDone)
                self.get_thread.start(priority=QThread.HighestPriority - 1)

    def graphDone(self, ):
        plotend = int(round(time.time() * 1000))
        plotime = plotend - self.plotstart
        self.frameNUm += 1
        try:
            if (self.frameNum > 1):
                self.averagePlot = (plotime * 1 / self.frameNum) + (
                        self.averagePlot * (self.frameNum - 1) / (self.frameNum))
            else:
                self.averagePlot = plotime
        except:
            self.averagePlot = plotime
        self.graphFin = 1
        pltstr = 'Average Plot time: ' + str(plotime)[:5] + ' ms'
        # fnstr = 'Frame: ' + str(self.frameNum)

        # self.frameNumDisplay.setText(fnstr)
        self.plotTimeDisplay.setText(pltstr)
        self.frameNumDisplay.setText("frame_number: "+ str(self.frameNUm))

        print("graphDone")

    def colorGradient(self):
        self.gw = pg.GradientWidget(orientation='right')  # 颜色梯度的朝向
        self.gw.restoreState(self.gradientMode)

    def whoVisible(self):
        if (self.threeD):
            self.threeD = 0
        else:
            self.threeD = 1
        print('3d: ', self.threeD)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    main = Window(size=size)
    print("111")
    main.show()
    sys.exit(app.exec_())


