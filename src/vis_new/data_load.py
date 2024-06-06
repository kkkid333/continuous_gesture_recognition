import struct
import sys
import serial
import binascii
import time
import numpy as np
import math

from graphUtilities import rotX

import os
import datetime
#from dataupdate import pointchange
#from config import SAVE_DATA,Replay
from savedata.replay_main import get_data
if_start = 0

class Dataload():
    def __init__(self, ):
        self.fail = 0
        self.get_data1 = get_data()
        self.if_start = False
    def readAndParseUart(self):
        print(self.if_start)
        if self.if_start == False:
            self.get_data1.start()
            self.if_start = True
        else:
            data = []
            data = self.get_data1.get_data()
            print(type(data))
            load_dict = data
            # print(load_dict)
            # print(type(load_dict))

            PointCloud = load_dict['PointCloud']
            #print("pointcloud", PointCloud)
            pcBufPing = np.array(PointCloud)
            #print("pcBufPing", pcBufPing)

            targetBufPing = load_dict['Targets']
            targetBufPing = np.array(targetBufPing)

            indexes = load_dict['Indexes']
            indexes = list(indexes)

            numDetectedObj = load_dict['NumPoints']
            numDetectedObj = int(numDetectedObj)

            numDetectedTarget = load_dict['NumTargets']
            numDetectedTarget = int(numDetectedTarget)

            frameNum = load_dict['FrameNum']
            frameNum = int(frameNum)

            fail = load_dict['Fail']
            fail = int(fail)

            classifierOutput = load_dict['ClassifierOutput']
            classifierOutput = list(classifierOutput)

            print("readAndParseUart")
            # print(pcBufPing, targetBufPing, indexes, numDetectedObj, numDetectedTarget, frameNum,
            #       fail, classifierOutput)
            #return

            replay_time = load_dict['Now_time']

            if pcBufPing is not None:
                return pcBufPing, targetBufPing, indexes, numDetectedObj, numDetectedTarget, frameNum, fail, classifierOutput, replay_time

