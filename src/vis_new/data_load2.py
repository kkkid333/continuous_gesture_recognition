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
from savedata2.replay_main import get_data
if_start = 0

class Dataload():
    def __init__(self, ):
        self.fail = 0
        self.get_data1 = get_data()
        self.if_start = False
    def readAndParseUart(self):
        #print(self.if_start)
        print("call_dataload")
        if self.if_start == False:
            self.get_data1.start()
            self.if_start = True
        else:
            data = []
            data = self.get_data1.get_data()
            #print(type(data))
            load_dict = data
            # print(load_dict)
            # print(type(load_dict))

            PointCloud = load_dict['PointCloud']
            #print("pointcloud",PointCloud)

            pcBufPing = np.array(PointCloud)
            #print("pcBufPing", pcBufPing)

            if pcBufPing is not None:
                return pcBufPing

