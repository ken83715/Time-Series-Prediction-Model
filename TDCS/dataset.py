"""
dataset and info
dataset.py
Created on 2018/08/28
@author ken83715
"""

import csv
import numpy as np

class TDCSDATA:
    """
    TDCS datainfo
    """
    def __init__(self, std_r):
        """
        set parameters
        """
        self.filepath = 'D:/python/jupyter/data/TDCSDIVIDEBYSEG/01F0467S-01F0509S.csv'
        self.roaddata = []
        self.roadtime = []
        self.std = []
        self.avg = []
        self.cleardata = []
        self.maxd = 0
        self.mind = 0
        self.std_range = std_r
        self.data_per_day = 288

        self.readdata()
        self.calculate_std_avg()

    def readdata(self):
        with open(self.filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row[18] != 'Null':
                    self.roaddata.append(int(row[18]))
                else:
                    self.roaddata.append(193)
                t1 = row[20].split(':')
                t2 = int((int(t1[0]) * 60 + int(t1[1])) / 5)
                self.roadtime.append(t2)
        f.close()
    
    def calculate_std_avg(self):
        dtime = []
        for i in range(self.data_per_day):
            dtime.append([])
        for i in range(len(self.roaddata)):
            dtime[self.roadtime[i]].append(self.roaddata[i])

        for t in dtime:
            self.std.append(np.std(np.array(t)))
            self.avg.append(np.mean(np.array(t)))

    def normaldata(self, data, avg, std):
        if data > avg - self.std_range * std and data < avg + self.std_range * std:
            return True
        else:
            return False
    
    def get_cleardata_delete(self):
        for i in range(len(self.roaddata)):
            if self.normaldata(self.roaddata[i], self.avg[self.roadtime[i]], self.std[self.roadtime[i]]) == True:
               self.cleardata.append(self.roaddata[i])
        self.maxd = max(self.cleardata)
        self.mind = min(self.cleardata)

    def get_cleardata_movingavg(self):
        for i in range(len(self.roaddata)):
            if self.normaldata(self.roaddata[i], self.avg[self.roadtime[i]], self.std[self.roadtime[i]]) == True:
                self.cleardata.append(self.roaddata[i])
            else:
                s = self.roaddata[i - 1] + self.roaddata[i - 2] + self.roaddata[i - 3] + self.roaddata[i - 4] + self.roaddata[i - 5]
                self.cleardata.append(int(s / 5))
        self.maxd = max(self.cleardata)
        self.mind = min(self.cleardata)

    def get_cleardata_totalavg(self):
        for i in range(len(self.roaddata)):
            if self.normaldata(self.roaddata[i], self.avg[self.roadtime[i]], self.std[self.roadtime[i]]) == True:
                self.cleardata.append(self.roaddata[i])
            else:
                self.cleardata.append(self.avg[self.roadtime[i]])
        self.maxd = max(self.cleardata)
        self.mind = min(self.cleardata)

    #resize to -1 ~ 1
    def to_std(self, x):
        return 2 * (x - self.mind) / (self.maxd - self.mind) - 1

    def array_to_std(self, x):
        newx = []
        for i in x:
            newx.append(2 * (i - self.mind) / (self.maxd - self.mind) - 1)
        return newx

    def to_orig(self, x):
        return (x + 1) / 2 * (self.maxd - self.mind) + self.mind