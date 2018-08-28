"""
Recurrent Neural Network
RNNv1.py
Created on 2018/06/06
@author ken83715
"""

import random
import math
import pickle

class Neu:
    """
    Recurrent Neural Network
    """
    def __init__(self):
        """
        set parameters
        """
        self.inputnumber = 5
        self.hidnumber = 2
        self.outputnumber = 1

        self.learnspeed_s = 0.1
        self.learnspeed_d = 0.01

        self.error = 0

        self.maxd = 0
        self.mind = 0

        self.w1 = []
        self.d1 = []

        self.wr = []
        self.opt = []
        self.fpt = []
        self.fpt_old = []

        self.w2 = []
        self.output = []

        self.createnetwork()

    def saveneu(self, neuname):
        """
        write neu to file
        """
        f = open(neuname, 'wb')
        # dump the object to a file
        pickle.dump(self, f)
        f.close()

    def createzero(self, createlist, row, col):
        """
        create zero list
        """
        createlist = []
        for i in range(row):
            temp = []
            for j in range(col):
                temp.append(0)
            createlist.append(temp)
        return createlist

    def createrandom(self, createlist, row, col):
        """
        create random list
        """
        createlist = []
        for i in range(row):
            temp = []
            for j in range(col):
                temp2 = random.uniform(-1, 1)
                temp.append(temp2)
            createlist.append(temp)
        return createlist

    def createnetwork(self):
        """
        create the network structure
        """
        self.w1 = self.createrandom(self.w1, self.inputnumber, self.hidnumber)
        self.d1 = self.createrandom(self.d1, 1, self.hidnumber)

        self.wr = self.createrandom(self.wr, self.hidnumber, self.hidnumber)
        self.opt = self.createzero(self.opt, 1, self.hidnumber)
        self.fpt = self.createzero(self.fpt, 1, self.hidnumber)
        self.fpt_old = self.createzero(self.fpt_old, 1, self.hidnumber)

        self.w2 = self.createrandom(self.w2, self.hidnumber, self.outputnumber)
        self.output = self.createzero(self.output, 1, self.outputnumber)
        
    def cleartemporalepoch(self):
        """
        clear temp data every epoch
        """
        self.opt = self.createzero(self.opt, 1, self.hidnumber)
        self.fpt = self.createzero(self.fpt, 1, self.hidnumber)
        self.fpt_old = self.createzero(self.fpt_old, 1, self.hidnumber)
        self.output = self.createzero(self.output, 1, self.outputnumber)

    def activate(self, input):
        """
        activate function
        """
        #tanh
        up = math.exp(input) - math.exp(-input)
        down = math.exp(input) + math.exp(-input)
        return up / down
        
    def forward(self, inputlist):
        """
        calculate result
        """
        #put last time fpt in fpt_old
        for i in range(self.hidnumber):
            self.fpt_old[0][i] = self.fpt[0][i]

        #caculate new fpt
        temp = 0
        for i in range(self.hidnumber):
            #caculate opt
            temp = 0
            for j in range(self.inputnumber):
                temp = temp + inputlist[j] * self.w1[j][i]
            self.opt[0][i] = temp - self.d1[0][i]
            
            #caculate fpt
            temp = self.opt[0][i]
            for j in range(self.hidnumber):
                temp = temp + self.fpt_old[0][j] * self.wr[j][i]
            
            self.fpt[0][i] = self.activate(temp)

        #caculate output
        for i in range(self.outputnumber):
            self.output[0][i] = 0
            for j in range(self.hidnumber):
                self.output[0][i] = self.output[0][i] + self.fpt[0][j] * self.w2[j][i]

    def backward(self, error, inputlist):
        """
        adjust weight
        """
        self.error = error

        for i in range(self.outputnumber):
            for j in range(self.hidnumber):
                self.w2[j][i] = self.w2[j][i] + self.error * self.fpt[0][j] * self.learnspeed_d

        for i in range(self.hidnumber):
            for j in range(self.hidnumber):
                self.wr[i][j] = self.wr[i][j] + self.error * self.w2[j][0] * (1 - self.fpt[0][j] * self.fpt[0][j]) * self.fpt_old[0][i] * self.learnspeed_d

        for i in range(self.hidnumber):
            for j in range(self.inputnumber):
                self.w1[j][i] = self.w1[j][i] + self.error * self.w2[i][0] * (1 - self.fpt[0][i] * self.fpt[0][i]) * inputlist[j] * self.learnspeed_s
            
            self.d1[0][i] = self.d1[0][i] - self.error * self.w2[i][0] * (1 - self.fpt[0][i] * self.fpt[0][i]) * self.learnspeed_s