"""
Additional model
ADDv1.py
Created on 2018/06/06
@author ken83715
"""

import random
import math
import pickle

class Add_model:
    """
    Additional model
    """
    def __init__(self):
        """
        set parameters
        """
        self.inputnumber = 5
        self.outputnumber = 1

        self.maxd = 0
        self.mind = 0

        self.w1 = []
        self.w2 = []
        self.wr = []
        self.lasthidout = []
        self.output = []

        self.createnetwork()

    def savemodel(self, modelname):
        """
        write model to file
        """
        f = open(modelname, 'wb')
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
                temp2 = random.uniform(-16, 16)
                temp.append(temp2)
            createlist.append(temp)
        return createlist

    def createnetwork(self):
        """
        create the network structure
        """
        self.w1 = self.createrandom(self.w1, 1, self.inputnumber)
        self.w2 = self.createrandom(self.w2, 1, self.outputnumber)
        self.wr = [[random.uniform(-1, 1)]]
        self.output = self.createzero(self.output, 1, self.outputnumber)
        self.lasthidout = self.createzero(self.lasthidout, 1, 1)

    def activate(self, input):
        """
        activate function
        """
        #tanh
        try:
            up = math.exp(input) - math.exp(-input)
            down = math.exp(input) + math.exp(-input)
            return up / down
        except OverflowError:
            print(input)

    def forward(self, inputlist):
        """
        calculate result
        """
        #caculate output
        for i in range(self.outputnumber):
            temp = 0
            for j in range(self.inputnumber):
                temp = temp + inputlist[j] * self.w1[0][j]
            temp = temp + self.lasthidout[0][0] * self.wr[0][0]
            temp = self.activate(temp)
            self.lasthidout[0][0] = temp
            self.output[0][i] = temp * self.w2[0][i]