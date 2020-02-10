"""
Additional model
Created on 2018/06/06
"""

import random
import math
import pickle

class AddModel:
    """
    Additional model
    """

    def __init__(self, input_number):
        """
        set parameters
        """

        self.fuzzy_class = 3
        self.inputnumber = input_number
        self.fuzzy1 = self.inputnumber * self.fuzzy_class
        self.de_fuzzy1 = self.fuzzy_class
        self.outputnumber = 1

        self.maxd = 0
        self.mind = 0

        self.fuzzy_c = []
        self.fuzzy_sig = []
        self.w0 = []
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

    def createrandom(self, createlist, row, col, a, b):
        """
        create random list
        """

        createlist = []
        for i in range(row):
            temp = []
            for j in range(col):
                temp2 = random.uniform(a, b)
                temp.append(temp2)
            createlist.append(temp)
        return createlist

    def createnetwork(self):
        """
        create the network structure
        """

        self.fuzzy_c = self.createrandom(self.fuzzy_c, 1, self.fuzzy1, -1, 1)
        self.fuzzy_sig = self.createrandom(self.fuzzy_sig, 1, self.fuzzy1, 0, 1)
        self.w0 = self.createrandom(self.w0, 1, self.fuzzy1, -4, 4)
        self.w1 = self.createrandom(self.w1, 1, self.de_fuzzy1, -4, 4)
        self.w2 = self.createrandom(self.w2, 1, self.outputnumber, -4, 4)
        self.wr = self.createrandom(self.wr, 1, self.outputnumber, -1, 1)
        self.output = self.createzero(self.output, 1, self.outputnumber)
        self.lasthidout = self.createzero(self.lasthidout, 1, 1)

    def activate(self, inputn):
        """
        activate function
        """

        #tanh
        up = math.exp(inputn) - math.exp(-inputn)
        down = math.exp(inputn) + math.exp(-inputn)
        return up / down

    def gaussianmf(self, inputn, sig, c):
        """
        gaussion membership function
        """

        try:
            up = -(c - inputn) * (c - inputn)
            down = 2 * sig * sig
            return math.exp(up / down)
        except OverflowError:
            print(inputn)

    def forward(self, inputlist):
        """
        calculate result
        """

        #caculate fuzzy output
        gaussian_out = []
        for i in range(self.inputnumber):
            for j in range(self.fuzzy_class):
                gaussian_out.append(self.gaussianmf(inputlist[i], self.fuzzy_sig[0][self.fuzzy_class * i + j], self.fuzzy_c[0][self.fuzzy_class * i + j]))
        
        #decode fuzzy
        decode_fuz = []
        for i in range(self.fuzzy_class):
            temp = 0
            for j in range(self.inputnumber):
                temp = temp + self.w0[0][i + j * self.fuzzy_class]
            decode_fuz.append(temp)
        
        #output
        for i in range(self.outputnumber):
            temp2 = 0
            for j in range(self.de_fuzzy1):
                temp2 = temp2 + decode_fuz[j] * self.w1[0][j]
            temp2 = temp2 + self.lasthidout[0][0] * self.wr[0][0]
            temp2 = self.activate(temp2)
            self.lasthidout[0][0] = temp2
            self.output[0][i] = temp2 * self.w2[0][i]