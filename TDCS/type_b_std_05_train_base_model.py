"""
training base RNN
train_base_model.py
Created on 2018/06/06
@author ken83715
"""

import math
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

from RNNmodels2 import RNNv1, neural, ADDv2
import dataset

input_num = 5

#==========================
stdn_str = '05'
stdn_str2 = '0.5'
stdn = 0.5
#==========================

test_day = 7
data_per_day = 288

DATASET = dataset.TDCSDATA(stdn)
roaddata = DATASET.roaddata
roadtime = DATASET.roadtime
DATASET.get_cleardata_totalavg()
cleardata = DATASET.cleardata

tempmodel_path = 'TDCS/models/b_std_' + stdn_str + '_base_temp.pkl'
model_path = 'TDCS/models/b_std_' + stdn_str + '_base.pkl'
model_h_path = 'TDCS/models/b_std_' + stdn_str + '_base_h.pkl'

#train test split
train_in = []
train_exp = []
test_in = []
test_in_orig = []
test_exp = []
test_exp_orig = []

for i in range(len(cleardata) - data_per_day * test_day * 2 - input_num):
    temp = []
    for j in range(input_num + 1):
        if j < input_num:
            temp.append(cleardata[i + j])
        else:
            train_exp.append(cleardata[i + j])
    train_in.append(temp)

for i in range(len(cleardata) - data_per_day * test_day * 2, len(cleardata) - input_num):
    temp = []
    temp2 = []
    for j in range(input_num + 1):
        if j < input_num:
            temp.append(cleardata[i + j])
            temp2.append(roaddata[i + j])
        else:
            test_exp.append(cleardata[i + j])
            test_exp_orig.append(roaddata[i + j])
    test_in.append(temp)
    test_in_orig.append(temp2)

#training
bestneu = RNNv1.Neu()
besterror = 1000000000
bestneu_h = neural.Neu()
besterror_h = 1000000000

neunum = 1
epochs = 1

for n in range(neunum): #num of neural network
    pasterror = 1000000000
    hamm = neural.Neu()
    for k in range(epochs): # num of epoch
        for i in range(len(train_in)):
            try:
                y = hamm.forward(DATASET.array_to_std(train_in[i]))
                hamm.backward([DATASET.to_std(train_exp[i])])
            except:
                hamm = neural.Neu()
        
        error_h = 0
        for i in range(len(test_in) - data_per_day * test_day):
            y = hamm.forward(DATASET.array_to_std(test_in[i]))
            error_h = error_h + (test_exp[i] - DATASET.to_orig(y[0])) * (test_exp[i] - DATASET.to_orig(y[0]))
        error_h = math.sqrt(error_h / (len(test_in) - data_per_day * test_day))
        print(n, ' hamm rmse:', error_h)
        hamm.cleartemporalepoch()

        if pasterror > error_h and math.isnan(error_h) != True:
            pasterror = error_h
            hamm.saveneu(tempmodel_path)
        else:
            f = open(tempmodel_path, 'rb')
            hamm = pickle.load(f)
            f.close()
            break
    if pasterror < besterror_h:
        besterror_h = pasterror
        bestneu_h = hamm

for n in range(neunum): #num of neural network
    pasterror = 1000000000
    testneu = RNNv1.Neu()
    for k in range(epochs): # num of epoch
        for i in range(len(train_in)):
            try:
                testneu.forward(DATASET.array_to_std(train_in[i]))
                testneu.backward(DATASET.to_std(train_exp[i]) - testneu.output[0][0], DATASET.array_to_std(train_in[i]))
            except:
                testneu = RNNv1.Neu()
        
        error = 0
        for i in range(len(test_in) - data_per_day * test_day):
            testneu.forward(DATASET.array_to_std(test_in[i]))
            error = error + (test_exp[i] - DATASET.to_orig(testneu.output[0][0])) * (test_exp[i] - DATASET.to_orig(testneu.output[0][0]))
        error = math.sqrt(error / (len(test_in) - data_per_day * test_day))
        print(n, ' rmse:', error)
        testneu.cleartemporalepoch()

        if pasterror > error and math.isnan(error) != True:
            pasterror = error
            testneu.saveneu(tempmodel_path)
        else:
            f = open(tempmodel_path, 'rb')
            testneu = pickle.load(f)
            f.close()
            break
    if pasterror < besterror:
        besterror = pasterror
        bestneu = testneu

bestneu.maxd = DATASET.maxd
bestneu.mind = DATASET.mind
bestneu_h.maxd = DATASET.maxd
bestneu_h.mind = DATASET.mind
bestneu.saveneu(model_path)
bestneu_h.saveneu(model_h_path)

#testing
x = []
y_orig = []
y_mov = []
y2 = []
y3 = []
error = 0
errorlist = []
error_h = 0
errorlist_h = []

for i in range(len(test_in) - data_per_day * test_day, len(test_in)):
    bestneu.forward(DATASET.array_to_std(test_in_orig[i]))
    y = bestneu_h.forward(DATASET.array_to_std(test_in_orig[i]))

    error = error + (test_exp_orig[i] - DATASET.to_orig(bestneu.output[0][0])) * (test_exp_orig[i] - DATASET.to_orig(bestneu.output[0][0]))
    errorlist.append(abs(test_exp_orig[i] - DATASET.to_orig(bestneu.output[0][0])))

    error_h = error_h + (test_exp_orig[i] - DATASET.to_orig(y[0])) * (test_exp_orig[i] - DATASET.to_orig(y[0]))
    errorlist_h.append(abs(test_exp_orig[i] - DATASET.to_orig(y[0])))

    y_orig.append(test_exp_orig[i])
    y_mov.append(test_exp[i])
    y2.append(DATASET.to_orig(bestneu.output[0][0]))
    y3.append(DATASET.to_orig(y[0]))

error = math.sqrt(error / (data_per_day * test_day))
error_h = math.sqrt(error_h / (data_per_day * test_day))
print('final rmse:', error)
print('final hamm rmse:', error_h)

for i in range(data_per_day * test_day):
    x.append(i + 1)

plt.figure(figsize=(20,5))
plt.title('std ' + stdn_str2)
plt.plot(x, y_orig, '--', c='grey', label='act', linewidth=2.0)
plt.plot(x, y2, '-', c='black', label='basicRNN')
plt.plot(x, y3, '-.', c='black', label='Hamm')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('std ' + stdn_str2 + ' Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.xlabel('ERROR')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()