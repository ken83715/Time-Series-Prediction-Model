"""
training base RNN delete out of range data
train_base_model.py
Created on 2018/06/06
@author ken83715
"""

import math
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

from RNNmodels2 import RNNv1, neural
import dataset

input_num = 5
stdn_str = '20'
stdn = 2
test_day = 7
data_per_day = 288

DATASET = dataset.TDCSDATA(stdn)
roaddata = DATASET.roaddata
roadtime = DATASET.roadtime
DATASET.get_cleardata_delete()
cleardata = DATASET.cleardata

tempmodel_path = 'TDCS/models/base_del_temp.pkl'
model_path = 'TDCS/models/base_tdcs_del.pkl'
model_h_path = 'TDCS/models/base_tdcs_del_h.pkl'

#train test split
train_in = []
train_exp = []
train_time = []
test_in = []
test_exp = []
test_time = []

for i in range(len(roaddata) - data_per_day * test_day * 2 - input_num):
    temp = []
    temp2 = []
    for j in range(input_num + 1):
        if j < input_num:
            temp.append(roaddata[i + j])
            temp2.append(roadtime[i + j])
        else:
            train_exp.append(roaddata[i + j])
    train_in.append(temp)
    train_time.append(temp2)

for i in range(len(roaddata) - data_per_day * test_day * 2, len(roaddata) - input_num):
    temp = []
    temp2 = []
    for j in range(input_num + 1):
        if j < input_num:
            temp.append(roaddata[i + j])
            temp2.append(roadtime[i + j])
        else:
            test_exp.append(roaddata[i + j])
    test_in.append(temp)
    test_time.append(temp2)

#training
bestneu = RNNv1.Neu()
besterror = 1000000000
bestneu_h = neural.Neu()
besterror_h = 1000000000

neu_try_num = 10
epochs = 10

for n in range(neu_try_num): #num of neural network
    pasterror = 1000000000
    hamm = neural.Neu()
    for k in range(epochs): # num of epoch
        for i in range(len(train_in)):
            try:
                if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) == True:
                    y = hamm.forward(DATASET.array_to_std(train_in[i]))
                    # print(y[0])
                    hamm.backward([DATASET.to_std(train_exp[i])])
                else:
                    hamm.cleartemporalepoch()
            except:
                print('reset')
                hamm = neural.Neu()
        
        error_h = 0
        for i in range(len(test_in) - data_per_day * test_day):
            if DATASET.normaldata(test_in[i][-1], DATASET.avg[test_time[i][-1]], DATASET.std[test_time[i][-1]]) == True:
                y = hamm.forward(DATASET.array_to_std(test_in[i]))
                error_h = error_h + (test_exp[i] - DATASET.to_orig(y[0])) * (test_exp[i] - DATASET.to_orig(y[0]))
            else:
                hamm.cleartemporalepoch()      
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

for n in range(neu_try_num): #num of neural network
    pasterror = 1000000000
    testneu = RNNv1.Neu()
    for k in range(epochs): # num of epoch
        for i in range(len(train_in)):
            try:
                if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) == True:
                    testneu.forward(DATASET.array_to_std(train_in[i]))
                    # print(testneu.output[0][0])
                    testneu.backward(DATASET.to_std(train_exp[i]) - testneu.output[0][0], DATASET.array_to_std(train_in[i]))
                else:
                    testneu.cleartemporalepoch()
            except:
                print('reset')
                testneu = RNNv1.Neu()
        
        error = 0
        for i in range(len(test_in) - data_per_day * test_day):
            if DATASET.normaldata(test_in[i][-1], DATASET.avg[test_time[i][-1]], DATASET.std[test_time[i][-1]]) == True:
                testneu.forward(DATASET.array_to_std(test_in[i]))
                error = error + (test_exp[i] - DATASET.to_orig(testneu.output[0][0])) * (test_exp[i] - DATASET.to_orig(testneu.output[0][0]))
            else:
                testneu.cleartemporalepoch()
        error = math.sqrt(error / (len(test_in) - data_per_day * test_day))
        print(n, ' rnn rmse:', error)
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
y1 = [] #expect
y2 = [] #basicRNN output
y3 = [] #hammerstein output
error = 0
errorlist = []
error_h = 0
errorlist_h = []

for i in range(len(test_in) - data_per_day * test_day, len(test_in)):
    bestneu.forward(DATASET.array_to_std(test_in[i]))
    y = bestneu_h.forward(DATASET.array_to_std(test_in[i]))

    error = error + (test_exp[i] - DATASET.to_orig(bestneu.output[0][0])) * (test_exp[i] - DATASET.to_orig(bestneu.output[0][0]))
    errorlist.append(abs(test_exp[i] - DATASET.to_orig(bestneu.output[0][0])))
    error_h = error_h + (test_exp[i] - DATASET.to_orig(y[0])) * (test_exp[i] - DATASET.to_orig(y[0]))
    errorlist_h.append(abs(test_exp[i] - DATASET.to_orig(y[0])))

    y1.append(test_exp[i])
    y2.append(DATASET.to_orig(bestneu.output[0][0]))
    y3.append(DATASET.to_orig(y[0]))

error = math.sqrt(error / (data_per_day * test_day))
error_h = math.sqrt(error_h / (data_per_day * test_day))
print('final rnn rmse:', error)
print('final hamm rmse:', error_h)

for i in range(data_per_day * test_day):
    x.append(i + 1)

plt.figure(figsize=(20,5))
plt.title('process_delete')
plt.plot(x, y1, '--', c='grey', label='act', linewidth=3.0)
plt.plot(x, y2, '-', c='black', label='basicRNN')
plt.plot(x, y3, '-.', c='black', label='Hamm')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('process_delete Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.xlabel('ERROR')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()