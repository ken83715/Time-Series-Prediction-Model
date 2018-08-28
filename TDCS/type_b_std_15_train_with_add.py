"""
training base RNN and additional model
train_with_add.py
Created on 2018/06/06
@author ken83715
"""

import math
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

from RNNmodels2 import RNNv1, neural, ADDv1
import dataset

input_num = 5

#==========================
stdn_str = '15'
stdn_str2 = '1.5'
stdn = 1.5
#==========================

test_day = 7
data_per_day = 288

DATASET = dataset.TDCSDATA(stdn)
roaddata = DATASET.roaddata
roadtime = DATASET.roadtime
DATASET.get_cleardata_totalavg()
cleardata = DATASET.cleardata

model_path = 'TDCS/models/b_std_' + stdn_str + '_add_r.pkl'
model_h_path = 'TDCS/models/b_std_' + stdn_str + '_add_r_h.pkl'

#load base model
f = open('TDCS/models/b_std_' + stdn_str + '_base.pkl', 'rb')
baseneu = pickle.load(f)
f2 = open('TDCS/models/b_std_' + stdn_str + '_base_h.pkl', 'rb')
hammneu = pickle.load(f2)

#get max min
maxd = baseneu.maxd
mind = baseneu.mind

#train test split
train_in = []
train_exp = []
train_time = []
test_in = []
test_exp = []
test_time = []

for i in range(128667 - input_num):
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

for i in range(128667, len(roaddata) - input_num):
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
neu_try_num = 10

bestadd = ADDv1.Add_model()
besterror = 100000000
for n in range(neu_try_num): #num of neural network
    testadd = ADDv1.Add_model()
    error = 0
    for i in range(len(train_in)):
        try:
            out = 0
            inputbase = DATASET.array_to_std(train_in[i])
            if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) != True:
                baseneu.forward(inputbase)
                testadd.forward(inputbase)
                out = DATASET.to_orig(baseneu.output[0][0]) + DATASET.to_orig(testadd.output[0][0])
            else:
                baseneu.forward(inputbase)
                out = DATASET.to_orig(baseneu.output[0][0])
            error = error + (train_exp[i] - out) * (train_exp[i] - out)
        except:
            print('exception')
            testadd = ADDv1.Add_model()

    error = math.sqrt(error / len(train_in))
    print('rmse:', error)
    baseneu.cleartemporalepoch()
    if error < besterror:
        bestadd = testadd
        besterror = error
bestadd.savemodel(model_path)

bestadd_h = ADDv1.Add_model()
besterror = 100000000
for n in range(neu_try_num): #num of neural network
    testadd_h = ADDv1.Add_model()
    error = 0
    for i in range(len(train_in)):
        try:
            out = 0
            inputbase = DATASET.array_to_std(train_in[i])
            if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) != True:
                y = hammneu.forward(inputbase)
                testadd_h.forward(inputbase)
                out = DATASET.to_orig(y[0]) + DATASET.to_orig(testadd_h.output[0][0])
            else:
                y = hammneu.forward(inputbase)
                out = DATASET.to_orig(y[0])
            error = error + (train_exp[i] - out) * (train_exp[i] - out)
        except:
            print('exception')
            testadd_h = ADDv1.Add_model()

    error = math.sqrt(error / len(train_in))
    print('rmse:', error)
    hammneu.cleartemporalepoch()
    if error < besterror:
        bestadd_h = testadd_h
        besterror = error
bestadd_h.savemodel(model_h_path)

#testing
x = []
for i in range(data_per_day * test_day):
    x.append(i + 1)

y1 = []
y2 = []
error = 0
errorlist = []
for i in range(data_per_day * test_day):
    out = 0
    inputbase = DATASET.array_to_std(test_in[i])
    if DATASET.normaldata(test_in[i][-1], DATASET.avg[test_time[i][-1]], DATASET.std[test_time[i][-1]]) != True:
        baseneu.forward(inputbase)
        bestadd.forward(inputbase)
        out = DATASET.to_orig(baseneu.output[0][0]) + DATASET.to_orig(bestadd.output[0][0])
    else:
        baseneu.forward(inputbase)
        out = DATASET.to_orig(baseneu.output[0][0])
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    errorlist.append(abs(test_exp[i] - out))
    y1.append(test_exp[i])
    y2.append(out)
error = math.sqrt(error / (data_per_day * test_day))
print('basicRNN rmse:', error)

y1h = []
y2h = []
error = 0
errorlist_h = []
for i in range(data_per_day * test_day):
    out = 0
    inputbase = DATASET.array_to_std(test_in[i])
    if DATASET.normaldata(test_in[i][-1], DATASET.avg[test_time[i][-1]], DATASET.std[test_time[i][-1]]) != True:
        y = hammneu.forward(inputbase)
        bestadd_h.forward(inputbase)
        out = DATASET.to_orig(y[0]) + DATASET.to_orig(bestadd_h.output[0][0])
    else:
        y = hammneu.forward(inputbase)
        out = DATASET.to_orig(y[0])
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    errorlist_h.append(abs(test_exp[i] - out))
    y1h.append(test_exp[i])
    y2h.append(out)
error = math.sqrt(error / (data_per_day * test_day))
print('Hamm rmse:', error)

plt.figure(figsize=(20,5))
plt.title('Random std ' + stdn_str2 + ' Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2, '-', c='black', label='basicRNN with add')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(20,5))
plt.title('Random std ' + stdn_str2 + ' Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2h, '-.', c='black', label='Hamm with add')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('Random std ' + stdn_str2 + ' with Add Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.xlabel('ERROR')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()