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

from RNNmodels2 import RNNv1, neural

input_num = 5

#read data
filepath = 'D:/Python/Jupyter/data/TDCSDIVIDEBYSEG/01F0467S-01F0509S.csv'
roaddata = []
roadtime = []
with open(filepath, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if row[18] != 'Null':
            roaddata.append(int(row[18]))
        else:
            roaddata.append(193)
        t1 = row[20].split(':')
        t2 = int((int(t1[0]) * 60 + int(t1[1])) / 5)
        roadtime.append(t2)
f.close()

#calculate std avg
dtime = []
for i in range(288):
    dtime.append([])
for i in range(len(roaddata)):
    dtime[roadtime[i]].append(roaddata[i])

std = []
avg = []
for t in dtime:
    std.append(np.std(np.array(t)))
    avg.append(np.mean(np.array(t)))

def normaldata(data, avg, std):
    if data > avg - 2*std and data < avg + 2*std:
        return True
    else:
        return False

#get max min
cleardata = []
for i in range(len(roaddata)):
    if normaldata(roaddata[i], avg[roadtime[i]], std[roadtime[i]]) == True:
        cleardata.append(roaddata[i])
    else:
        cleardata.append(avg[roadtime[i]])
maxd = max(cleardata)
mind = min(cleardata)

#resize to -1 ~ 1
def to_std(x):
    return 2 * (x - mind) / (maxd - mind) - 1
def array_to_std(x):
    newx = []
    for i in x:
        newx.append(2 * (i - mind) / (maxd - mind) - 1)
    return newx
def to_orig(x):
    return (x + 1) / 2 * (maxd - mind) + mind

#train test split
test_day = 14

train_in = []
train_exp = []
test_in = []
test_in_orig = []
test_exp = []
test_exp_orig = []

for i in range(len(cleardata) - 288 * test_day * 2 - input_num):
    temp = []
    for j in range(input_num + 1):
        if j < input_num:
            temp.append(cleardata[i + j])
        else:
            train_exp.append(cleardata[i + j])
    train_in.append(temp)

for i in range(len(cleardata) - 288 * test_day * 2, len(cleardata) - input_num):
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

neunum = 30
epochs = 30

for n in range(neunum): #num of neural network
    pasterror = 1000000000
    hamm = neural.Neu()
    for k in range(epochs): # num of epoch
        for i in range(len(train_in)):
            try:
                y = hamm.forward(array_to_std(train_in[i]))
                hamm.backward([to_std(train_exp[i])])
            except:
                hamm = neural.Neu()
        
        error_h = 0
        for i in range(len(test_in) - 288 * test_day):
            y = hamm.forward(array_to_std(test_in[i]))
            error_h = error_h + (test_exp[i] - to_orig(y[0])) * (test_exp[i] - to_orig(y[0]))
        error_h = math.sqrt(error_h / (len(test_in) - 288 * test_day))
        print(n, ' hamm rmse:', error_h)
        hamm.cleartemporalepoch()

        if pasterror > error_h and math.isnan(error_h) != True:
            pasterror = error_h
            hamm.saveneu('TDCS/models/b_std_20_base_temp.pkl')
        else:
            f = open('TDCS/models/b_std_20_base_temp.pkl', 'rb')
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
                testneu.forward(array_to_std(train_in[i]))
                testneu.backward(to_std(train_exp[i]) - testneu.output[0][0], array_to_std(train_in[i]))
            except:
                testneu = RNNv1.Neu()
        
        error = 0
        for i in range(len(test_in) - 288 * test_day):
            testneu.forward(array_to_std(test_in[i]))
            error = error + (test_exp[i] - to_orig(testneu.output[0][0])) * (test_exp[i] - to_orig(testneu.output[0][0]))
        error = math.sqrt(error / (len(test_in) - 288 * test_day))
        print(n, ' rmse:', error)
        testneu.cleartemporalepoch()

        if pasterror > error and math.isnan(error) != True:
            pasterror = error
            testneu.saveneu('TDCS/models/b_std_20_base_temp2.pkl')
        else:
            f = open('TDCS/models/b_std_20_base_temp2.pkl', 'rb')
            testneu = pickle.load(f)
            f.close()
            break
    if pasterror < besterror:
        besterror = pasterror
        bestneu = testneu

bestneu.maxd = maxd
bestneu.mind = mind
bestneu_h.maxd = maxd
bestneu_h.mind = mind
bestneu.saveneu('TDCS/models/b_std_20_base.pkl')
bestneu_h.saveneu('TDCS/models/b_std_20_base_h.pkl')

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

for i in range(len(test_in) - 288 * test_day, len(test_in)):
    bestneu.forward(array_to_std(test_in_orig[i]))
    y = bestneu_h.forward(array_to_std(test_in_orig[i]))

    error = error + (test_exp_orig[i] - to_orig(bestneu.output[0][0])) * (test_exp_orig[i] - to_orig(bestneu.output[0][0]))
    errorlist.append(abs(test_exp_orig[i] - to_orig(bestneu.output[0][0])))

    error_h = error_h + (test_exp_orig[i] - to_orig(y[0])) * (test_exp_orig[i] - to_orig(y[0]))
    errorlist_h.append(abs(test_exp_orig[i] - to_orig(y[0])))

    y_orig.append(test_exp_orig[i])
    y_mov.append(test_exp[i])
    y2.append(to_orig(bestneu.output[0][0]))
    y3.append(to_orig(y[0]))

error = math.sqrt(error / (288 * test_day))
error_h = math.sqrt(error_h / (288 * test_day))
print('final rmse:', error)
print('final hamm rmse:', error_h)

for i in range(288 * test_day):
    x.append(i + 1)

plt.figure(figsize=(20,5))
plt.title('std 2.0')
plt.plot(x, np.log2(y_orig), '--', c='grey', label='act', linewidth=2.0)
plt.plot(x, np.log2(y2), '-', c='black', label='basicRNN')
plt.plot(x, np.log2(y3), '-.', c='black', label='Hamm')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('std 2.0 Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.legend(loc='upper right')
plt.show()