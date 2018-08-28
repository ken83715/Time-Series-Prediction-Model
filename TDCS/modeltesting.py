"""
training base RNN and additional model
train_with_add.py
Created on 2018/06/06
@author ken83715
"""

import math
import random
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

from RNNmodels2 import RNNv1, neural, ADDv2

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

#load base model
f = open('TDCS/models/b_std_20_base.pkl', 'rb')
baseneu = pickle.load(f)
f2 = open('TDCS/models/b_std_20_base_h.pkl', 'rb')
hammneu = pickle.load(f2)

#get max min
maxd = baseneu.maxd
mind = baseneu.mind

#resize to -1 ~ 1
def to_std(x):
    newx = []
    for i in x:
        newx.append(2 * (i - mind) / (maxd - mind) - 1)
    return newx
def to_orig(x):
    return (x + 1) / 2 * (maxd - mind) + mind

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

f3 = open('TDCS/models/b_std_20_add_g.pkl', 'rb')
bestadd = pickle.load(f3)

f4 = open('TDCS/models/b_std_20_add_g_h.pkl', 'rb')
bestadd_h = pickle.load(f4)

#testing
test_day = 60
x = []
for i in range(288 * test_day):
    x.append(i + 1)

y1 = []
y2 = []
y_base = []
error = 0
base_error = 0
errorlist = []
for i in range(288 * test_day):
    out = 0
    inputbase = to_std(test_in[i])
    if normaldata(test_in[i][-1], avg[test_time[i][-1]], std[test_time[i][-1]]) != True:
        baseneu.forward(inputbase)
        bestadd.forward(inputbase)
        out = to_orig(baseneu.output[0][0]) + to_orig(bestadd.output[0][0])
    else:
        baseneu.forward(inputbase)
        out = to_orig(baseneu.output[0][0])
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    base_error = base_error + (test_exp[i] - to_orig(baseneu.output[0][0])) * (test_exp[i] - to_orig(baseneu.output[0][0]))
    errorlist.append(abs(test_exp[i] - out))
    y1.append(test_exp[i])
    y2.append(out)
    y_base.append(to_orig(baseneu.output[0][0]))
error = math.sqrt(error / (288 * test_day))
base_error = math.sqrt(base_error / (288 * test_day))
print('basicRNN rmse:', error)
print('base model rmse:', base_error)

y1h = []
y2h = []
y_baseh = []
error = 0
base_errorh = 0
errorlist_h = []
for i in range(288 * test_day):
    out = 0
    inputbase = to_std(test_in[i])
    y = hammneu.forward(inputbase)
    if normaldata(test_in[i][-1], avg[test_time[i][-1]], std[test_time[i][-1]]) != True:        
        bestadd_h.forward(inputbase)
        out = to_orig(y[0]) + to_orig(bestadd_h.output[0][0])
    else:
        out = to_orig(y[0])
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    base_errorh = base_errorh + (test_exp[i] - to_orig(y[0])) * (test_exp[i] - to_orig(y[0]))
    errorlist_h.append(abs(test_exp[i] - out))
    y1h.append(test_exp[i])
    y2h.append(out)
    y_baseh.append(to_orig(y[0]))
error = math.sqrt(error / (288 * test_day))
base_errorh = math.sqrt(base_errorh / (288 * test_day))
print('Hamm rmse:', error)
print('base model rmse:', base_errorh)

plt.figure(figsize=(20,5))
plt.title('Genealgo std 2.0 Performance')
plt.plot(x, y1, '--', c='grey', label = 'act')
plt.plot(x, y_base, '-.', c='orange', label='base')
plt.plot(x, y2, '-', c='blue', label='basicRNN with ADDmodel')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(20,5))
plt.title('Genealgo std 2.0 Performance')
plt.plot(x, y1, '--', c='grey', label = 'act')
plt.plot(x, y_baseh, '-', c='orange', label='base')
plt.plot(x, y2h, '-.', c='blue', label='Hamm with ADDmodel')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('Genealgo std 2.0 with Add Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.legend(loc='upper right')
plt.show()