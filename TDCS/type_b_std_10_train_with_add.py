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
    if data > avg - std and data < avg + std:
        return True
    else:
        return False

#load base model
f = open('TDCS/models/b_std_10_base.pkl', 'rb')
baseneu = pickle.load(f)
f2 = open('TDCS/models/b_std_10_base_h.pkl', 'rb')
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

#training
neural_num = 60

bestadd = ADDv1.Add_model()
besterror = 100000000
for n in range(neural_num): #num of neural network
    testadd = ADDv1.Add_model()
    error = 0
    for i in range(len(train_in)):
        try:
            out = 0
            inputbase = to_std(train_in[i])
            if normaldata(train_in[i][-1], avg[train_time[i][-1]], std[train_time[i][-1]]) != True:
                baseneu.forward(inputbase)
                testadd.forward(inputbase)
                out = to_orig(baseneu.output[0][0]) + to_orig(testadd.output[0][0])
            else:
                baseneu.forward(inputbase)
                out = to_orig(baseneu.output[0][0])
            error = error + (train_exp[i] - out) * (train_exp[i] - out)
        except:
            testadd = ADDv1.Add_model()

    error = math.sqrt(error / len(train_in))
    print('rmse:', error)
    baseneu.cleartemporalepoch()
    if error < besterror:
        bestadd = testadd
        besterror = error
bestadd.savemodel('TDCS/models/b_std_10_add.pkl')

bestadd_h = ADDv1.Add_model()
besterror = 100000000
for n in range(neural_num): #num of neural network
    testadd_h = ADDv1.Add_model()
    error = 0
    for i in range(len(train_in)):
        try:
            out = 0
            inputbase = to_std(train_in[i])
            if normaldata(train_in[i][-1], avg[train_time[i][-1]], std[train_time[i][-1]]) != True:
                y = hammneu.forward(inputbase)
                testadd_h.forward(inputbase)
                out = to_orig(y[0]) + to_orig(testadd_h.output[0][0])
            else:
                y = hammneu.forward(inputbase)
                out = to_orig(y[0])
            error = error + (train_exp[i] - out) * (train_exp[i] - out)
        except:
            testadd_h = ADDv1.Add_model()

    error = math.sqrt(error / len(train_in))
    print('rmse:', error)
    hammneu.cleartemporalepoch()
    if error < besterror:
        bestadd_h = testadd_h
        besterror = error
bestadd_h.savemodel('TDCS/models/b_std_10_add_h.pkl')

#testing
test_day = 14
x = []
for i in range(288 * test_day):
    x.append(i + 1)

y1 = []
y2 = []
error = 0
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
    errorlist.append(abs(test_exp[i] - out))
    y1.append(test_exp[i])
    y2.append(out)
error = math.sqrt(error / (288 * test_day))
print('basicRNN rmse:', error)

y1h = []
y2h = []
error = 0
errorlist_h = []
for i in range(288 * test_day):
    out = 0
    inputbase = to_std(test_in[i])
    if normaldata(test_in[i][-1], avg[test_time[i][-1]], std[test_time[i][-1]]) != True:
        y = hammneu.forward(inputbase)
        bestadd_h.forward(inputbase)
        out = to_orig(y[0]) + to_orig(bestadd_h.output[0][0])
    else:
        y = hammneu.forward(inputbase)
        out = to_orig(y[0])
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    errorlist_h.append(abs(test_exp[i] - out))
    y1h.append(test_exp[i])
    y2h.append(out)
error = math.sqrt(error / (288 * test_day))
print('Hamm rmse:', error)

plt.figure(figsize=(20,5))
plt.title('Random std 1.0 Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2, '-', c='black', label='basicRNN with add')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(20,5))
plt.title('Random std 1.0 Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2h, '-.', c='black', label='Hamm with add')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('Random std 1.0 with Add Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.legend(loc='upper right')
plt.show()