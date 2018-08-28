"""
training base RNN and additional model
train_with_add.py
Created on 2018/04/20
@author ken83715
"""

import math
import matplotlib.pyplot as plt
import pickle
from RNNmodels2 import RNNv1, ADDv1

#generate data
train = []
k = 0
for i in range(20):
    for j in range(35):
        train.append(math.sin(k))
        k = k + 1
    for j in range(5):
        train.append(math.sin(k) + 2 * math.cos(k))
        k = k + 1
for i in range(5):
    train.append(math.sin(k))
    k = k + 1

test = []
k = 1000
for i in range(10):
    for j in range(35):
        test.append(math.sin(k))
        k = k + 1
    for j in range(5):
        test.append(math.sin(k) + 2 * math.cos(k))
        k = k + 1
for i in range(5):
    test.append(math.sin(k))
    k = k + 1

train_in = []
train_exp = []
for i in range(795):
    temp = []
    for j in range(5):
        temp.append(train[i + j])
    train_in.append(temp)
    train_exp.append(train[i + 5])

test_in = []
test_exp = []
for i in range(395):
    temp = []
    for j in range(5):
        temp.append(test[i + j])
    test_in.append(temp)
    test_exp.append(test[i + 5])

#training
f = open('sincos/neu_sinwave.pkl', 'rb')
baseneu = pickle.load(f)
bestadd = ADDv1.Add_model()
besterror = 10000

for n in range(10000):
    testadd = ADDv1.Add_model()
    
    error = 0
    for i in range(795):
        out = 0
        if train_in[i][4] > 1 or train_in[i][4] < -1:
            baseneu.forward(train_in[i])
            testadd.forward(train_in[i])
            out = baseneu.output[0][0] + testadd.output[0][0]
        else:
            baseneu.forward(train_in[i])
            out = baseneu.output[0][0]
        
        error = error + (train_exp[i] - out) * (train_exp[i] - out)
    error = math.sqrt(error / 795)
    #print('mse:', error)
    baseneu.cleartemporalepoch()

    if error < besterror:
        besterror = error
        bestadd = testadd
        print('besterror ', besterror)

bestadd.savemodel('sincos/add_sin_cos.pkl')

x = []
y1 = []
y2 = []
error = 0
for i in range(395):
    x.append(i + 1000)
    out = 0
    if test_in[i][4] > 1 or test_in[i][4] < -1:
        baseneu.forward(test_in[i])
        bestadd.forward(test_in[i])
        out = baseneu.output[0][0] + bestadd.output[0][0]
    else:
        baseneu.forward(test_in[i])
        out = baseneu.output[0][0]
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    y1.append(test_exp[i])
    y2.append(out)

error = math.sqrt(error / 395)
print('final rmse:', error)
plt.figure(figsize=(15,5))
plt.plot(x, y1)
plt.plot(x, y2, '--')
plt.show()
