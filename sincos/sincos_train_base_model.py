"""
training base RNN
train_base_model.py
Created on 2018/04/19
@author ken83715
"""

import math
import matplotlib.pyplot as plt
from RNNmodels1 import RNNv1

train_in = []
for i in range(400):
    temp = []
    for j in range(5):
        temp.append(math.sin(i + j))
    train_in.append(temp)

test_in = []
for i in range(401, 451):
    temp = []
    for j in range(5):
        temp.append(math.sin(i + j))
    test_in.append(temp)

train_exp = []
for i in range(400):
    train_exp.append(math.sin(i + 5))

test_exp = []
for i in range(401, 451):
    test_exp.append(math.sin(i + 5))

bestneu = RNNv1.Neu()
besterror = 10000

for n in range(10):
    pasterror = 10000
    testneu = RNNv1.Neu()
    for k in range(1000):
        for i in range(400):
            try:
                testneu.forward(train_in[i])
                testneu.backward(train_exp[i] - testneu.output[0][0], train_in[i])
            except:
                testneu = testneu = RNNv1.Neu()
        
        error = 0
        for i in range(50):
            testneu.forward(test_in[i])
            error = error + (test_exp[i] - testneu.output[0][0]) * (test_exp[i] - testneu.output[0][0])
        error = math.sqrt(error / 50)
        #print('mse:', error)
        testneu.cleartemporalepoch()
        if pasterror > error:
            pasterror = error
        else:
            break
    if pasterror < besterror:
        besterror = pasterror
        bestneu = testneu
        print('besterror ', besterror)

bestneu.saveneu('sincos/neu_sinwave.pkl')

x = []
y1 = []
y2 = []
error = 0
for i in range(50):
    x.append(i + 406)
    bestneu.forward(test_in[i])
    error = error + (test_exp[i] - bestneu.output[0][0]) * (test_exp[i] - bestneu.output[0][0])
    y1.append(test_exp[i])
    y2.append(bestneu.output[0][0])

error = math.sqrt(error / 50)
print('final rmse:', error)
plt.figure(figsize=(10,5))
plt.plot(x, y1)
plt.plot(x, y2, '--')
plt.show()
