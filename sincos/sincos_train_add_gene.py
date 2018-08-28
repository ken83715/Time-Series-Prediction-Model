"""
training base RNN and additional model
train_with_add.py
Created on 2018/04/20
@author ken83715
"""

import math
import random
import matplotlib.pyplot as plt
import pickle
from RNNmodels2 import RNNv1, ADDv2

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

genelength = 120
populationCnt = 50
iteration = 200
crossoverRate = 0.8
mutationRate = 0.4

def initial():
    """
    initial gene
    """
    gene = []
    for i in range(populationCnt):
        pop = []
        for j in range(genelength):
            temp = random.random()
            if temp > 0.5:
                temp = 1
            else:
                temp = 0
            pop.append(temp)
        gene.append(pop)

    return gene

def decode(gene):
    """
    decode the gene
    """
    #print('decode')
    parameter = []
    for i in range(populationCnt):
        temp = []
        wlength = int(genelength / 6)
        for j in range(6):
            weight = 0
            po = 0
            for k in range(j * wlength, j * wlength + wlength):
                weight = weight + gene[i][k] * math.pow(2, po)
                po = po + 1
            weight = (weight - 524288) / 131072 #-4 to 4
            temp.append(weight)
        parameter.append(temp)

    return parameter

def generateaddmodel(para):
    """
    generate addmodel
    """
    #print('generate addmodel')
    addlist = []
    for i in range(populationCnt):
        addmodel = ADDv2.Add_model(para[i])
        addlist.append(addmodel)

    return addlist

def fitness(addmodel):
    """
    testing the model
    """
    error = 0
    for i in range(400):
        out = 0
        if train_in[i][4] > 1 or train_in[i][4] < -1:
            baseneu.forward(train_in[i])
            addmodel.forward(train_in[i])
            out = baseneu.output[0][0] + addmodel.output[0][0]
        else:
            baseneu.forward(train_in[i])
            out = baseneu.output[0][0]
        
        error = error + (train_exp[i] - out) * (train_exp[i] - out)
    error = math.sqrt(error / 400)
    #print('mse:', error)
    return error

def copy(nerror, gene):
    """
    copy to next gen, decide by error
    """
    #print('copy')
    newgene = []
    for i in range(populationCnt):
        index1 = random.randint(0, populationCnt - 1)
        index2 = random.randint(0, populationCnt - 1)
        while index2 == index1:
            index2 = random.randint(0, populationCnt - 1)
        if nerror[index1] > nerror[index2]:
            newgene.append(list(gene[index2]))
        else:
            newgene.append(list(gene[index1]))
    return newgene

def crossover(gene):
    """
    exchange gene info
    """
    #print('crossover')
    cross = int(populationCnt * crossoverRate / 2)

    crossoverchk = []
    for i in range(populationCnt):
        crossoverchk.append(0)

    for i in range(cross):
        index1 = random.randint(0, populationCnt - 1)
        while crossoverchk[index1] != 0:
            index1 = random.randint(0, populationCnt - 1)

        index2 = random.randint(0, populationCnt - 1)
        while index2 == index1 or crossoverchk[index2] != 0:
            index2 = random.randint(0, populationCnt - 1)

        crosspoint1 = random.randint(0, genelength - 1)
        crosspoint2 = random.randint(0, genelength - 1)
        while crosspoint1 == crosspoint2:
            crosspoint2 = random.randint(0, genelength - 1)

        if crosspoint1 < crosspoint2:
            for j in range(crosspoint1, crosspoint2):
                t = gene[index1][j]
                gene[index1][j] = gene[index2][j]
                gene[index2][j] = t
        else:
            for j in range(crosspoint2, crosspoint1):
                t = gene[index1][j]
                gene[index1][j] = gene[index2][j]
                gene[index2][j] = t

        crossoverchk[index1] = 1
        crossoverchk[index2] = 1

    return gene

def mutation(gene):
    """
    mutation the gene
    """
    #print('mutation')
    muta = int(populationCnt * mutationRate)

    mutationchk = []
    for i in range(populationCnt):
        mutationchk.append(0)

    for i in range(muta):
        index = random.randint(0, populationCnt - 1)
        while mutationchk[index] != 0:
            index = random.randint(0, populationCnt - 1)

        for i in range(10):
            mupoint = random.randint(0, genelength - 1)
            if gene[index][mupoint] == 0:
                gene[index][mupoint] = 1
            else:
                gene[index][mupoint] = 0

        mutationchk[index] = 1

    return gene

gene = initial()
bestaddmodel = ADDv2.Add_model([0, 0, 0, 0, 0, 0])
for i in range(iteration):
    print('iter:', i + 1)
    parameter = decode(gene)
    addmodel_list = generateaddmodel(parameter)
    error_list = []
    for j in range(populationCnt):
        error_list.append(fitness(addmodel_list[j]))
    print(min(error_list))
    copygene = copy(error_list, gene)
    crossgene = crossover(copygene)
    mutegene = mutation(crossgene)
    gene = mutegene

parameter = decode(gene)
addmodel_list = generateaddmodel(parameter)
error_list = []
for j in range(populationCnt):
    error_list.append(fitness(addmodel_list[j]))
minerror = min(error_list)
for i in range(len(error_list)):
    if error_list[i] == minerror:
        bestaddmodel = addmodel_list[i]

bestaddmodel.savemodel('sincos/add_trainwithgene.pkl')

x = []
y1 = []
y2 = []
error = 0
for i in range(200):
    x.append(i + 500)
    out = 0
    if train_in[i][4] > 1 or train_in[i][4] < -1:
        baseneu.forward(test_in[i])
        bestaddmodel.forward(test_in[i])
        out = baseneu.output[0][0] + bestaddmodel.output[0][0]
    else:
        baseneu.forward(test_in[i])
        out = baseneu.output[0][0]
    error = error + (test_exp[i] - out) * (test_exp[i] - out)
    y1.append(test_exp[i])
    y2.append(out)

error = math.sqrt(error / 200)
print('final rmse:', error)
plt.figure(figsize=(10,5))
plt.plot(x, y1)
plt.plot(x, y2, '--')
plt.show()
