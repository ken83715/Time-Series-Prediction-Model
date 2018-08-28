"""
training base RNN and additional model
train_add_gene.py
Created on 2018/06/06
@author ken83715
"""

import math
import pickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from RNNmodels2 import RNNv1, neural, ADDv2
import dataset

input_num = 5

#==========================
stdn_str = '20'
stdn_str2 = '2.0'
stdn = 2
#==========================

test_day = 7
data_per_day = 288

DATASET = dataset.TDCSDATA(stdn)
roaddata = DATASET.roaddata
roadtime = DATASET.roadtime
DATASET.get_cleardata_totalavg()
cleardata = DATASET.cleardata

model_path = 'TDCS/models/b_std_' + stdn_str + '_add_g.pkl'
model_h_path = 'TDCS/models/b_std_' + stdn_str + '_add_g_h.pkl'

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
param = []
for i in range(1000):
    param.append(0)
bestadd = ADDv2.Add_model(param)
bestadd_h = ADDv2.Add_model(param)

genelength = bestadd.para_num * 20
populationCnt = 30
iteration = 10
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
        wlength = int(genelength / bestadd.para_num) #will be 20
        for j in range(bestadd.fuzzy1 * 3 + bestadd.de_fuzzy1 + 2):
            weight = 0
            po = 0
            for k in range(j * wlength, j * wlength + wlength):
                weight = weight + gene[i][k] * math.pow(2, po)
                po = po + 1
            if j < bestadd.fuzzy1:                    
                weight = (weight - math.pow(2, wlength - 1)) / math.pow(2, wlength - 1) #-1 to 1
            if bestadd.fuzzy1 <= j and j < bestadd.fuzzy1 * 2:
                weight = weight / (math.pow(2, wlength) - 1) #0 to 1
            if bestadd.fuzzy1 * 2 <= j and j < bestadd.fuzzy1 * 3 + bestadd.de_fuzzy1 + 1:
                weight = (weight - math.pow(2, wlength - 1)) / math.pow(2, wlength - 5) #-16 to 16
            if j == bestadd.fuzzy1 * 3 + bestadd.de_fuzzy1 + 1:
                weight = (weight - math.pow(2, wlength - 1)) / math.pow(2, wlength - 1) #-1 to 1
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

def fitnessbase(addmodel):
    """
    testing the model
    """
    error = 0
    for i in range(len(train_in)):
        out = 0
        inputbase = DATASET.array_to_std(train_in[i])
        if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) != True:
            baseneu.forward(inputbase)
            addmodel.forward(inputbase)
            out = DATASET.to_orig(baseneu.output[0][0]) + DATASET.to_orig(addmodel.output[0][0])
        else:
            baseneu.forward(inputbase)
            out = DATASET.to_orig(baseneu.output[0][0])
        error = error + (train_exp[i] - out) * (train_exp[i] - out)
        
    error = math.sqrt(error / len(train_in))
    #print('rmse:', error)
    return error

def fitnesshamm(addmodel):
    """
    testing the model
    """
    error = 0
    for i in range(len(train_in)):
        out = 0
        inputbase = DATASET.array_to_std(train_in[i])
        if DATASET.normaldata(train_in[i][-1], DATASET.avg[train_time[i][-1]], DATASET.std[train_time[i][-1]]) != True:
            y = hammneu.forward(inputbase)
            addmodel.forward(inputbase)
            out = DATASET.to_orig(y[0]) + DATASET.to_orig(addmodel.output[0][0])
        else:
            y = hammneu.forward(inputbase)
            out = DATASET.to_orig(y[0])
        error = error + (train_exp[i] - out) * (train_exp[i] - out)       
    error = math.sqrt(error / len(train_in))
    #print('rmse:', error)
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
for i in range(iteration):
    print('iter:', i + 1)
    parameter = decode(gene)
    addmodel_list = generateaddmodel(parameter)
    error_list = []
    for j in range(populationCnt):
        try:
            error_list.append(fitnessbase(addmodel_list[j]))
        except:
            error_list.append(400)
    print(min(error_list))
    copygene = copy(error_list, gene)
    crossgene = crossover(copygene)
    mutegene = mutation(crossgene)
    gene = mutegene

parameter = decode(gene)
addmodel_list = generateaddmodel(parameter)
error_list = []
for j in range(populationCnt):
    try:
        error_list.append(fitnessbase(addmodel_list[j]))
    except:
        error_list.append(400)
minerror = min(error_list)
for i in range(len(error_list)):
    if error_list[i] == minerror:
        bestadd = addmodel_list[i]

bestadd.maxd = maxd
bestadd.mind = mind
bestadd.savemodel(model_path)

geneh = initial()
for i in range(iteration):
    print('iter:', i + 1)
    parameter = decode(geneh)
    addmodel_list = generateaddmodel(parameter)
    error_list = []
    for j in range(populationCnt):
        try:
            error_list.append(fitnesshamm(addmodel_list[j]))
        except:
            error_list.append(400)
    print(min(error_list))
    copygene = copy(error_list, geneh)
    crossgene = crossover(copygene)
    mutegene = mutation(crossgene)
    geneh = mutegene

parameter = decode(geneh)
addmodel_list = generateaddmodel(parameter)
error_list = []
for j in range(populationCnt):
    try:
        error_list.append(fitnessbase(addmodel_list[j]))
    except:
        error_list.append(400)
minerror = min(error_list)
for i in range(len(error_list)):
    if error_list[i] == minerror:
        bestadd_h = addmodel_list[i]

bestadd_h.maxd = maxd
bestadd_h.mind = mind
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
plt.title('Genealgo std ' + stdn_str2 + ' Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2, '-', c='black', label='basicRNN with add')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(20,5))
plt.title('Genealgo std ' + stdn_str2 + ' Performance')
plt.plot(x, y1, '--', c='grey', label = 'act', linewidth=2.0)
plt.plot(x, y2h, '-.', c='black', label='Hamm with add')
plt.xlabel('time')
plt.ylabel('second')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(0, 100, 10)
plt.figure(figsize=(15,5))
plt.title('Genealgo std ' + stdn_str2 + ' with Add Error distribution')
plt.hist([errorlist, errorlist_h], bins, label=['basicRNN', 'Hamm'], color=['black', 'grey'])
plt.xlabel('ERROR')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()