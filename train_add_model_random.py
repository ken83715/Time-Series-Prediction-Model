"""
training additional model
Created on 2018/04/20
"""

import math
import pickle
import matplotlib.pyplot as plt
from add_model import AddModel

INPUT_NUMBER = 5

#generate simulated data
def generate_data(train_ctl, test_ctl, input_num):
    """
    generate a sin wave data
    with several cos noise
    """
    
    train = []
    iteration = 0
    for i in range(train_ctl):
        for j in range(35):
            train.append(math.sin(iteration))
            iteration = iteration + 1
        for j in range(5):
            train.append(math.sin(iteration) + 2 * math.cos(iteration))
            iteration = iteration + 1
    for i in range(5):
        train.append(math.sin(iteration))
        iteration = iteration + 1

    test = []
    iteration = train_ctl * 50
    for i in range(test_ctl):
        for j in range(35):
            test.append(math.sin(iteration))
            iteration = iteration + 1
        for j in range(5):
            test.append(math.sin(iteration) + 2 * math.cos(iteration))
            iteration = iteration + 1
    for i in range(5):
        test.append(math.sin(iteration))
        iteration = iteration + 1

    x_train = []
    y_train = []
    for i in range(train_ctl * 40 - input_num):
        temp = []
        for j in range(input_num):
            temp.append(train[i + j])
        x_train.append(temp)
        y_train.append(train[i + input_num])

    x_test = []
    y_test = []
    for i in range(test_ctl * 40 - input_num):
        temp = []
        for j in range(input_num):
            temp.append(test[i + j])
        x_test.append(temp)
        y_test.append(test[i + input_num])

    return (x_train, y_train, x_test, y_test)

#training
def training(x_train, y_train, try_count):
    """
    training add model
    add model create with random parameters
    increase try_count may get better add model result
    """

    model_path = open('hammerstein_model.pkl', 'rb')
    base_model = pickle.load(model_path)
    best_add_model = AddModel(INPUT_NUMBER)
    best_error = 10000

    for n in range(try_count):
        test_add_model = AddModel(INPUT_NUMBER)

        error = 0
        for i in range(len(x_train)):
            out = 0
            if x_train[i][4] > 1 or x_train[i][4] < -1:
                try:
                    base_out = base_model.forward(x_train[i])
                    test_add_model.forward(x_train[i])
                    out = base_out[0] + test_add_model.output[0][0]
                except OverflowError:
                    print('Add Model broken, renew a Add Model.')
                    test_add_model = AddModel(INPUT_NUMBER)
            else:
                base_out = base_model.forward(x_train[i])
                out = base_out[0]

            error = error + (y_train[i] - out) * (y_train[i] - out)
        error = math.sqrt(error / len(x_train))
        base_model.cleartemporalepoch()

        if error < best_error:
            best_error = error
            best_add_model = test_add_model
            print('add network ', n + 1, ' best_error: ', best_error)

    best_add_model.savemodel('add_model.pkl')
    return (base_model, best_add_model)

#testing
def testing(x_test, y_test, base_model, add_model):
    """
    testing base_model + add_model
    """

    x_axis = []
    y_true = []
    y_pred = []
    error = 0
    for i in range(len(x_test)):
        x_axis.append(i)
        out = 0
        if x_test[i][4] > 1 or x_test[i][4] < -1:
            base_out = base_model.forward(x_test[i])
            add_model.forward(x_test[i])
            out = base_out[0] + add_model.output[0][0]
        else:
            base_out = base_model.forward(x_test[i])
            out = base_out[0]
        error = error + (y_test[i] - out) * (y_test[i] - out)
        y_true.append(y_test[i])
        y_pred.append(out)

    error = math.sqrt(error / len(x_test))
    print('final rmse:', error)
    plt.figure(figsize=(20, 5))
    plt.plot(x_axis, y_true, label='y_true')
    plt.plot(x_axis, y_pred, '--', label='y_pred')
    plt.title('sin wave with noise')
    plt.legend()
    plt.show()

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = generate_data(30, 5, INPUT_NUMBER)
BASE_MODEL, BEST_ADD_MODEL = training(X_TRAIN, Y_TRAIN, 1000)
testing(X_TEST, Y_TEST, BASE_MODEL, BEST_ADD_MODEL)
