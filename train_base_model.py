"""
training Hammerstein Recurrent Neural Network
Created on 2018/04/19
"""

import math
import matplotlib.pyplot as plt
from hammerstein import Hammerstein

INPUT_NUMBER = 5

#generate simulated data
def generate_data(train_ctl, test_ctl, input_num):
    """
    generate a sin wave data
    """

    train = []
    iteration = 0
    for i in range(train_ctl):
        for j in range(40):
            train.append(math.sin(iteration))
            iteration = iteration + 1

    test = []
    iteration = train_ctl * 50
    for i in range(test_ctl):
        for j in range(40):
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
    train Hammerstein model
    try_count means how many times to try creating model
    """
    
    best_model = Hammerstein(INPUT_NUMBER)
    best_error = 10000

    for n in range(try_count):
        print('neuralnetwork ', n + 1)
        last_epoch_error = 10000
        test_model = Hammerstein(INPUT_NUMBER)
        for k in range(1000):
            for i in range(len(x_train) - 50):
                try:
                    test_model.forward(x_train[i])
                    test_model.backward([y_train[i]])
                except OverflowError:
                    print('Model broken, renew a Model.')
                    test_model = Hammerstein(INPUT_NUMBER)

            error = 0
            for i in range((len(x_train) - 50), len(x_train)):
                y_pred = test_model.forward(x_train[i])
                error = error + (y_train[i] - y_pred[0]) * (y_train[i] - y_pred[0])
            error = math.sqrt(error / 50)

            if k % 100 == 0:
                print('mse:', error)

            test_model.cleartemporalepoch()

            if last_epoch_error > error:
                last_epoch_error = error
            else:
                print('error increasing, break ', error)
                break

        if last_epoch_error < best_error:
            best_error = last_epoch_error
            best_model = test_model
            print('neuralnetwork ', n + 1, 'besterror ', best_error)

    best_model.saveneu('hammerstein_model.pkl')
    return best_model

#testing
def testing(x_test, y_test, model):
    """
    testing model
    """

    x_axis = []
    y_true = []
    y_pred = []
    error = 0
    for i in range(len(x_test)):
        x_axis.append(i)
        out = model.forward(x_test[i])
        error = error + (y_test[i] - out[0]) * (y_test[i] - out[0])
        y_true.append(y_test[i])
        y_pred.append(out[0])

    error = math.sqrt(error / 50)
    print('final rmse:', error)
    plt.figure(figsize=(20, 5))
    plt.plot(x_axis, y_true, label='y_true')
    plt.plot(x_axis, y_pred, '--', label='y_pred')
    plt.title('sin wave')
    plt.legend()
    plt.show()

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = generate_data(30, 5, INPUT_NUMBER)
MODEL = training(X_TRAIN, Y_TRAIN, 2)
testing(X_TEST, Y_TEST, MODEL)
