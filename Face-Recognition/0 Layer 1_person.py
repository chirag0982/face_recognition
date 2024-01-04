import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
import pandas as pd
from subprocess import check_output
import os, sys
from IPython.display import display
from IPython.display import Image
from PIL import Image
import numpy as np
from time import time
from time import sleep

def sigmoid(z):
    return 1/(1+np.exp(-z))

def Sigmoid_Derivative(x):
    
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return d

def initialize_with_zeros(n,a):
    
    w = np.random.randn(n,a)
    b = 0
    return w,b



def propagate(w, b, X, Y):

    m = X.shape[1]
    Y = np.array(Y)
    Y = Y.reshape(1,Y.shape[0])
    Z = np.dot(w.T,X) + b
    A = sigmoid(Z)

    
    dw = np.dot(X,(A-Y).T)/m

    db = np.sum(A-Y)*1/m
    logprobs = np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),1-Y)
    cost = - np.sum(logprobs)/m
      

    
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
    params = {"w": w,
                 "b": b}

    grads = {"dw": dw,
              "db": db}

    return params, grads, costs

def predict(w, b, X):

    m = X.shape[1]
    a=w.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(m):
        if A[0,i] >= 0.5:
            Y_Prediction[0,i] = 1

    return Y_prediction


def accuracy (Y_train, Y_predict):
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(1,Y_train.shape[0])
    m = Y_train.shape[1]
    correct = 0
    for i in range(m):
        if Y_train[0,i] == Y_predict[0,i]:
            correct = correct + 1 
    
    Acc = (correct/m)*100
    return Acc

def model(X_train, Y_train,X_test,Y_test,num_iterations = 25000, learning_rate = 0.001):
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate,print_cost=False)
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    Train_Accuracy = accuracy(Y_train,Y_prediction_train)
    Test_Accuracy = accuracy(Y_test,Y_prediction_test)
    
    d={"costs": costs,
      "Y_prediction_test": Y_prediction_test, 
      "Y_prediction_train" : Y_prediction_train,
      "w" : w,
      "b" : b,
      "learning_rate" : learning_rate,
      "num_iterations": num_iterations,
      "Train_Accuracy": Train_Accuracy,
      "Test_Accuracy": Test_Accuracy}
    
    return d


## While using collab we use this cell to import files. 

from google.colab import files
uploaded = files.upload()

print(model(X_train, Y_train, X_test, Y_test, num_iterations = 25000, learning_rate = 0.001))