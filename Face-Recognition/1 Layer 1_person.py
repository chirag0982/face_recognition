import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
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

def layer_sizes(X, Y,n_m):
    
    n_x = X.shape[0] # size of input layer
    n_h = n_m          # size of hidden layer
    n_y = Y.shape[0] # size of output layer
    
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h,n_x)
    b1 = np.random.randn(n_h,1)
    W2 = np.random.randn(n_y,n_h)
    b2 = np.random.randn(n_y,1)
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    
    W1 = parameters['W1']         ### retrieving the parameters
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1,X) + b1        ### implementing forward propagation
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (W2.shape[0], X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = - np.sum(logprobs)/m
      
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters['W1']      # Retrieving parameters 
    W2 = parameters['W2']
    
    A1 = cache['A1']           # Retrieving required things from cache
    A2 = cache['A2']
    
    
    dZ2 = A2 - Y                    # Backward Propagation
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims = True)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.001):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_m, num_iterations, print_cost):

    n_x, n_h, n_y = layer_sizes(X, Y, n_m )
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, num_iterations): 
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 0.001)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
   
    A2 , cache = forward_propagation(X,parameters)
    m = X.shape[1]
    Y_Prediction = np.zeros((1,m))
    for i in range(m):
        if A2[0,i] >= 0.5:
            Y_Prediction[0,i] = 1
    return Y_Prediction

def accuracy (Y_train, Y_predict):
    
    m = Y_train.shape[1]
    correct = 0
    for i in range(m):
        if Y_train[0,i] == Y_predict[0,i]:
            correct = correct + 1 
    
    Acc = (correct/m)*100
    return Acc


def model(X_train, Y_train, X_test, Y_test, n_m, num_iterations , learning_rate , print_cost=True):
    
    parameters = nn_model(X_train, Y_train, n_m , num_iterations , print_cost)    
                 
    Y_prediction_train = predict(parameters, X_train)
    Y_prediction_test = predict(parameters, X_test)
    

    Train_Accuracy = accuracy(Y_train,Y_prediction_train)
    Test_Accuracy = accuracy(Y_test,Y_prediction_test)
    
    return Train_Accuracy , Test_Accuracy


## While using collab we use this cell to import files. 

from google.colab import files
uploaded = files.upload()

print(model(X_train, Y_train, X_test, Y_test,n_m = 25, num_iterations = 50000, learning_rate = 0.001, print_cost = False))