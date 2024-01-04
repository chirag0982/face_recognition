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

 
 

def initialize_parameters_deep(layer_dims):
    
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.001
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.001
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1)) 
        
    parameters['W' + str(L)] = np.random.randn(layer_dims[L], layer_dims[L-1]) * 0.001
    parameters['b' + str(L)] = np.random.randn(layer_dims[L], 1) * 0.001
        
    assert(parameters['W' + str(L)].shape == (layer_dims[L], layer_dims[L-1]))
    assert(parameters['b' + str(L)].shape == (layer_dims[L], 1))    
    
            
    return parameters


def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s,x

def relu(x):
    s = np.maximum(x,0)
    return  s,x

def linear_forward(A, W, b):

    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation=="tanh":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)] ,parameters["b" + str(l)], activation= "tanh")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A,parameters["W" + str(L)] ,parameters["b" + str(L)],activation = "sigmoid")
    caches.append(cache)
    
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def tanh(x):
    s=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return s,x

def tanh_backward(dA,cache):
    Z = cache
    back_value=np.zeros((Z.shape[0],Z.shape[1]),dtype=float)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            back_value[i][j]=Z[i][j]*dA[i][j]
    return back_value


def compute_cost(AL, Y):
    m = Y.shape[1]
    
    cost = -np.sum((np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))))/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
   
    A_prev, W, b = cache
    m = A_prev.shape[1]

    
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T,dZ)
   
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    
    Z = cache
    
    a = 1/(1 + np.exp(-Z))
    
    a = np.array(a)
      
    g = np.multiply(a,1-a)
    
    dZ = np.multiply(dA,g)
    
    return dZ


def relu_backward(dA, cache):#linear_activation_backward calls this   
    Z = cache
    back_value=np.zeros((Z.shape[0],Z.shape[1]),dtype = float)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
             if Z[i][j]>=0:
                    back_value[i][j]=dA[i][j]  
    return back_value 

def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "tanh":
        
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
    
        
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL,phele kya shape tha y ka
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "tanh")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        

    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate , num_iterations = 5000, print_cost=True):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)
    
        grads = L_model_backward(AL, Y, caches)
        if i <3000:
            parameters = update_parameters(parameters, grads, learning_rate)
        else:
            parameters = update_parameters(parameters, grads, learning_rate/2)
        if print_cost and i % 100 == 0:
            print ("after i iterations"+str(cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters, AL


def accuracy (Y_train, Y_predict):
    
    m = Y_train.shape[1]
    correct = 0
    for i in range(m):
        if Y_train[0,i] == Y_predict[0,i]:
            correct = correct + 1 
    
    Acc = (correct/m)*100
    return Acc

def predict(AL, X):
   
    m = X.shape[1]
    Y_Prediction = np.zeros((1,m),dtype = float)
    for i in range(m):
        if AL[0,i] >= 0.5:
            Y_Prediction[0,i] = 1
    
    return Y_Prediction

def model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations , learning_rate , print_cost=True):
    
    parameters , AL_train = L_layer_model(X_train, Y_train, layers_dims, learning_rate , num_iterations = 25000, print_cost=True)    
    
    AL_test , caches = L_model_forward(X_test, parameters)

    Y_prediction_train = predict(AL_train, X_train)
    
    Y_prediction_test = predict(AL_test, X_test)

    Train_Accuracy = accuracy(Y_train,Y_prediction_train)
  
    Test_Accuracy = accuracy(Y_test,Y_prediction_test)
    
    return Train_Accuracy , Test_Accuracy

layers_dims = [X_train.shape[0], 25, 13, 1]
print(model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations = 25000 , learning_rate = 0.002, print_cost=True))