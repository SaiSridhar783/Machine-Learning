import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b, Lambda=0):
    m = len(X)
    regular = Lambda*(np.sum(np.square(W)))/(2*m)
    z = sigmoid(np.dot(X,W) + b)
    z[z==1] = 0.9999999
    z[z==0] = 0.0000001 
    
    J = (-1/m) * np.sum(Y * np.log(z) + (1-Y) * np.log(1-z))
    cost = (J + regular)
    
    return cost

def compute_gradients_of_cost_function(X, Y, W, b, Lambda=0):
    m = len(X)
    dB = (np.sum(np.dot(X,W)+b - Y))/m
    dW = (np.dot(X.T,(np.dot(X,W)+b - Y)))/m
    regular = Lambda*W
    dW += regular
    return dW, np.array(dB)

def optimize_weights(X, Y, W, b, alpha):
    m = len(Y)
    prev = 0
    iter_no = 0
    
    while True:
        iter_no += 1
        dW, db = compute_gradients_of_cost_function(X, Y, W, b)
        W -= (alpha/m)*dW
        b -= (alpha/m)*db
        cost = compute_cost(X, Y, W, b)

        if iter_no%1000==0:
            print(iter_no, cost)
            
        t = (prev-cost).values
        if abs(t[0])< 0.0000001:
            print(iter_no, cost)
            break
        
        prev = cost
    
    return W, b

def predict_labels(X, W, b):
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    arr = []
    for i in A:
        if i>=0.5:
            arr.append(1)
            
        else:
            arr.append(0)
            
    return np.array(arr)

"""
trainX = pd.read_csv("./processed_train_data.csv")
trainY = pd.read_csv("./train_Y_pr.csv", header=None)
features = trainX.shape[1]

weights = np.ones((features,1))
b = 1

W,b = optimize_weights(trainX, trainY, weights, b, 0.1)

pred_Y = predict_labels(trainX, W, b)

f1 = f1_score(trainY, pred_Y, average="weighted")

arr = np.append(W,b)
pd.DataFrame(arr).to_csv("./WEIGHTS_FILE.csv", index=False)
print(f1)
"""