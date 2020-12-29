# -*- coding: utf-8 -*-
import numpy as np
import csv

def import_data():
    X = np.genfromtxt("./train_X_lr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("./train_Y_lr.csv", delimiter=',', dtype=np.float64)
    
    return X,Y

def compute_gradient(X, Y, W):
    h = np.dot(X, W)
    p = h-Y
    gradient = np.dot(p.T,X)/len(X)
    
    return gradient.T

def compute_cost(X, Y, W):
    h = np.dot(X, W)
    mse = sum((h-Y)**2)
    cost = mse/(2*len(X))
    
    return cost

def optimize_weights(X, Y, W, n, alpha):
    
    a = np.dot(X.T,X)
    a = np.linalg.inv(a)
    a = np.dot(a,X.T)
    a = np.dot(a,Y)
    W = a
        
    return W

def train_model(X, Y):
    X = np.insert(X, 0, 1, axis=1)
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1], 1))
    W = optimize_weights(X, Y, W, 10**10, 0.0002)
    
    return W

def save_model(weights, file_name):
    with open(file_name, 'w') as file:
        wr = csv.writer(file)
        wr.writerows(weights)
        file.close()

if __name__=="__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")