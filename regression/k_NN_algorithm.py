#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:02:16 2020

@author: sridhar
"""

import numpy as np
import pandas as pd
import math
import csv
import pickle

from collections import Counter
from sklearn.model_selection import train_test_split


def compute_ln_norm_distance(vector1, vector2, n):
    res = 0
    for i in range(len(vector1)):
        res += abs((vector1[i]-vector2[i]))**n

    return res**(1/n)


def find_k_nearest_neighbors(train_X, trainY, test_example, k, n):
    arr = []
    count = 0
    for i in train_X:
        dist = compute_ln_norm_distance(i, test_example, n)
        arr.append((dist, trainY[count], i))
        count += 1

    arr.sort(key=lambda x: -x[0])
    X = []
    Y = []
    for i in range(k):
        Y.append(arr[i][1])
        X.append(arr[i][2])

    return np.array(X), np.array(Y)


def get_WeightMatrix_for_LOWES(query_point, Training_examples, Bandwidth):
    # M is the No of training examples
    M = Training_examples.shape[0]
    # Initialising W with identity matrix
    W = np.mat(np.eye(M))
    # calculating weights for query points
    for i in range(M):
        xi = Training_examples[i]
        denominator = (-2 * Bandwidth * Bandwidth)
        W[i, i] = np.exp(np.dot((xi-query_point), (xi-query_point).T)/denominator)
        return W


def predict_Y(Training_examples, Y, query_x, Bandwidth):
    M = Training_examples.shape[0]
    all_ones = np.ones((M, 1))
    X_ = np.hstack((Training_examples, all_ones))
    qx = np.mat(np.append(query_x, 1))
    
    W = get_WeightMatrix_for_LOWES(qx, X_, Bandwidth)
    # calculating parameter theta
    theta = np.linalg.pinv(X_.T*(W * X_))*(X_.T*(W * Y))
    # calculating predictions
    pred = np.dot(qx, theta)
    return theta, pred


def check_mse(actual_Y, pred_Y):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(actual_Y, pred_Y)
    print("MSE", mse)

    return mse

"""
########################
# Trying out the model #
########################

X = pd.read_csv("./train_X_re.csv").values
Y = pd.read_csv("./train_Y_re.csv", header=None).values

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1,
                                                random_state=1)


Bandwidth = 10
Y_test = []

for query in testX:
    x,y = find_k_nearest_neighbors(trainX, trainY, query, 83, 1)
    theta, pred = predict(x, y, query, Bandwidth)
    Y_test.append(pred[0][0])

pred = np.array(Y_test).reshape(len(Y_test),1)
check_mse(Y,pred)
"""