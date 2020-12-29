import numpy as np
import pandas as pd
import math
import csv
from collections import Counter

def compute_ln_norm_distance(vector1, vector2, n):
    #TODO Complete the function implementation. Read the Question text for details
    res = 0
    for i in range(len(vector1)):
        res += abs((vector1[i]-vector2[i]))**n
        
    return res**(1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    arr = []
    t = 0
    for i in train_X:
        dist = compute_ln_norm_distance(i, test_example, n)
        arr.append((dist,t))
        t += 1
        
    arr.sort()
    lis = []
    for i in range(k):
        lis.append(arr[i][1])
        
    return lis

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    arr = []
    for i in test_X:
        arr.append(find_k_nearest_neighbors(train_X, i, k, n))
        
    lis = []
    for i in arr:
        for j in range(len(i)):
            i[j] = train_Y[i[j]]
            
        c = Counter(i)
        c = sorted(c.items(), key= lambda x: -x[1])
        lis.append(c[0][0])
        
    return lis

def check_weighted_f1_score(actual_Y, pred_Y):
    #pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=',', dtype=np.int)
    #actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=',', dtype=np.int)
    from sklearn.metrics import f1_score
    weighted_f1_score = f1_score(actual_Y, pred_Y, average = 'weighted')
    #print("Weighted F1 score", weighted_f1_score, k)
    return weighted_f1_score

def get_best_k_using_validation_set(trainX, trainY, testX, testY, split, n):
    
    best_k = -1
    best_accuracy = 0
    for k in range(1, (split+1)//2):
        predicted_Y = classify_points_using_knn(trainX,trainY,testX, n,k)
        accuracy = check_weighted_f1_score(testY, predicted_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k

'''
if __name__=="__main__":
    trainX = np.genfromtxt("./train_X_knn.csv", delimiter=',', dtype=np.float64,
                           skip_header=1)
    trainY = np.genfromtxt("./train_Y_knn.csv", delimiter=',', dtype=np.float64)
    
    total = len(trainX)
    split = math.floor(float(98)/100 * total)
    
    trainX, testX = trainX[:split], trainX[split:]
    trainY, testY = trainY[:split], trainY[split:]
    
    k = get_best_k_using_validation_set(trainX, trainY, testX, testY, split, 6)
    print(k)
    arr = classify_points_using_knn(trainX, trainY, testX, k, 3)
    score = check_weighted_f1_score(arr, testY)
    print(score)'''