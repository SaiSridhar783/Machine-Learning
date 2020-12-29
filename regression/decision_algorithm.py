import numpy as np
import pandas as pd
import math
import pickle

def rss(y_left, y_right):
    def squared_residual_sum(y):
        return np.sum((y - np.mean(y)) ** 2)
    
    return (squared_residual_sum(y_left) + squared_residual_sum(y_right))

def find_best_rule(trainX, trainY):
    best_feature, best_threshold, best_rss = 0, 0, 10**10
    
    for feature in trainX.columns:
        thresholds = trainX[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        
        for t in thresholds:
            y_left_ix = trainX[feature] < t
            y_left, y_right = trainY[y_left_ix], trainY[~y_left_ix]
            t_rss = rss(y_left, y_right)
            t_rss = t_rss.values[0]

            if t_rss < best_rss:
                best_rss = t_rss
                best_threshold = t
                best_feature = feature
                
    return {'feature':best_feature, 'threshold':best_threshold}


def split(trainX, trainY, depth, max_depth=None):
    if depth == max_depth or len(trainX) < 2:
        return {'prediction': np.mean(trainY).values[0]}

    rule = find_best_rule(trainX, trainY)
    left_ix = trainX[rule['feature']] < rule['threshold']
    rule['left'] = split(trainX[left_ix], trainY[left_ix], depth+1, max_depth)
    rule['right'] = split(
        trainX[~left_ix], trainY[~left_ix], depth+1, max_depth)
    return rule

def predict_Y(sample, rules):
    prediction = 0
    while not prediction:
        feature, threshold = rules['feature'], rules['threshold']
        if sample[feature].values[0] < threshold:
            rules = rules['left']
            
        else:
            rules = rules['right']
            
        prediction = rules.get('prediction', 0)
        
    return prediction


"""
X = pd.read_csv("./train_X_re.csv")
Y = pd.read_csv("./train_Y_re.csv", header=None)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1,
                                                random_state=69)


rules = split(trainX, trainY, 0, 7)

arr = []
for i in range(len(testX)):
    check = testX[i:i+1]
    pred = predict_Y(check, rules)
    arr.append(pred)
    
check_mse(testY, np.array(arr))

"""