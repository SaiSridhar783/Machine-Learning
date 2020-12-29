import numpy as np
import pandas as pd
import math
import pickle

from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def gini(S):
    c = Counter(S)
    m = len(S)
    t = 0
    for i in c:
        t += (c[i]/m)**2
        
    return 1-t

def calculate_gini_index(Y_subsets):
    m = 0
    for i in Y_subsets:
        m += len(i)
    
    gin = 0 
    for i in Y_subsets:
        coef = len(i)/m
        gin += coef*gini(i)
        
    return gin

def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])
    
    return left_X, left_Y, right_X, right_Y

def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 99999
    best_feature = 0
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = sorted(set(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
                
    return best_feature, best_threshold

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def construct_tree(X, Y, max_depth, min_size, depth):
    Y2 = np.array(Y)
    classes = list(set(Y2))
    predicted_class = classes[np.argmax([np.sum(Y2 == c) for c in classes])]
    node = Node(predicted_class, depth)

    #check is pure
    if len(set(Y)) == 1:
        return node
    
    #check max depth reached
    if depth >= max_depth:
        return node

    #check min subset at node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y)

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth + 1)
    
    return node

def print_tree(node):
    if node.left is not None and node.right is not None:
        print("X" + str(node.feature_index) + " " + str(node.threshold))
        
    if node.left is not None:
        print_tree(node.left)
        
    if node.right is not None:
        print_tree(node.right)
        
def predict_class(root, X):
    node = root #root is a tree constructed using construct_tree method
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
            
    return node.predicted_class

'''
trainX = pd.read_csv("./train_X_de.csv")
trainY = pd.read_csv("./train_Y_de.csv", header=None, names=["Class"])
X = np.array(trainX)
Y = np.array([i[0] for i in trainY.values])

#root = construct_tree(X, Y, 5, 1, 0)
#pickle.dump(root, open("MODEL_FILE.sav","wb"))
#print_tree(root)
root = pickle.load(open("./MODEL_FILE.sav","rb"))
predY = [predict_class(root, X[i]) for i in range(len(X))]
'''