#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:07:35 2020

@author: sridhar
"""
import numpy as np
import pandas as pd
import json
import string
import pickle

from collections import Counter
from sklearn.metrics import accuracy_score


def preprocessing(s):
    s = s.strip().lower()
    new = ""
    letters = string.ascii_lowercase
    
    for i in s:
        if i in letters or i==" ":
            new += i
        
    return ' '.join(new.split()) 

"""
############################
# Methods for making Model #
############################

def class_wise_words_frequency_dict(X, Y):
    dic = {}
    Y = Y.values.reshape(len(Y),)
    X = X.values
    for i in range(len(Y)):
        dic[Y[i]] = dic.get(Y[i], []) + [X[i][0]]
        
    for c in dic:
        dic[c] = dict(Counter((' '.join(dic[c])).split()))
        
    return dic

def compute_prior_probabilities(Y):
    c = Counter(Y.values.reshape(len(Y),))
    dic = {}

    for i in c:
        dic[i] = c[i]/len(Y)
        
    return dic

def get_class_wise_denominators_likelihood(X, Y):
    class_wise_denominators = dict()
    vocabulary = []
    
    for c in classes:
        frequency_dict = class_wise_frequency_dict[c]
        class_wise_denominators[c] = sum(list(frequency_dict.values()))
        vocabulary += list(frequency_dict.keys())
    
    vocabulary = list(set(vocabulary))

    for c in classes:
        class_wise_denominators[c] += len(vocabulary)

    return class_wise_denominators
"""

def compute_likelihood(test_X, c, class_wise_frequency_dict,
                       class_wise_denominators):
    likelihood = 0    
    words = test_X.split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + 1)/class_wise_denominators[c])
        
    return likelihood

def predictClass(test_X, model):
    best_p = -99999
    best_c = -1
    classes = {0,1}
    
    for c in classes:
        p = compute_likelihood(test_X, c, model['class_wise_frequency_dict'], model['class_wise_denominators']) + np.log(model['prior_probabilities'][c])
        if p > best_p:
            best_p = p
            best_c = c
            
    return best_c

'''
#############################
# Code to make the NB Model #
#############################

trainX = pd.read_csv("./train_X_nb.csv", header=None)
trainY = pd.read_csv("./train_Y_nb.csv", header=None)
trainX[0] = trainX[0].apply(preprocessing)

classes = sorted(list(set(trainY.values.reshape(len(trainY),)))) 
class_wise_frequency_dict = class_wise_words_frequency_dict(trainX, trainY)
class_wise_denominators = get_class_wise_denominators_likelihood(trainX, trainY)
prior_probabilities = compute_prior_probabilities(trainY)

model = {"class_wise_frequency_dict": dict(class_wise_frequency_dict),
         "class_wise_denominators": dict(class_wise_denominators),
         "prior_probabilities": dict(prior_probabilities)}


###############################
# Saving Model as a pickle file #
###############################
    
pickle.dump(model, open("MODEL_FILE.sav","wb"))
    
'''