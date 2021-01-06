import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

truth = [False, True, True, True, True, False, True, True, True, True, True]
scale = [2,4,6]

num = int(11**0.5)

X = pd.read_csv("./train_X_p1.csv")
Y = pd.read_csv("./train_Y_p1.csv", header=None)

scaler = StandardScaler()
columns = X.columns

def preprocess_data(file):
    for i in columns[scale]:
        k = scaler.fit(X[i].values.reshape(len(X[i]),1))
        l = k.transform(file[i].values.reshape(len(file[i]),1))
        file[i] = l
        
    return file[columns[truth]]