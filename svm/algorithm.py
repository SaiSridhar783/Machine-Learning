import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def gaussian_kernel(X1, X2, sigma):
    numerator = -np.linalg.norm(X1-X2)**2
    denominator = 2*(sigma**2)
    similarity = np.exp(numerator/denominator)
    return similarity

'''
X = np.round(pd.read_csv("./train_X_svm.csv"),3)
Y = np.array(pd.read_csv("./train_Y_svm.csv", header=None))

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.22, random_state=4)
trainY = np.array([i[0] for i in trainY])

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.33, random_state=69)
trainY = trainY.reshape(len(trainY),)

model = SVC(kernel="rbf", C=10)
model.fit(trainX, trainY)

pred_Y = model.predict(testX)
print(accuracy_score(pred_Y,testY))
'''