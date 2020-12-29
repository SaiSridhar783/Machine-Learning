import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def replace_null_values_with_mean(X):
    #Obtain mean of columns
    col_mean = np.nanmedian(X, axis=0)
    col_mean = np.round(col_mean, 3)
    
    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices.
    for i in range(len(inds[0])):
        a,b = inds[0][i], inds[1][i]
        X[a,b] = col_mean[b]
    
    return X

def convert_to_numerical_labels(X):
    dic = {}
    k = sorted(list(set(X)))
    for i in range(len(k)):
        dic[k[i]] = i
        
    for i in range(len(X)):
        X[i] = dic.get(X[i])
        
    return X

def mean_normalize(X, column_indices=[2,4]):
    col_mean = np.mean(X, axis=0)
    X_transpose = X.T
    
    for i in column_indices:
        a = min(X_transpose[i])
        b = max(X_transpose[i])
        X_transpose[i] = (X_transpose[i]-col_mean[i])/(b-a)
        
    return np.round(X_transpose.T, 3)

def min_max_normalize(X, column_indices):
    col_mean = np.mean(X, axis=0)
    X_transpose = X.T
    
    for i in column_indices:
        a = min(X_transpose[i])
        b = max(X_transpose[i])
        X_transpose[i] = (X_transpose[i]-a)/(b-a)
        
    return np.round(X_transpose.T,3)

def standardize(X, column_indices):
    col_mean = np.mean(X, axis=0)
    X_transpose = X.T
    
    for i in column_indices:
        std_dev = np.std(X_transpose[i])
        X_transpose[i] = (X_transpose[i]-col_mean[i])/std_dev
        
    return np.round(X_transpose.T,3)

def preprocess(file, output="./processed_train_data.csv"):
    data = pd.read_csv(file)
    headers = data.columns
    data = np.array(data)
    data = replace_null_values_with_mean(data)
    data = min_max_normalize(data,[2,4,5])
    data = pd.DataFrame(data, columns = headers)
    
    data.to_csv(output, index=False)
    
#preprocess("./train_X_pr.csv")
