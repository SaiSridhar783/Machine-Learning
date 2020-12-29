import numpy as np
import pandas as pd
import csv
import multiprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from validate import validate

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b 
    A = sigmoid(Z)
    A[A==1] = 0.9999999
    A[A==0] = 0.0000001 
    cost = -1/m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A))) 
    return cost

def compute_gradients_of_cost_function(X, Y, W, b):
    m = len(X)
    
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    dW = 1/m * np.dot((A-Y).T, X)
    db = 1/m * np.sum(A-Y)
    dW = dW.T
    return dW, db

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
        
        if iter_no%200000==0:
            print(iter_no, cost)
            
        if abs(prev-cost)< 0.00000001:
            print(iter_no, cost)
            break
        
        prev = cost
        
    return W, b

def get_train_data_for_class(trainX, trainY, classy):
    X = np.copy(trainX)
    Y = np.copy(trainY)
    Y = np.where(Y==classy, 1, 0)
    
    return X, Y

def save_model(weights, b, file_name):
    with open(file_name, 'w') as file:
        wr = csv.writer(file)
        wr.writerows(weights)
        wr.writerow(b)
        file.close()


data = pd.read_csv("./train_X_lg_v2.csv")
trainX, testX = data, data[1000:]

data2 = pd.read_csv("./train_Y_lg_v2.csv", header=None)
trainY, testY = data2, data2[1000:]

b = -1.657476622996
W = np.ones((20,1))


X,Y = get_train_data_for_class(trainX, trainY, 2)

#W,b = optimize_weights(X, Y, W, b, 0.03)
#print(W,b)
'''
# Check F1 Score of optimized weights

W = np.array([[0.133216773150504],
 [-0.01361072631331356],
 [-0.211179200078481],
 [ 0.02061023821441409],
 [-0.2127428769768107],
 [ 0.19284189311205482],
 [ 1.9801136628857483],
 [-0.06861081335989366],
 [-0.1102711968303362],
 [-0.05284630089080014],
 [-0.25967688800692046],
 [-0.18738485620444165],
 [-0.10110561847198067],
 [ 0.1248136059241448],
 [ 0.28476542739828326],
 [-0.007387711237913287],
 [-0.03651524804201174],
 [-0.17184332179961745],
 [-0.24044812879586766],
 [ 0.5286479836109026]])

val = pd.read_csv("./Weights0.csv", header=None)

#W = val[:-1]
#b = val.values.tolist()[-1]

b = -1.657476622996

arr = []
predY = predict_labels(X, W, b)

print(f1_score(Y, predY, average="weighted"))
print(precision_score(Y, predY, average="weighted"))
print(recall_score(Y, predY, average="weighted"))

print(np.array([-1.6210151970207245,
 0.09821578668598152,
 0.07706068362392104,
 -0.305555425936915,
 0.11155760711186403,
 -0.17054916143900825,
 0.14802673389360116,
 2.0490821578194836,
 -0.10499899112427635,
 0.051986185720389114,
 -0.29188525176291336,
 -0.3061273245794543,
 -0.07190451166120723,
 -0.05801381813484611,
 0.03336053781544581,
 0.2340834012154903,
 -0.006878888886267364,
 -0.05825998981533747,
 -0.14599678136455635,
 -0.31598324705220354,
 0.4871371597089672]))

'''
'''
# Validate trained model on given dataset

data = pd.read_csv("./train_X_lg_v2.csv")
data2 = pd.read_csv("./train_Y_lg_v2.csv", header=None)
data2 = [i[0] for i in data2.values.tolist()]


lis = []
for i in data.values:
    arr = [] 
    for j in range(4):
        file = pd.read_csv("./Weights"+str(j)+".csv", header = None)
        file, b = file[:-1], file.values[-1:]
        
        h = np.dot(file.T,i) + b
        h = sigmoid(h)
        #W, b = file.values.tolist()[:-1], file.values.tolist()[-1]
        arr.append((h,j))
        #lis = predict_labels(data, W, b)
        #arr.append(sum(lis))
        
    
    arr.sort(reverse=True)
    lis.append((arr[0][1]))

lis = np.array(lis)

pred_Y = lis.reshape(len(lis), 1)
with open("predicted_test_Y_lg.csv", 'w', newline='') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerows(pred_Y)
    csv_file.close()

   
validate("train_X_lg_v2.csv", actual_test_Y_file_path="train_Y_lg_v2.csv")

'''
'''
# Saving all weights in one file
arr = pd.DataFrame()

for i in range(4):
    file = pd.read_csv("./Weights"+str(i)+".csv", header = None)
    file = file.values
    file = [i[0] for i in file]
    name = "Class "+str(i)
    arr[str(name)] = file
    
print(arr)

arr.to_csv("./WEIGHTS_FILE.csv", index=False)
'''
'''
test_X = np.genfromtxt("./train_X_lg_v2.csv", delimiter=',', dtype=np.float64, skip_header=1)
weights = np.genfromtxt("WEIGHTS_FILE.csv", delimiter=',', dtype=np.float64)

weights, B = weights[:-1], weights[-1]
weights = np.array(weights)
weights = weights.reshape((4,20))
test_X = np.array(test_X)

lis = []
for i in test_X:
    arr = [] 
    for j in range(4):
        W, b = weights[j], B[j]
        h = np.dot(W.T,i) + b
        h = sigmoid(h)
        arr.append((h,j))
        print(arr)
    
    arr.sort(reverse=True)
    lis.append((arr[0][1]))

pred_Y = np.array(lis)
'''
