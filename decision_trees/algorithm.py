import pandas as pd
import numpy as np
import os
import math
import re
import matplotlib.pyplot as plt
import pickle
import csv

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import *
from sklearn.tree import DecisionTreeClassifier


def arff2csv(file):
    features = []
    data = []
    out = file[:-5]+".csv"
    with open(file, "r") as file:
        for row in file.readlines():
            if "@ATTRIBUTE" in row:
                feat = re.search('BUTE (.+){', row)
                string = str(feat[1])
                string = string.replace('"', '').strip()
                features.append(string)
                
            elif row[0] != "@":
                row = row.strip()
                if row:
                   data.append(row.split(","))
                   
            elif "@RELATION" in row:
                name = str(row.strip())
    
    
    output = pd.DataFrame(data=data, columns=features)
    output.to_csv(out, index=False)
    
    return out,name


def encode(X2, name):
    f = X2[name].values
    feat = []
    for i in f:
        feat.append([i])
        
    ordinal = OrdinalEncoder()
    ordinal.fit(feat)
    newFeat = ordinal.transform(feat)
    
    return newFeat


######################
# Data Preprocessing #
######################

def preprocess_data(file):
    trainX = pd.read_csv(file)
    columns = np.array(trainX.columns)
    X = trainX[columns[truth]]
    X = X.drop(columns = ["Num Pin Dot Pattern Views"])
    X.to_csv("./modified_trainX.csv", index=False)
    
    X2 = pd.read_csv("./modified_trainX.csv")
    col_names = np.array(X2.columns)
        
    
    enc = []
    for i in col_names:
        k = list(set(np.array(X2[i].values)))
        X2[i] = X2[i].fillna(k[-1])
        
    for i in col_names:
        arr = encode(X2, i)
        enc.append(arr.reshape(len(arr),))
    
    temp = str(file).split("/")[-1]
    
    for i in range(len(col_names)):
        X2[col_names[i]] = enc[i]
        
    X2.to_csv("./processed_"+temp, index=False)
    
    return "./processed_"+temp

#############################################################
# Truth Array ignoring features with significant data       #
# in the form of "?" or "NULL" (created with help of Excel) #
#############################################################

truth = np.array([False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, True, True, False, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, True, True, True, True, True, True, True, True, True, True, 
         True, True, False, False, False, False, False, False, False, False, False, 
         False, False, False, False, False, False, False, False, False, False, 
         True, False, True, True, True, False, True, True, True, True, True, True])
#############################################################


class convert(object):

    content = []
    name = ''

    def __init__(self, file, output, relation):
        self.file = file
        self.output = output
        self.relation = relation
        self.csvInput()
        self.arffOutput()
        

    #import CSV
    def csvInput(self):

        user = self.file

        #remove .csv
        if user.endswith('.csv') == True:
            self.name = self.relation
            
        #print ('Opening CSV file.')     
        try:
            with open(user, 'r') as csvfile:
               lines = csv.reader(csvfile, delimiter = ',')
               for row in lines:
                   self.content.append(row)
            csvfile.close()
            #sleep(2) #sleeps added for dramatic effect!
            
        #just in case user tries to open a file that doesn't exist
        except IOError:
            sleep(2)
            print ('File not found.\n')
            self.csvInput()
            
    #export ARFF
    def arffOutput(self):
        #print ('Converting to ARFF file.\n')
        title = self.output
        new_file = open(title, 'w')

        ##
        #following portions formats and writes to the new ARFF file
        ##

        #write relation
        new_file.write('@RELATION ' + str(self.name)+ '\n\n')

        #get attribute type input
        '''for i in range(len(self.content[0])-1):
            attribute_type = input('Is the type of ' + str(self.content[0][i]) + ' numeric or nominal? ')
            new_file.write('@ATTRIBUTE ' + str(self.content[0][i]) + ' ' + str(attribute_type) + '\n')'''

        #create list for class attribute
        last = len(self.content[0])
        class_items = []
        for i in range(len(self.content)):
            name = self.content[i][last-1]
            if name not in class_items:
                class_items.append(self.content[i][last-1])
            else:
                pass  
        del class_items[0]
    
        string = '{' + ','.join(sorted(class_items)) + '}'
        new_file.write('@ATTRIBUTE ' + str(self.content[0][last-1]) + ' ' + str(string) + '\n')

        #write data
        new_file.write('\n@DATA\n')

        del self.content[0]
        for row in self.content:
            new_file.write(','.join(row) + '\n')

        #close file
        new_file.close()
        

'''
arff2csv("./train_X_dt.arff")
arff2csv("./train_Y_dt.arff")

trainX = pd.read_csv("./train_X_dt.csv")
trainY = pd.read_csv("./train_Y_dt.csv")
columns = np.array(trainX.columns)

preprocess_data(trainX)
'''

"""
#######################
# Decision Tree Model #
#######################

X = pd.read_csv("./processed_trainX.csv")
trainY = pd.read_csv("./train_Y_dt.csv")

clf = DecisionTreeClassifier()
clf.fit(X, trainY)
pred_Y = clf.predict(X)
print(accuracy_score(pred, trainY))
"""
'''
X = pd.read_csv("./processed_trainX.csv")
trainY = pd.read_csv("./train_Y_dt.csv")
model = pickle.load(open("MODEL_FILE.sav", 'rb'))
pred = model.predict(X)
accuracy_score(pred, trainY)
'''