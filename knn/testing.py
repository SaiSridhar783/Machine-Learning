'''import numpy as np
import csv
import sys

from algorithm import *
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64,
                           skip_header=1)
    
    return test_X


def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    trainX = np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64,
                            skip_header=1)
    trainY = np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.float64)
    
    predicted_Y = classify_points_using_knn(trainX, trainY, test_X, 6, 6)
    
    return np.array(predicted_Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") '''
    
import matplotlib.pyplot as plt
import pandas as pd

file = pd.read_csv("./train_X_knn.csv")
file2 = pd.read_csv("train_Y_knn.csv", header = None)
print(len(file2), len(file))
'''
plt.scatter(file["Chlorine"], file["Hardness"], c="cyan")
plt.scatter(file["Hydrogen"], file["Hardness"], c="teal")
plt.scatter(file["Sulphur"], file["Hardness"], c="magenta")
plt.scatter(file["Carbon"], file["Hardness"], c="red")
plt.scatter(file["Oxygen"], file["Hardness"], c="green")
plt.scatter(file["Nitrogen"], file["Hardness"], c="blue")'''

#plt.scatter(file2, file["Nitrogen"], c="black")
plt.plot(file)
