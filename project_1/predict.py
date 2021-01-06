import numpy as np
import pandas as pd
import csv
import sys
import pickle

from algorithm import *
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'
Writes the predicted values to the file named "predicted_test_Y_lr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, model_file):
    test_X = pd.read_csv(test_X_file_path)
    test_X = preprocess_data(test_X)
    
    model = pickle.load(open(model_file,"rb"))
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    pred_Y = model.predict(test_X)
    
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_weights(test_X_file_path, "MODEL_FILE.sav")
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_p1.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_p1.csv") 