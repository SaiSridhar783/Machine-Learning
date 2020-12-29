import numpy as np
import pandas as pd
import csv
import sys

from algorithm import *
from GD import *
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X, weights


def predict_target_values(test_X, weights):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    
    preprocess(test_X,"./test_X_processed.csv")
    testX = pd.read_csv("./test_X_processed.csv")
    """
    Note:
    The preprocessing techniques which are used on the train data, should also be applied on the test 
    1. The feature scaling technique used on the training data should be applied as it is (with same mean/standard deviation/min/max) on the test data as well.
    2. The one-hot encoding mapping applied on the train data should also be applied on test data during prediction.
    3. During training, you have to write any such values (mentioned in above points) to a file, so that they can be used for prediction.
     
    You can load the weights/parameters and the above mentioned preprocessing parameters, by reading them from a csv file which is present in the preprocessing_regularization.zip
    """
    
    # Predict Target Variables
    
    W, b = weights[:-1], weights[-1]
    pred_Y = predict_labels(testX, W, b)
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in preprocessing_regularization.zip and imported properly.
    """
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X_file_path, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 