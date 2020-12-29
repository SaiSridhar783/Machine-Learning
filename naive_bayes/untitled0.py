import numpy as np
import csv
import sys
import json
import math
from collections import defaultdict
from validate import validate


def preprocessing(X):
    result = []
    for s in X:
        S = s
        for i in range(len(S)):
            if not(S[i].isalpha()) and not(S[i] == " "):
                s = s.replace(S[i], "")
        new = s.strip()
        new = new.lower()
        while not(new.find("  ") == -1):
            i = new.find("  ")
            new = new[0:i] + new[i+1::]
        result.append(new)
    return result


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    with open(model_file_path, "r") as read_file:
        model = json.load(read_file)
    return test_X, model


def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators):
    clas = defaultdict(int,class_wise_frequency_dict[c])
    llk = 0
    for x in test_X.split(" "):
        num = clas[x]+1
        llk += math.log(num/(class_wise_denominators[c]))
    return llk


def predict_target_values(test_X, model):
    max_llk = -math.inf
    [classes, class_wise_frequency_dict, class_wise_denominators] = model
    y = []
    for x in test_X:
        for c in classes:
            llk = compute_likelihood(x, c, class_wise_frequency_dict, class_wise_denominators)
            if llk > max_llk:
                max_llk = llk
                predicted_class = c
        y.append(predicted_class)
    return np.array(y)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, "MODEL_FILE.json")
    pred_Y = predict_target_values(preprocessing(test_X), model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    #test_X_file_path = sys.argv[1]
    test_X_file_path = "train_X_nb.csv"
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate("train_X_nb.csv", actual_test_Y_file_path="train_Y_nb.csv")