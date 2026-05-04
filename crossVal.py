import sys
import pandas as pd
import json
from pprint import pprint
import subprocess
from c45 import predict, fit


def CV_10fold(data: pd.DataFrame):
    fold_array = []
    fold_y = []
    keys = data.drop(columns=['y']).columns

    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    split_index = int(len(data) * 0.1)

    for i in range(10):
        split = data.iloc[ (i * split_index) : ((i + 1) * split_index) ]
        fold_array.append( pd.DataFrame(split.drop(columns=['y']), columns=keys) )
        fold_array.append( pd.DataFrame(split['y'], columns=['y']) )

    return {"data": fold_array, "ground": fold_y}


def test_tree(x, y, x_test, y_test, hyperparams):
    model = c45()
    y_pred = model.predict(x_test)

    results.append(evaluate(y_test, y_pred))

def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = sorted(pd.Index(y_true).union(pd.Index(y_pred)).unique())

    cm = pd.DataFrame(0, index=labels, columns=labels)

    for actual, predicted in zip(y_true, y_pred):
        cm.loc[actual, predicted] += 1

    return cm


def class_counts_from_cm(cm, cls):
    TP = cm.loc[cls, cls]
    FP = cm[cls].sum() - TP
    FN = cm.loc[cls].sum() - TP
    TN = cm.values.sum() - TP - FP - FN

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

def perform_GS(data, params, output=None):
    results = pd.DataFrame(columns=['Splitting Metric', 'Threshold', 'CV Accuracy', 'CM'])
    x = data['data']   # list of dataframes
    y = data['ground']

    IG = params['InfoGain']
    R = params['Ratio']
    Th = params['Threshold']

    for threshhold in Th:
        for g in IG:
            for i in range(10):
                x_train = data[-i:] + data[-(10 - (i + 1)):]
                y_train = data[-i:] + data[-(10 - (i + 1)):]
                x_test = data[i]
                y_test = data[i]


        for r in R:
            for i in range (10):
                x_train = data[-i:] + data[-(10 - (i+1)):]
                y_train = data[-i:] + data[-(10 - (i+1)):]
                x_test = data[i]
                y_test = data[i]

    return


# Main
test_set_path = sys.argv[1]
thresh_path = sys.argv[2]
print("Reading JSON of hyperparams from " + thresh_path)
if (len(sys.argv) > 3):
    out_file = sys.argv[3]
else:
    out_file = None

with open(thresh_path, 'r') as params_file:
    params_json = json.load(params_file)

test_set = pd.read_csv(test_set_path, skiprows=[1,2])
if "iris" in test_set_path:
    ground_truth = test_set['species']
    test_set = test_set.drop('species', axis=1)
elif "nursery" in test_set_path:
    ground_truth = test_set['class']
    test_set = test_set.drop('class', axis=1)
elif "letter" in test_set_path:
    ground_truth = test_set['lettr']
    test_set = test_set.drop('lettr', axis=1)
else:
    ground_truth = test_set['Inflated']
    test_set = test_set.drop('Inflated', axis=1)

test_set['y'] = ground_truth

folded = CV_10fold(test_set) # folded -> dict["data", "ground"]
pprint(params_json['InfoGain'])
# perform_GS(folded, params_json)
