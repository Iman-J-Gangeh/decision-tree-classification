import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from c45 import load_lab_csv


def CV_10fold(data: pd.DataFrame):
    fold_array = []
    fold_y = []
    keys = data.drop(columns=['y']).columns

    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    splits = np.array_split(data.index, 10)

    for idx in splits:
        split = data.loc[idx].reset_index(drop=True)
        fold_array.append(pd.DataFrame(split.drop(columns=['y']), columns=keys))
        fold_y.append(pd.DataFrame(split['y'], columns=['y']))

    return {"data": fold_array, "ground": fold_y}



def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = sorted(pd.Index(y_true).union(pd.Index(y_pred)).unique(), key=str)

    cm = pd.DataFrame(0, index=labels, columns=labels)

    for actual, predicted in zip(y_true, y_pred):
        cm.loc[actual, predicted] += 1

    return cm



def get_thresholds(params):
    if 'Threshold' in params:
        vals = params['Threshold']
    else:
        vals = params.get('InfoGain', [])

    thresholds = []
    for val in vals:
        thresholds.append(float(val))

    if len(thresholds) == 0:
        raise ValueError("No thresholds found in JSON")

    return thresholds



def encode_sets(x_train, x_test, feature_types):
    x_train = x_train.copy()
    x_test = x_test.copy()

    cat_cols = [col for col in x_train.columns if feature_types.get(col) == 'categorical']
    num_cols = [col for col in x_train.columns if feature_types.get(col) == 'numeric']

    if len(num_cols) > 0:
        x_train[num_cols] = x_train[num_cols].apply(pd.to_numeric)
        x_test[num_cols] = x_test[num_cols].apply(pd.to_numeric)

    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        x_train[cat_cols] = enc.fit_transform(x_train[cat_cols].astype(str))
        x_test[cat_cols] = enc.transform(x_test[cat_cols].astype(str))

    return x_train, x_test



def test_tree(x, y, x_test, threshold, feature_types):
    x_train_enc, x_test_enc = encode_sets(x, x_test, feature_types)

    model = DecisionTreeClassifier(
        criterion='entropy',
        min_impurity_decrease=threshold,
        random_state=1
    )

    model.fit(x_train_enc, pd.Series(y['y']))
    y_pred = model.predict(x_test_enc)
    return y_pred



def perform_GS(data, params, labels, feature_types):
    results = []
    x = list(data['data'])
    y = list(data['ground'])
    thresholds = get_thresholds(params)

    for threshold in thresholds:
        overall_true = []
        overall_pred = []

        for i in range(10):
            x_train = pd.concat(x[:i] + x[i + 1:], axis=0, ignore_index=True)
            y_train = pd.concat(y[:i] + y[i + 1:], axis=0, ignore_index=True)
            x_test = x[i].reset_index(drop=True)
            y_test = y[i].reset_index(drop=True)

            predictions = test_tree(x_train, y_train, x_test, threshold, feature_types)

            overall_true.extend(list(y_test['y']))
            overall_pred.extend(list(predictions))

        confusion_matrix = confusion_matrix_multiclass(overall_true, overall_pred, labels)
        overall_TP = confusion_matrix.to_numpy().diagonal().sum()
        num_predictions = len(overall_pred)
        accuracy = overall_TP / num_predictions

        result = {
            'Splitting Metric': 'InfoGain',
            'Threshold': threshold,
            'CV Accuracy': accuracy,
            'CM': confusion_matrix
        }
        results.append(result)

    return pd.DataFrame(results).sort_values(by=['CV Accuracy', 'Threshold'], ascending=[False, True]).reset_index(drop=True)



def save_tree_pic(X, y, feature_types, threshold, out_file):
    ext = os.path.splitext(out_file)[1].lower()
    if ext == '':
        out_file = out_file + '.png'
    elif ext == '.json':
        out_file = out_file[:-5] + '.png'

    X_enc, _ = encode_sets(X, X.copy(), feature_types)

    model = DecisionTreeClassifier(
        criterion='entropy',
        min_impurity_decrease=threshold,
        random_state=1
    )
    model.fit(X_enc, y)

    plt.figure(figsize=(22, 12))
    plot_tree(
        model,
        feature_names=list(X.columns),
        class_names=[str(c) for c in sorted(pd.Series(y).unique(), key=str)],
        filled=False,
        rounded=True,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    return out_file


# Main
if len(sys.argv) not in [3, 4]:
    print("Usage: python3 crossValSKL.py <CSVFile> <HyperparameterGridFile.json> [<OutputTreeFile>]")
    sys.exit(1)

csv_path = sys.argv[1]
thresh_path = sys.argv[2]
print("Reading JSON of hyperparams from " + thresh_path)
if len(sys.argv) > 3:
    out_file = sys.argv[3]
else:
    out_file = None

with open(thresh_path, 'r') as params_file:
    params_json = json.load(params_file)

X, y, feature_types = load_lab_csv(csv_path)

test_set = X.copy().reset_index(drop=True)
test_set['y'] = y.reset_index(drop=True)
labels = sorted(list(pd.Series(y).unique()), key=str)

folded = CV_10fold(test_set)
best_params = perform_GS(folded, params_json, labels, feature_types)
best = best_params.iloc[0]

print("Best Model:")
print("Splitting Metric: Information Gain")
print("Threshold: " + str(best['Threshold']))
print("CV Accuracy: " + str(best['CV Accuracy']))
print("Confusion Matrix:")
print(best['CM'])

if out_file is not None:
    saved_file = save_tree_pic(X, y, feature_types, float(best['Threshold']), out_file)
    print("Saved tree to " + saved_file)
