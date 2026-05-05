import os
import sys
import json

import numpy as np
import pandas as pd

from c45 import c45, load_lab_csv


DATASET = ""


def _predict_one_safe(row, node):
    while "leaf" not in node:
        attr = node["var"]
        edges = node["edges"]
        matched = False

        for edge_obj in edges:
            edge = edge_obj["edge"]

            if "op" in edge:
                if edge["op"] == "<=" and row[attr] <= edge["value"]:
                    if "node" in edge:
                        node = edge["node"]
                    else:
                        node = {"leaf": edge["leaf"]}
                    matched = True
                    break
                elif edge["op"] == ">" and row[attr] > edge["value"]:
                    if "node" in edge:
                        node = edge["node"]
                    else:
                        node = {"leaf": edge["leaf"]}
                    matched = True
                    break
            else:
                if row[attr] == edge["value"]:
                    if "node" in edge:
                        node = edge["node"]
                    else:
                        node = {"leaf": edge["leaf"]}
                    matched = True
                    break

        if not matched:
            return None

    return node["leaf"]["decision"]



def predict_with_model(model, X_test):
    return [_predict_one_safe(row, model.tree["node"]) for _, row in X_test.iterrows()]



def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


# Accept both of these parameter-grid styles:
# 1) Lab-style:
#    {"InfoGain": [0.01, 0.02], "Ratio": [0.1, 0.2]}
#    -> thresholds attached directly to each metric
# 2) Legacy/current-script style:
#    {"InfoGain": ["InfoGain"], "Ratio": ["Ratio"], "Threshold": [0.1, 0.2]}
#    -> threshold list shared across the listed metric names
# It also tolerates mixed files like the uploaded example_params.json.

def normalize_param_grid(params):
    candidates = []
    fallback_thresholds = params.get("Threshold", [])

    metric_specs = [
        ("InfoGain", params.get("InfoGain", [])),
        ("Ratio", params.get("Ratio", [])),
    ]

    for metric_name, values in metric_specs:
        numeric_values = [float(v) for v in values if _is_number(v)]
        string_values = [str(v) for v in values if isinstance(v, str)]

        # Lab-style: threshold grid is stored directly under the metric name.
        for threshold in numeric_values:
            candidates.append((metric_name, threshold))

        # Legacy-style: metric name(s) are listed here, thresholds live in Threshold.
        if string_values and fallback_thresholds:
            for metric_value in string_values:
                normalized_metric = str(metric_value)
                if normalized_metric.lower() in ["infogain", "informationgain", "gain"]:
                    normalized_metric = "InfoGain"
                elif normalized_metric.lower() in ["ratio", "gainratio", "infogainratio", "informationgainratio"]:
                    normalized_metric = "Ratio"

                for threshold in fallback_thresholds:
                    if _is_number(threshold):
                        candidates.append((normalized_metric, float(threshold)))

    # Defensive fallback: if the file only gives Threshold, test both metrics.
    if not candidates and fallback_thresholds:
        for threshold in fallback_thresholds:
            if _is_number(threshold):
                candidates.append(("InfoGain", float(threshold)))
                candidates.append(("Ratio", float(threshold)))

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for metric_name, threshold in candidates:
        key = (metric_name, float(threshold))
        if key not in seen:
            seen.add(key)
            deduped.append({"metric": metric_name, "threshold": float(threshold)})

    if not deduped:
        raise ValueError(
            "No valid hyperparameter combinations found. Expected something like "
            "{'InfoGain':[0.01, 0.02], 'Ratio':[0.1, 0.2]} or the legacy format "
            "with a shared Threshold list."
        )

    return deduped



def CV_10fold(data: pd.DataFrame):
    fold_array = []
    fold_y = []
    keys = data.drop(columns=["y"]).columns

    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    split_indices = np.array_split(data.index.to_numpy(), 10)

    for i in range(10):
        split = data.iloc[ (i * split_index) : ((i + 1) * split_index) ]
        fold_array.append( pd.DataFrame(split.drop(columns=['y']), columns=keys) )
        fold_y.append( pd.DataFrame(split['y'], columns=['y']) )

    return {"data": fold_array, "ground": fold_y}



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



def test_tree(x, y, x_test, metric, threshold):
    model = c45(metric=metric, threshold=threshold)

    model.fit(X=x, y=pd.Series(y['y']), dataset=DATASET)

    y_pred = model.predict(x_test)
    return y_pred



def perform_GS(data, params, labels):
    results = pd.DataFrame(columns=['Splitting Metric', 'Threshold', 'CV Accuracy', 'CM'])
    x = list(data['data'])   # list of dataframes
    y = list(data['ground'])

    IG = params['InfoGain']
    R = params['Ratio']
    Th = params['Threshold']

    for threshhold in Th:
        for g in IG:
            accuracy_sum_g = 0
            overall_cm_g = pd.DataFrame(0, index=labels, columns=labels)
            for i in range(10):
                x_train = pd.concat(x[-i:] + x[-(10 - (i + 1)):], axis=0)
                y_train = pd.concat(y[-i:] + y[-(10 - (i + 1)):], axis=0)
                x_test = x[i]
                y_test = y[i]

                predictions = test_tree(x_train, y_train, x_test, metric=g, threshold=threshhold)

                confusion_matrix = confusion_matrix_multiclass(y_test, predictions, labels)

                overall_TP = confusion_matrix.diag().sum()
                num_predictions = len(predictions)

                accuracy_sum_g += overall_TP / num_predictions
                overall_cm_g = overall_cm_g + confusion_matrix

            avg_accuracy = accuracy_sum_g / 10
            result = {
                'Splitting Metric': g,
                'Threshold': threshhold,
                'CV Accuracy': avg_accuracy,
                'CM': overall_cm_g
            }
            results = results.append(result)
        for r in R:
            accuracy_sum_r = 0
            overall_cm_r = pd.DataFrame(0, index=labels, columns=labels)
            for i in range(10):
                x_train = pd.concat(x[-i:] + x[-(10 - (i + 1)):], axis=0)
                y_train = pd.concat(y[-i:] + y[-(10 - (i + 1)):], axis=0)
                x_test = x[i]
                y_test = y[i]

                predictions = test_tree(x_train, y_train, x_test, metric=r, threshold=threshhold)

                confusion_matrix = confusion_matrix_multiclass(y_test, predictions, labels)

                overall_TP = confusion_matrix.diag().sum()
                num_predictions = len(predictions)

                accuracy_sum_r += overall_TP / num_predictions
                overall_cm_r = overall_cm_r + confusion_matrix

            avg_accuracy = accuracy_sum_r / 10
            result = {
                'Splitting Metric': r,
                'Threshold': threshhold,
                'CV Accuracy': avg_accuracy,
                'CM': overall_cm_r
            }
            results = results.append(result)
    return results.sort_values(by=['CV Accuracy'], ascending=False)



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
    DATASET = "iris"
elif "nursery" in test_set_path:
    ground_truth = test_set['class']
    test_set = test_set.drop('class', axis=1)
    DATASET = "nursery"
elif "letter" in test_set_path:
    ground_truth = test_set['lettr']
    test_set = test_set.drop('lettr', axis=1)
    DATASET = "letter"
else:
    ground_truth = test_set['Inflated']
    test_set = test_set.drop('Inflated', axis=1)
    DATASET = "balloon"

test_set['y'] = ground_truth
labels = test_set['y'].unique()

folded = CV_10fold(test_set) # folded -> dict["data", "ground"]
# pprint(params_json['InfoGain'])
best_params = perform_GS(folded, params_json, labels)
pprint(best_params)
