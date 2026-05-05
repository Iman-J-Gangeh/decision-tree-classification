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

    for idx in split_indices:
        split = data.loc[idx].reset_index(drop=True)
        fold_array.append(pd.DataFrame(split.drop(columns=["y"]), columns=keys))
        fold_y.append(pd.DataFrame(split["y"], columns=["y"]))

    return {"data": fold_array, "ground": fold_y}



def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0]
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0]

    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = list(pd.Index(y_true).union(pd.Index(y_pred)).unique())
    else:
        labels = list(labels)
        extra_labels = [label for label in pd.Index(y_true).union(pd.Index(y_pred)).unique() if label not in labels]
        labels = labels + extra_labels

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



def test_tree(x, y, x_test, metric, threshold, feature_types):
    model = c45(metric=metric, threshold=threshold)
    model.fit(X=x, y=pd.Series(y["y"]), feature_types=feature_types, df=DATASET)

    majority_class = pd.Series(y["y"]).mode().iloc[0]
    y_pred = predict_with_model(model, x_test)
    y_pred = [pred if pred is not None else majority_class for pred in y_pred]
    return y_pred



def perform_GS(data, params, labels, feature_types):
    results_list = []
    x = list(data["data"])
    y = list(data["ground"])
    param_grid = normalize_param_grid(params)

    for setting in param_grid:
        metric_name = setting["metric"]
        threshold = setting["threshold"]

        overall_true = []
        overall_pred = []

        for i in range(10):
            x_train = pd.concat(x[:i] + x[i + 1:], axis=0, ignore_index=True)
            y_train = pd.concat(y[:i] + y[i + 1:], axis=0, ignore_index=True)
            x_test = x[i].reset_index(drop=True)
            y_test = y[i].reset_index(drop=True)

            predictions = test_tree(
                x_train,
                y_train,
                x_test,
                metric=metric_name,
                threshold=threshold,
                feature_types=feature_types,
            )

            overall_true.extend(y_test["y"].tolist())
            overall_pred.extend(predictions)

        overall_cm = confusion_matrix_multiclass(overall_true, overall_pred, labels)
        num_predictions = len(overall_true)
        num_correct = sum(1 for actual, predicted in zip(overall_true, overall_pred) if actual == predicted)
        accuracy = num_correct / num_predictions if num_predictions > 0 else 0.0

        result = {
            "Splitting Metric": metric_name,
            "Threshold": threshold,
            "CV Accuracy": accuracy,
            "CM": overall_cm,
        }
        results_list.append(result)

    results = pd.DataFrame(results_list)
    return results.sort_values(by=["CV Accuracy", "Splitting Metric", "Threshold"], ascending=[False, True, True]).reset_index(drop=True)


# Main
test_set_path = sys.argv[1]
thresh_path = sys.argv[2]
out_file = sys.argv[3] if len(sys.argv) > 3 else None
DATASET = os.path.basename(test_set_path)

print("Reading JSON of hyperparams from " + thresh_path)
with open(thresh_path, "r") as params_file:
    params_json = json.load(params_file)

X, y, feature_types = load_lab_csv(test_set_path)

test_set = X.copy().reset_index(drop=True)
test_set["y"] = y.reset_index(drop=True)
labels = list(pd.unique(test_set["y"]))

folded = CV_10fold(test_set)
best_params = perform_GS(folded, params_json, labels, feature_types)

best_model = best_params.iloc[0]
print("Best Decision Tree Model")
print("Splitting Metric:", best_model["Splitting Metric"])
print("Threshold:", best_model["Threshold"])
print("Overall CV Accuracy:", best_model["CV Accuracy"])
print("Overall CV Confusion Matrix:")
print(best_model["CM"])

if out_file is not None:
    final_model = c45(metric=best_model["Splitting Metric"], threshold=float(best_model["Threshold"]))
    final_model.fit(X=X, y=y, feature_types=feature_types, df=DATASET)
    final_model.save_tree(out_file)
    print("Saved best-model tree to", out_file)
