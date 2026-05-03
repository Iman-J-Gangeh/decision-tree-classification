import sys
import pandas as pd
import json
from pprint import pprint

from pandas.core.interchange.dataframe_protocol import DataFrame


# func predeclarations

def read_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)


def read_csv(filename):
    return pd.read_csv(filename)


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


def predict(tree: dict, row):
    return traverse_node(tree['node'], row)


def traverse_node(node, row):
    var_name = node["var"]
    row_value = row[var_name]
    edges = node["edges"]

    # Numerical
    if any("op" in edge_wrapper["edge"] for edge_wrapper in edges):
        for edge_wrapper in edges:
            edge = edge_wrapper["edge"]
            threshold = edge["value"]
            op = edge["op"]

            if op == "<=" and row_value <= threshold:
                return follow_edge(edge, row)
            elif op == ">" and row_value > threshold:
                return follow_edge(edge, row)

    # Categorical
    else:
        for edge_wrapper in edges:
            edge = edge_wrapper["edge"]
            if row_value == edge["value"]:
                return follow_edge(edge, row)


def follow_edge(edge, row):
    if "leaf" in edge:
        return edge["leaf"]["decision"]

    if "node" in edge:
        return traverse_node(edge["node"], row)

    raise ValueError("Edge has neither 'node' nor 'leaf'")


def parse_JSON_for_pred(data: DataFrame, y_ground, json, eval=None):
    columns = data.keys()
    data_dict = data.to_dict(orient="records") #: list[dict]
    predicted = []

    for row in data_dict:
        y_pred = predict(json, row)
        # row["y_pred"] = y_pred

        predicted.append(y_pred)

    # predicted = pd.concat(data, pd.DataFrame(predicted, columns = ['y_pred']))
    data['y_pred'] = pd.DataFrame(predicted, columns=['y_pred'])

    if not eval:
        return predicted
    else:
        cm = confusion_matrix_multiclass(y_ground, data['y_pred'])
        num_TP = 0

        for level in y_ground.unique():
            num_TP += class_counts_from_cm(cm, level)['TP']

        return {
            "predictions": predicted,
            "num_records": len(predicted),
            "num_TP": num_TP,
            "num_TN": len(predicted) - num_TP,
            "accuracy": num_TP / len(predicted),
            "error_rate": 1 - (num_TP / len(predicted)),
            "CM": cm
        }



# Main
test_set_path = sys.argv[1]
tree_path = sys.argv[2]
print("Reading JSON from " + tree_path)
if (len(sys.argv) > 3):
    eval = sys.argv[3]
else:
    eval = None

with open(tree_path, 'r') as tree_file:
    json_data = json.load(tree_file)
test_set = pd.read_csv(test_set_path, skiprows=[1,2])

if "iris" in test_set_path:
    ground_truth = test_set['species']
elif "nursery" in test_set_path:
    ground_truth = test_set['class']
else:
    ground_truth = test_set['lettr']

parsed_data = parse_JSON_for_pred(test_set, ground_truth,  json_data, eval)

if (len(sys.argv) >= 3):
    pprint(parsed_data)