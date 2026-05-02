import sys
import pandas as pd
import json

# func predeclarations

def read_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)

def read_csv(filename):
    return pd.read_csv(filename)

def find_prediction(node, x_bar):
     if isinstance(node, dict):
         for key, value in node.items():
             print(f"Node: {key}")
             traverse_tree(value,)
     elif isinstance(node, list):
         for i, item in enumerate(node):
             print(f"{indent}List Item {i}:")
             traverse_tree(item, depth + 1)
     else:
         print(f"{indent}Leaf Value: {node}")


def predict():


def parse_JSON_for_pred(data, series):
    return 0
# read in arguments, assign
if (len(sys.argv) == 3):
    test_set_path = sys.argv[1]
    tree_data = sys.argv[2]
else:
    out_file = sys.argv[3]

json_data = read_json(tree_data)
test_set = pd.read_csv(test_set_path)

predicted = []

for row in test_set.to_dict('records'):
    y_pred = predict(row, json_data)
    row["y_pred"] = y_pred
    predicted.append(row)

# print to file? output?
