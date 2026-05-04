import json
import math
import numpy as np
import pandas as pd


class c45:
    def __init__(self, metric="InfoGain", threshold=0.1):
        self.metric = metric
        self.threshold = threshold
        self.tree = None
        self.feature_types = {}
        self.class_name = None
        self.dataset = None

    def fit(self, X, y, feature_types=None, dataset=None):
        self.dataset = dataset
        self.class_name = y.name if y.name else "class"
        self.feature_types = feature_types or {col: "categorical" for col in X.columns}

        self.tree = {
            "dataset": dataset,
            "node": self._build_tree(X, y, list(X.columns))
        }
        return self.tree

    def predict(self, X_test):
        return [self._predict_one(row, self.tree["node"]) for _, row in X_test.iterrows()]

    def save_tree(self, filename):
        with open(filename, "w") as f:
            json.dump(self.tree, f, indent=2)

    def read_tree(self, filename):
        with open(filename, "r") as f:
            self.tree = json.load(f)

    def _build_tree(self, X, y, attributes):
        if len(y) < 5:
            return self._leaf(y)
        
        if len(y.unique()) == 1:
            return self._leaf(y)

        if len(attributes) == 0:
            return self._leaf(y)

        best_attr, best_split, best_score = self._select_best_attribute(X, y, attributes)

        if best_attr is None or best_score < self.threshold:
            return self._leaf(y)

        node = {
            "var": best_attr,
            "edges": []
        }

        if self.feature_types[best_attr] == "numeric":
            threshold = best_split

            left_mask = X[best_attr] <= threshold
            right_mask = X[best_attr] > threshold

            for op, mask in [("<=", left_mask), (">", right_mask)]:
                if mask.sum() == 0:
                    child = {"leaf": self._leaf(y)["leaf"]}
                else:
                    new_attrs = [a for a in attributes if a != best_attr]

                    child_node = self._build_tree(
                        X.loc[mask],
                        y.loc[mask],
                        new_attrs
                    )
                    child = child_node

                edge = {
                    "edge": {
                        "value": threshold,
                        "op": op
                    }
                }

                if "leaf" in child:
                    edge["edge"]["leaf"] = child["leaf"]
                else:
                    edge["edge"]["node"] = child

                node["edges"].append(edge)

        else:
            remaining_attrs = [a for a in attributes if a != best_attr]

            for val in sorted(X[best_attr].dropna().unique()):
                mask = X[best_attr] == val

                if mask.sum() == 0:
                    child = {"leaf": self._leaf(y)["leaf"]}
                else:
                    child_node = self._build_tree(
                        X.loc[mask],
                        y.loc[mask],
                        remaining_attrs
                    )
                    child = child_node

                edge = {
                    "edge": {
                        "value": val
                    }
                }

                if "leaf" in child:
                    edge["edge"]["leaf"] = child["leaf"]
                else:
                    edge["edge"]["node"] = child

                node["edges"].append(edge)

        return node

    def _select_best_attribute(self, X, y, attributes):
        best_attr = None
        best_split = None
        best_score = -1

        for attr in attributes:
            if self.feature_types[attr] == "numeric":
                score, split = self._best_numeric_split(X[attr], y)
            else:
                score = self._information_gain(X[attr], y)
                split = None

            if self.metric.lower() in ["ratio", "infogainratio", "gainratio"]:
                split_info = self._split_info(X[attr], split)
                if split_info != 0:
                    score = score / split_info
                else:
                    score = 0

            if score > best_score:
                best_score = score
                best_attr = attr
                best_split = split

        return best_attr, best_split, best_score

    def _best_numeric_split(self, x, y):
        values = sorted(x.dropna().unique())

        if len(values) <= 1:
            return 0, None

        candidates = [
            (values[i] + values[i + 1]) / 2
            for i in range(len(values) - 1)
        ]

        best_gain = -1
        best_threshold = None

        for threshold in candidates:
            split_series = pd.Series(
                np.where(x <= threshold, "<=", ">"),
                index=x.index
            )
            gain = self._information_gain(split_series, y)

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_gain, best_threshold

    def _information_gain(self, x, y):
        base_entropy = self._entropy(y)
        weighted_entropy = 0
        n = len(y)

        for val in x.dropna().unique():
            mask = x == val
            weighted_entropy += (mask.sum() / n) * self._entropy(y.loc[mask])

        return base_entropy - weighted_entropy

    def _split_info(self, x, threshold=None):
        if threshold is not None:
            groups = pd.Series(np.where(x <= threshold, "<=", ">"), index=x.index)
        else:
            groups = x

        n = len(groups)
        result = 0

        for count in groups.value_counts():
            p = count / n
            result -= p * math.log2(p)

        return result

    def _entropy(self, y):
        result = 0
        n = len(y)

        for count in y.value_counts():
            p = count / n
            result -= p * math.log2(p)

        return result

    def _leaf(self, y):
        counts = y.value_counts()
        decision = counts.idxmax()
        probability = counts.max() / counts.sum()

        return {
            "leaf": {
                "decision": decision,
                "p": round(float(probability), 4)
            }
        }

    def _predict_one(self, row, node):
        while "leaf" not in node:
            attr = node["var"]
            edges = node["edges"]
            matched = False

            for edge_obj in edges:
                edge = edge_obj["edge"]

                if "op" in edge:
                    if edge["op"] == "<=" and row[attr] <= edge["value"]:
                        node = edge.get("node", {"leaf": edge["leaf"]})
                        matched = True
                        break
                    elif edge["op"] == ">" and row[attr] > edge["value"]:
                        node = edge.get("node", {"leaf": edge["leaf"]})
                        matched = True
                        break
                else:
                    if row[attr] == edge["value"]:
                        node = edge.get("node", {"leaf": edge["leaf"]})
                        matched = True
                        break

            if not matched:
                return None

        return node["leaf"]["decision"]


def load_lab_csv(filename):
    with open(filename, "r") as f:
        names = f.readline().strip().split(",")
        domains = [int(x) for x in f.readline().strip().split(",")]
        class_name = f.readline().strip()

    df = pd.read_csv(filename, skiprows=[1, 2])

    row_id_cols = [
        names[i]
        for i, domain in enumerate(domains)
        if domain == -1
    ]

    df = df.drop(columns=row_id_cols, errors="ignore")
    df = df.dropna()

    y = df[class_name]
    X = df.drop(columns=[class_name])

    feature_types = {}

    for col in X.columns:
        idx = names.index(col)
        if domains[idx] == 0:
            feature_types[col] = "numeric"
            X[col] = pd.to_numeric(X[col])
        else:
            feature_types[col] = "categorical"

    return X, y, feature_types