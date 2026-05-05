import sys
import json
import pandas as pd
from c45 import c45


def main():
    csv = sys.argv[1]
    save = sys.argv[2] if len(sys.argv) == 3 else None

    X, y, feature_types = clean_csv(csv)

    model = c45(metric="InfoGain", threshold=0.05)
    tree = model.fit(X, y, feature_types=feature_types, dataset=csv)

    print(json.dumps(tree, indent=2))

    if save is not None:
        model.save_tree(save)

def clean_csv(filename):
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

if __name__ == "__main__":
    main()
