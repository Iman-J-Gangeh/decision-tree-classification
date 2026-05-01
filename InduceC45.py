import numpy as np
import pandas as pd
import math


class TreeNode: 
    def __init__(self, value, children=[]):
        self.value = value # holds the Attribute being considered
        self.children = children # children are (edge, Node)
                                 # label is the particular category of attribute

    def add_child(self, child):
        self.children.append(child)

def InduceC45(data, save=None): 
    x, y = clean(data)
    A = x.columns
    threshold = 0.1 
    fit(x ,y, A, threshold)
    return 0

# Induces a decision tree based of training data
def fit(x, y, A, threshold): 
    if len(A) == 0:
        return TreeNode(plurality(y))
    if np.unique(y).size == 1:
        return TreeNode(np.unique(y)[0])
    winner = selectSplittingAttribute(x, y, A, threshold)
    if winner is None: 
        return TreeNode(plurality(y))
    else: 
        tree = TreeNode(winner)
        for a in x[winner].unique():
            xj = x[x[winner] == a]
            yj = y[xj.index]
            c = fit(xj, yj, A, threshold)
            tree.add_child((a, c))
        return tree


# Select the best attribute to split on and return the attribute
def selectSplittingAttribute(x, y, A, threshold):
    metric = {}
    for a in A: 
        metric[a] = computeMetric(x[a], y)
    winner = np.argmax(metric)
    if metric[winner] >= threshold: 
        return metric[winner];
    return None
    
# comnpute the information gain 
def computeMetric(x, y):
    # calculate the overall entropy 
    values = y.value_counts()
    n = values.sum()
    entropy = 0
    for val in values:
        p = val / n
        entropy += -(p) * math.log2(p)
    
    # calculate the individual information gains
    entropy_split = 0
    xcounts = x.value_counts()
    # iterate over unique values
    for i in xcounts.index: 
        # isolate each level in x with associated y
        x_vals = x[x == i]
        y_vals = y[x_vals.index]
        w = x_vals.size / n
        sub_entropy = 0
        y_sub_values = y_vals.value_counts()
        for j in y_sub_values.index:
            p =  y_sub_values[j] / x_vals.size
            sub_entropy +=  -(p) * math.log2(p)
        entropy_split += w * sub_entropy
    return entropy - entropy_split



    


# takes a series and return the plurality label
def plurality(y):
    counts = y.value_counts()
    return counts.index[np.argmax(counts)]



# takes the name of a csv and returns x and y values 
def clean(fname): 
    df = pd.read_csv(fname)
    df.dropna(inplace=True)
    n = df.iloc[0,-1] # n is number of categories
    y = df[df.columns[-1]]
    x = df.drop(df.columns[-1], axis=1) 
    x = df.drop(index=0) # for categorical data
    return x, y
   