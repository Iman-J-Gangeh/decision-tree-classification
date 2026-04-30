import numpy as np
import pandas as pd


class TreeNode: 
    def __init__(self, value, lNode, rNode):
        self.value = value
        self.left = lNode
        self.right = rNode 


def InduceC45(data, save=None): 
    x, y = clean(data)
    A = x.columns
    threshold = 0.1 
    fit(x ,y, A, threshold)
    return 0

def fit(x, y, A, threshold): 
    if len(A) == 0:
        return TreeNode(plurality(y), None, None)
    if np.unique(y).size:
        return TreeNode(np.unique(y)[0], None, None)
    selectSplittingAttribute()


# Select the best attribute to split on 
def selectSplittingAttribute(x, y, A, threshold):
    metric = {}
    for a in A: 
        metric[a] = computeMetric(x[a], y)
    
# comnpute the information gain 
def computeMetric(x, y):



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
    return x, y
   