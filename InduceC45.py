
def InduceC45(data, save=None): 
    x, y = clean(data)
    return 0

# takes the name of a csv and returns x and y values 
def clean(fname): 
    df = pd.read_csv(fname)
    df = df.dropna()
    n = df.iloc[0,-1] # n is number of categories
    y = df[df.columns[-1]]
    x = df.drop(df.columns[-1], axis=1) 
    return x, y