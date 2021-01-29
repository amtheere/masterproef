from accuracy_test import accuracy_test
from sklearn import datasets

'''
file = "datasets/wifi.psv"
dataframe = pd.read_csv(file, skiprows=0, sep='|')
classes = set(dataframe[dataframe.columns[0]])
classToInt = dict(zip(classes, range(len(classes))))
dataframe['class'] = dataframe['class'].apply(lambda x: classToInt[x])
dataset = dataframe.to_numpy()
TARGET = dataset[:, 0].astype(int)
DATA = dataset[:, 1:]
'''

iris = datasets.load_iris()
DATA = iris.data
TARGET = iris.target

accuracy = accuracy_test(DATA, TARGET, n_splits=5, stratified=True)
