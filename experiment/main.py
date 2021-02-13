from accuracy_test import accuracy_test
import pandas as pd
import numpy as np

file = "datasets/datasets2/wpbc.psv"
dataset_name = file[file.rfind("/")+1:][:-4]
dataframe = pd.read_csv(file, skiprows=0, sep='|')
dataframe = dataframe.replace("?", np.nan).dropna()
classes = set(dataframe[dataframe.columns[0]])
classToInt = dict(zip(classes, range(len(classes))))
dataframe['class'] = dataframe['class'].apply(lambda x: classToInt[x])
dataset = dataframe.to_numpy()
TARGET = dataset[:, 0].astype(int)
DATA = dataset[:, 1:].astype(float)

accuracy = accuracy_test(DATA, TARGET, n_splits=10, stratified=True, balanced=True, dataset_name=dataset_name)
