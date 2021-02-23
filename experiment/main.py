from accuracy_test import accuracy_test
import pandas as pd
import numpy as np
import os as os
from texttable import Texttable
from latextable import draw_latex
from pyod.models.iforest import IForest

dataset_location = "datasets/datasets2/"
datasets = os.listdir(dataset_location)
datasets.sort()
mean_accuracy = np.zeros(9)
table = Texttable()
table.add_row(["Dataset", "Min", "Avg", "OWA", "Mino", "Avgo", "OWAo", "FR", "TS", "WOWA"])
datasets = datasets
for file in datasets:
    file = dataset_location + file
    dataset_name = file[file.rfind("/") + 1:][:-4]
    dataframe = pd.read_csv(file, skiprows=0, sep='|')
    dataframe = dataframe.replace("?", np.nan).dropna()
    classes = set(dataframe[dataframe.columns[0]])
    classToInt = dict(zip(classes, range(len(classes))))
    dataframe['class'] = dataframe['class'].apply(lambda x: classToInt[x])
    dataset = dataframe.to_numpy()
    TARGET = dataset[:, 0].astype(int)
    DATA = dataset[:, 1:].astype(float)
    result = accuracy_test(DATA, TARGET, n_splits=10, stratified=True, balanced=True, dataset_name=dataset_name,
                           outlierScoreAlgorithm=IForest())
    table.add_row(result)
    mean_accuracy += result[1:]
mean_accuracy = list(mean_accuracy / len(datasets))
mean_accuracy.insert(0, "Average")
table.add_row(mean_accuracy)
print(table.draw() + "\n")
print(draw_latex(table) + "\n")
