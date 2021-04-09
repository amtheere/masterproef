from accuracy_test import accuracy_test_cv
import pandas as pd
import numpy as np
import os as os
from texttable import Texttable
from latextable import draw_latex

dataset_location = "datasets/datasets2/"
datasets = os.listdir(dataset_location)
datasets.sort()
benchmark_results = []
table = Texttable()
table.add_row(["Dataset", "Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"])
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
    result = accuracy_test_cv(DATA, TARGET, n_splits=5, stratified=True, balanced=True, dataset_name=dataset_name)
    table.add_row(result)
    benchmark_results.append(result[1:])
'''
np.savetxt("benchmark_results.csv", benchmark_results, delimiter=", ")
'''
mean_accuracy = list(np.mean(benchmark_results, axis=0))
mean_accuracy.insert(0, "Average")
median_accuracy = list(np.median(benchmark_results, axis=0))
median_accuracy.insert(0, "Median")
table.add_row(mean_accuracy)
table.add_row(median_accuracy)
print(table.draw() + "\n")
print(draw_latex(table) + "\n")
