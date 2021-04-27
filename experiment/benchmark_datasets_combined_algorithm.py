from accuracy_test import accuracy_test_combine_algo_cv
import pandas as pd
import numpy as np
import os as os
from texttable import Texttable
from latextable import draw_latex

dataset_location = "datasets/datasets2/"
datasets = os.listdir(dataset_location)
datasets.sort()
benchmark_result = []
table_accuracy = Texttable()
table_accuracy.add_row(["Dataset", "accuracy"])
table_algo_used = Texttable()
table_algo_used.add_row(["Dataset", "Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"])
table_algo_best = Texttable()
table_algo_best.add_row(["Dataset", "Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"])

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
    accuracy, algorithm_used, algorithm_best = accuracy_test_combine_algo_cv(DATA, TARGET, dataset_name, n_splits=5,
                                                                             stratified=True, balanced=True)
    table_accuracy.add_row(accuracy)
    print(accuracy)
    table_algo_best.add_row(algorithm_best)
    print(algorithm_used)
    table_algo_used.add_row(algorithm_used)
    benchmark_result.append(accuracy[1])

mean_accuracy = ["Average", np.average(benchmark_result)]
median_accuracy = ["Median", np.median(benchmark_result)]
table_accuracy.add_row(mean_accuracy)
table_accuracy.add_row(median_accuracy)
print(table_accuracy.draw() + "\n")
print(draw_latex(table_accuracy) + "\n")
print(table_algo_used.draw() + "\n")
print(draw_latex(table_algo_used) + "\n")
print(table_algo_best.draw() + "\n")
print(draw_latex(table_algo_best) + "\n")
