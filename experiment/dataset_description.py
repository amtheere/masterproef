import numpy as np
import pandas as pd
import os as os
from texttable import Texttable
from latextable import draw_latex

dataset_location = "datasets/datasets2/"
datasets = os.listdir(dataset_location)
datasets.sort()
table = Texttable()
table.add_row(["Name", "#Features", "#Instances", "IR", "Name", "#Features", "#Instances", "IR"])
n = round(len(datasets)/2)
for i in range(n):
    row = []
    file1 = dataset_location + datasets[i]
    dataset_name1 = file1[file1.rfind("/") + 1:][:-4]
    row.append(dataset_name1)
    dataframe = pd.read_csv(file1, skiprows=0, sep='|')
    dataframe = dataframe.replace("?", np.nan).dropna()
    classes = set(dataframe[dataframe.columns[0]])
    classToInt = dict(zip(classes, range(len(classes))))
    dataframe['class'] = dataframe['class'].apply(lambda x: classToInt[x])
    dataset = dataframe.to_numpy()
    TARGET = dataset[:, 0].astype(int)
    DATA = dataset[:, 1:].astype(float)
    row.append(len(DATA[0]))
    row.append(len(DATA))
    class0 = [x for x in TARGET if x == 0]
    class1 = [x for x in TARGET if x == 1]
    IR = (max(len(class0), len(class1))/min(len(class0), len(class1)))
    row.append(IR)

    file2 = dataset_location + datasets[i + n]
    dataset_name2 = file2[file2.rfind("/") + 1:][:-4]
    row.append(dataset_name2)
    dataframe2 = pd.read_csv(file2, skiprows=0, sep='|')
    dataframe2 = dataframe2.replace("?", np.nan).dropna()
    classes2 = set(dataframe2[dataframe2.columns[0]])
    classToInt2 = dict(zip(classes2, range(len(classes2))))
    dataframe2['class'] = dataframe2['class'].apply(lambda x: classToInt2[x])
    dataset2 = dataframe2.to_numpy()
    TARGET2 = dataset2[:, 0].astype(int)
    DATA2 = dataset2[:, 1:].astype(float)
    row.append(len(DATA2[0]))
    row.append(len(DATA2))
    class0 = [x for x in TARGET2 if x == 0]
    class1 = [x for x in TARGET2 if x == 1]
    IR = (max(len(class0), len(class1)) / min(len(class0), len(class1)))
    row.append(IR)
    table.add_row(row)

print(table.draw() + "\n")
print(draw_latex(table) + "\n")
