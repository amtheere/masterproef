import matplotlib.pyplot as plt
import os as os
import pandas as pd
import numpy as np
from aggregation_elements import outlier_score, exp_trans

dataset_location = "datasets/datasets2/"
datasets = os.listdir(dataset_location)
datasets.sort()
file = dataset_location + datasets[1]
dataset_name = file[file.rfind("/") + 1:][:-4]
print(file[file.rfind("/") + 1:][:-4])
dataframe = pd.read_csv(file, skiprows=0, sep='|')
dataframe = dataframe.replace("?", np.nan).dropna()
classes = set(dataframe[dataframe.columns[0]])
classToInt = dict(zip(classes, range(len(classes))))
dataframe['class'] = dataframe['class'].apply(lambda x: classToInt[x])
dataset = dataframe.to_numpy()
TARGET = dataset[:, 0].astype(int)
DATA = dataset[:, 1:].astype(float)
scores, labels = outlier_score(DATA, TARGET)
plt.hist(scores, bins=50, histtype="stepfilled")
plt.title(dataset_name)
plt.show()
plt.hist(exp_trans(scores, 0.9), bins=50, histtype="stepfilled")
plt.title(dataset_name+" transformed")
plt.show()
