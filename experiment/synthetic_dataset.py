import numpy as np
import matplotlib.pyplot as plt
from accuracy_test import accuracy_test_cv, accuracy_test
from texttable import Texttable
from pyod.models.lof import LOF


DIM = 2

np.random.seed(1)

class_1 = np.random.normal(np.zeros(DIM), [1, 0.2], (100, 2))
class_1_outliers = np.random.normal(np.zeros(DIM), [2.5, 0.2], (30, 2))
class_2 = np.random.normal([4, 0], [0.3, 0.2], (130, 2))
plt.scatter(class_1[:, 0], class_1[:, 1])
plt.scatter(class_1_outliers[:, 0], class_1_outliers[:, 1])
plt.scatter(class_2[:, 0], class_2[:, 1])
plt.axis([-5, 5, -1, 1])
plt.show()
x_train = np.concatenate((class_1, class_1_outliers, class_2))
y_train = np.append(np.zeros(130), np.ones(130)).astype(int)
acc = accuracy_test_cv(x_train, y_train, "synthetic", b=0.75, outlierScoreAlgorithm=LOF(contamination=0.15))
table = Texttable()
table.add_row(["Dataset", "Min", "Avg", "OWA", "Mino", "Avgo", "OWAo", "FR", "TS", "WOWA"])
table.add_row(acc)
print(table.draw() + "\n")


number_of_test_samples = 1000
class_1 = np.random.normal(np.zeros(DIM), [1, 0.2], (number_of_test_samples, 2))
class_2 = np.random.normal([4, 0], [0.3, 0.2], (number_of_test_samples, 2))
x_test = np.concatenate((class_1, class_2))
y_test = np.append(np.zeros(number_of_test_samples), np.ones(number_of_test_samples)).astype(int)
acc = accuracy_test(x_train, y_train, x_test, y_test, outlierScoreAlgorithm=LOF(contamination=0.15), b=0.50001)
table = Texttable()
table.add_row(["Min", "Avg", "OWA", "Mino", "Avgo", "OWAo", "FR", "TS", "WOWA"])
table.add_row(acc)
print(table.draw() + "\n")