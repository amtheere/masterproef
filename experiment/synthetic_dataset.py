import numpy as np
import matplotlib.pyplot as plt
from accuracy_test import accuracy_test
from texttable import Texttable
from latextable import draw_latex
from pyod.models.lof import LOF
from scipy.stats import skewnorm


mean_class_1 = [-2, 0]
mean_class_2 = [2, 0]
var_class_1 = [0.6, 0.2]
var_class_2 = [0.2, 0.1]
number_outliers_1 = 30
number_outliers_2 = 30
alpha_1 = 2
omega_1 = 1.5
alpha_2 = -2
omega_2 = 0.7
ratio_outlier_1 = number_outliers_1/(100+number_outliers_1)
ratio_outlier_2 = number_outliers_2/(100+number_outliers_2)
results = []
times_the_best = list(np.zeros(9, dtype=int))

rv = skewnorm(5, scale=2, loc=-2)
x = np.linspace(-4, 10, 1000)
plt.plot(x, rv.pdf(x))
plt.show()

for i in range(50):
    np.random.seed(i)
    class_1 = np.random.normal(mean_class_1, var_class_1, (100, 2))
    normal_class_1 = np.random.normal(mean_class_1[1], var_class_1[1], number_outliers_1)
    skew_class_1 = skewnorm.rvs(alpha_1, size=number_outliers_1, scale=omega_1, loc=mean_class_1[0])
    class_1_outliers = np.array([np.array([x, y]) for x, y in zip(skew_class_1, normal_class_1)])

    class_2 = np.random.normal(mean_class_2, var_class_2, (100, 2))
    normal_class_2 = np.random.normal(mean_class_2[1], var_class_2[1], number_outliers_2)
    skew_class_2 = skewnorm.rvs(alpha_2, size=number_outliers_2, scale=omega_2, loc=mean_class_2[0])
    class_2_outliers = np.array([np.array([x, y]) for x, y in zip(skew_class_2, normal_class_2)])

    cl1 = plt.scatter(class_1[:, 0], class_1[:, 1], c="blue")
    cl1_o = plt.scatter(class_1_outliers[:, 0], class_1_outliers[:, 1], c="purple", marker="<")
    cl2 = plt.scatter(class_2[:, 0], class_2[:, 1], c="red")
    cl2_o = plt.scatter(class_2_outliers[:, 0], class_2_outliers[:, 1], c="orange", marker=">")
    plt.legend((cl1, cl1_o, cl2, cl2_o),
               ("class 1", "class 1 outliers", "class 2", "class 2 outliers"),
               scatterpoints=1,
               loc='lower left',
               ncol=2,
               fontsize=8)
    plt.axis([-4, 4, -1, 1])
    plt.show()
    x_train = np.concatenate((class_1, class_1_outliers, class_2, class_2_outliers))
    y_train = np.append(np.zeros(100 + number_outliers_1), np.ones(100 + number_outliers_2)).astype(int)

    test_samples_per_class = 500
    samples_class_1_outliers = round(test_samples_per_class*ratio_outlier_1)
    samples_class_1 = test_samples_per_class-samples_class_1_outliers
    samples_class_2_outliers = round(test_samples_per_class * ratio_outlier_2)
    samples_class_2 = test_samples_per_class - samples_class_2_outliers
    class_1 = np.random.normal(mean_class_1, var_class_1, (samples_class_1, 2))
    normal_class_1 = np.random.normal(mean_class_1[1], var_class_1[1], samples_class_1_outliers)
    skew_class_1 = skewnorm.rvs(alpha_1, size=samples_class_1_outliers, scale=omega_1, loc=mean_class_1[0])
    class_1_outliers = np.array([np.array([x, y]) for x, y in zip(skew_class_1, normal_class_1)])
    class_2 = np.random.normal(mean_class_2, var_class_2, (samples_class_2, 2))
    normal_class_2 = np.random.normal(mean_class_2[1], var_class_2[1], samples_class_2_outliers)
    skew_class_2 = skewnorm.rvs(alpha_2, size=samples_class_2_outliers, scale=omega_2, loc=mean_class_2[0])
    class_2_outliers = np.array([np.array([x, y]) for x, y in zip(skew_class_2, normal_class_2)])
    cl1 = plt.scatter(class_1[:, 0], class_1[:, 1], c="blue")
    cl1_o = plt.scatter(class_1_outliers[:, 0], class_1_outliers[:, 1], c="purple", marker="<")
    cl2 = plt.scatter(class_2[:, 0], class_2[:, 1], c="red")
    cl2_o = plt.scatter(class_2_outliers[:, 0], class_2_outliers[:, 1], c="orange", marker=">")
    plt.legend((cl1, cl1_o, cl2, cl2_o),
               ("class 1", "class 1 outliers", "class 2", "class 2 outliers"),
               scatterpoints=1,
               loc='lower left',
               ncol=2,
               fontsize=8)
    plt.axis([-4, 4, -1, 1])
    plt.show()
    x_test = np.concatenate((class_1, class_1_outliers, class_2, class_2_outliers))
    y_test = np.append(np.zeros(test_samples_per_class), np.ones(test_samples_per_class)).astype(int)
    acc = accuracy_test(x_train, y_train, x_test, y_test, outlierScoreAlgorithm=LOF(contamination=0.15))
    maximum = np.amax(acc)
    for j in range(9):
        if acc[j] == maximum:
            times_the_best[j] += 1
    table = Texttable()
    table.add_row(["Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"])
    table.add_row(acc)
    print(table.draw() + "\n")
    results.append(acc)

mean_accuracy = list(np.mean(results, axis=0))
mean_accuracy.insert(0, "Average")
median_accuracy = list(np.median(results, axis=0))
median_accuracy.insert(0, "Median")
times_the_best.insert(0, "times the best")
table = Texttable()
table.add_row(["Summary", "Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"])
table.add_row(mean_accuracy)
table.add_row(median_accuracy)
table.add_row(times_the_best)
print(table.draw() + "\n")
print(draw_latex(table) + "\n")
