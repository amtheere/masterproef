import numpy as np
import matplotlib.pyplot as plt
from accuracy_test import accuracy_test
from texttable import Texttable
from latextable import draw_latex
from pyod.models.lof import LOF
from scipy.stats import skewnorm

mean_class_1 = [-2, 0]
mean_class_2 = [2, 0]
sigma_1 = 0.6
sigma_2 = 0.2
var_class_1 = [sigma_1, 0.2]
var_class_2 = [sigma_2, 0.1]
test_samples_per_class = 500
number_outliers_1 = 30
number_outliers_2 = 30
ratio_outlier_1 = number_outliers_1 / (100 + number_outliers_1)
ratio_outlier_2 = number_outliers_2 / (100 + number_outliers_2)
samples_class_1_outliers = round(test_samples_per_class * ratio_outlier_1)
samples_class_1 = test_samples_per_class - samples_class_1_outliers
samples_class_2_outliers = round(test_samples_per_class * ratio_outlier_2)
samples_class_2 = test_samples_per_class - samples_class_2_outliers
samples_normal = samples_class_1 + samples_class_2
samples_outliers = samples_class_1_outliers + samples_class_2_outliers
alpha_1 = 2
omega_1 = 1.5
alpha_2 = -2
omega_2 = 0.7
table = Texttable()
table.add_row(["Summary", "Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA", "Value"])
median_accuracy = []
median_accuracy_outliers = []

start = 0.1
steps = 12
step_size = 0.1
step = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.4, 3.8, 4]
parameter_name = "alpha_1"
if __name__ == "__main__":
    for k in range(len(step)):
        alpha_1 = step[k]
        results_normal = []
        results_outlier = []
        results = []
        times_the_best = list(np.zeros(9, dtype=int))
        times_the_best_outliers = list(np.zeros(9, dtype=int))
        for i in range(40):
            np.random.seed(i+40*k)
            class_1_train = np.random.normal(mean_class_1, var_class_1, (100, 2))
            normal_class_1 = np.random.normal(mean_class_1[1], var_class_1[1], number_outliers_1)
            skew_class_1 = skewnorm.rvs(alpha_1, size=number_outliers_1, scale=omega_1, loc=mean_class_1[0])
            class_1_outliers_train = np.array([np.array([x, y]) for x, y in zip(skew_class_1, normal_class_1)])

            class_2_train = np.random.normal(mean_class_2, var_class_2, (100, 2))
            normal_class_2 = np.random.normal(mean_class_2[1], var_class_2[1], number_outliers_2)
            skew_class_2 = skewnorm.rvs(alpha_2, size=number_outliers_2, scale=omega_2, loc=mean_class_2[0])
            class_2_outliers_train = np.array([np.array([x, y]) for x, y in zip(skew_class_2, normal_class_2)])
            x_train = np.concatenate((class_1_train, class_1_outliers_train, class_2_train, class_2_outliers_train))
            y_train = np.append(np.zeros(100 + number_outliers_1), np.ones(100 + number_outliers_2)).astype(int)

            class_1_test = np.random.normal(mean_class_1, var_class_1, (samples_class_1, 2))
            class_2_test = np.random.normal(mean_class_2, var_class_2, (samples_class_2, 2))
            x_test = np.concatenate((class_1_test, class_2_test))
            y_test = np.append(np.zeros(samples_class_1), np.ones(samples_class_2)).astype(int)
            acc_normal = accuracy_test(x_train, y_train, x_test, y_test,
                                       outlierScoreAlgorithm=LOF(contamination=ratio_outlier_1))
            results_normal.append(acc_normal)

            normal_class_1 = np.random.normal(mean_class_1[1], var_class_1[1], samples_class_1_outliers)
            skew_class_1 = skewnorm.rvs(alpha_1, size=samples_class_1_outliers, scale=omega_1, loc=mean_class_1[0])
            class_1_outliers_test = np.array([np.array([x, y]) for x, y in zip(skew_class_1, normal_class_1)])
            normal_class_2 = np.random.normal(mean_class_2[1], var_class_2[1], samples_class_2_outliers)
            skew_class_2 = skewnorm.rvs(alpha_2, size=samples_class_2_outliers, scale=omega_2, loc=mean_class_2[0])
            class_2_outliers_test = np.array([np.array([x, y]) for x, y in zip(skew_class_2, normal_class_2)])
            x_test = np.concatenate((class_1_outliers_test, class_2_outliers_test))
            y_test = np.append(np.zeros(samples_class_1_outliers), np.ones(samples_class_2_outliers)).astype(int)
            acc_outliers = accuracy_test(x_train, y_train, x_test, y_test,
                                         outlierScoreAlgorithm=LOF(contamination=ratio_outlier_1))
            results_outlier.append(acc_outliers)
            acc = [(samples_normal * x + samples_outliers * y) / (2 * test_samples_per_class)
                   for (x, y) in zip(acc_normal, acc_outliers)]
            results.append(acc)

            if i == 0:
                cl1 = plt.scatter(class_1_train[:, 0], class_1_train[:, 1], c="blue")
                cl1_o = plt.scatter(class_1_outliers_train[:, 0], class_1_outliers_train[:, 1], c="purple", marker="<")
                cl2 = plt.scatter(class_2_train[:, 0], class_2_train[:, 1], c="red")
                cl2_o = plt.scatter(class_2_outliers_train[:, 0], class_2_outliers_train[:, 1], c="orange", marker=">")
                plt.legend((cl1, cl1_o, cl2, cl2_o),
                           ("class 1", "class 1 outliers", "class 2", "class 2 outliers"),
                           scatterpoints=1,
                           loc='lower left',
                           ncol=2,
                           fontsize=8)
                plt.axis([-4, 4, -1, 1])
                plt.savefig("plots/"+parameter_name+"_plots/train/"+str(step[k])+".png")
                plt.clf()
                cl1 = plt.scatter(class_1_test[:, 0], class_1_test[:, 1], c="blue")
                cl1_o = plt.scatter(class_1_outliers_test[:, 0], class_1_outliers_test[:, 1], c="purple", marker="<")
                cl2 = plt.scatter(class_2_test[:, 0], class_2_test[:, 1], c="red")
                cl2_o = plt.scatter(class_2_outliers_test[:, 0], class_2_outliers_test[:, 1], c="orange", marker=">")
                plt.legend((cl1, cl1_o, cl2, cl2_o),
                           ("class 1", "class 1 outliers", "class 2", "class 2 outliers"),
                           scatterpoints=1,
                           loc='lower left',
                           ncol=2,
                           fontsize=8)
                plt.axis([-4, 4, -1, 1])
                plt.savefig("plots/"+parameter_name+"_plots/test/"+str(step[k])+".png")
                plt.clf()

            '''
            maximum = np.amax(acc_outliers)
            for j in range(9):
                if acc_outliers[j] == maximum:
                    times_the_best_outliers[j] += 1
            maximum = np.amax(acc)
            for j in range(9):
                if acc[j] == maximum:
                    times_the_best[j] += 1
            '''

        # mean_accuracy = list(np.mean(results, axis=0))
        # mean_accuracy.insert(0, "Average")
        median_accuracy.append(list(np.median(results, axis=0)))
        results = list(np.median(results, axis=0))
        results.insert(0, "Median")
        results.append(start+k*step_size)
        # times_the_best.insert(0, "times the best")
        # mean_accuracy_outliers = list(np.mean(results_outlier, axis=0))
        # mean_accuracy_outliers.insert(0, "Average")
        median_accuracy_outliers.append(list(np.median(results_outlier, axis=0)))
        results_outliers = list(np.median(results_outlier, axis=0))
        results_outliers.insert(0, "Median")
        results_outliers.append(start+k*step_size)
        # times_the_best_outliers.insert(0, "times the best")
        # table.add_row(mean_accuracy)
        table.add_row(results)
        # table.add_row(times_the_best)
        # table.add_row(mean_accuracy_outliers)
        table.add_row(results_outliers)
        # table.add_row(times_the_best_outliers)
    x = [start + k * step_size for k in range(steps)]
    median_accuracy = np.array(median_accuracy)
    median_accuracy_outliers = np.array(median_accuracy_outliers)
    np.savetxt("results_synth/median_"+parameter_name+".csv", median_accuracy, delimiter=", ")
    np.savetxt("results_synth/median_"+parameter_name+"_outliers.csv", median_accuracy_outliers, delimiter=", ")
    print(table.draw() + "\n")
    print(draw_latex(table) + "\n")
