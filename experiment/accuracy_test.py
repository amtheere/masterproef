from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from sklearn import metrics

from aggregation_operators import *
from aggregation_elements import aggregation_elements, lof_score

# Library for outlier detection
from pyod.models.lof import LOF

from latextable import draw_latex
from texttable import Texttable


def accuracy_test(data, target, dataset_name, n_splits=5, stratified=False,
                  outlierScoreAlgorithm=LOF(), balanced=False):
    # Sets up the cross-validation
    seed = 7
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf = kf.split(data, target)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf = kf.split(data)

    # Stores the accuracy rates of all of the folds
    accuracy_rates_min = []
    accuracy_rates_avg = []
    accuracy_rates_owa = []
    accuracy_rates_owa_without_outliers = []
    accuracy_rates_avg_without_outliers = []
    accuracy_rates_min_without_outliers = []
    accuracy_rates_fuzzy_removal = []
    accuracy_rates_two_sym = []
    accuracy_rates_wowa = []

    def quantifier(t):
        return vague_quantifier(t, 0.2, 1)

    for train_index, test_index in kf:
        train_x = np.array([data[i] for i in train_index])
        train_y = np.array([target[i] for i in train_index])
        test_x = np.array([data[i] for i in test_index])
        test_y = np.array([target[i] for i in test_index])
        number_of_classes = len(set(train_y))

        outlier_values, outlier_labels = lof_score(train_x, train_y, outlierScoreAlgorithm)
        standard_deviations = np.std(train_x, axis=0)

        # predictions for the test instances
        pred_min = []
        pred_avg = []
        pred_owa = []
        pred_owa_without_outliers = []
        pred_avg_without_outliers = []
        pred_min_without_outliers = []
        pred_fuzzy_removal = []
        pred_two_sym = []
        pred_wowa = []

        for x in tqdm(test_x):

            # Arrays that contain the value of x for a certain lower approximator
            values_min = []
            values_avg = []
            values_owa = []
            values_owa_without_outliers = []
            values_avg_without_outliers = []
            values_min_without_outliers = []
            values_fuzzy_removal = []
            values_two_sym = []
            values_wowa = []

            # Calculates the lower approximations to each class
            for i in range(number_of_classes):
                # weights are the outlier scores and labels the outlier labels
                to_be_aggregated, weights, labels = aggregation_elements(x, train_x, train_y, i, outlier_values,
                                                                         outlier_labels, standard_deviations)
                to_be_aggregated_without_outliers = [k for j, k in enumerate(to_be_aggregated) if labels[j] == 0]

                def add_quantifier(t):
                    return additive_quantifier(t, len(to_be_aggregated))

                def add_quantifier_without_outliers(t):
                    return additive_quantifier(t, len(to_be_aggregated_without_outliers))

                values_min.append(np.amin(to_be_aggregated))
                values_avg.append(np.average(to_be_aggregated))
                values_owa.append(owa(to_be_aggregated, add_quantifier))
                values_min_without_outliers.append(np.amin(to_be_aggregated_without_outliers))
                values_avg_without_outliers.append(np.average(to_be_aggregated_without_outliers))
                values_owa_without_outliers.append(owa(to_be_aggregated_without_outliers,
                                                       add_quantifier_without_outliers))
                values_fuzzy_removal.append(fuzzy_removal_choquet_integral_min(to_be_aggregated, weights))
                values_wowa.append(wowa_outlier(to_be_aggregated, quantifier, weights))

                partition, measure = two_symmetric_measure(labels, 0.3, quantifier)
                values_two_sym.append(k_symmetric_choquet_integral(to_be_aggregated, partition, measure))

            # Classifies it to the class for which it has the greatest lower approximation value
            pred_min.append(np.argmax(values_min))
            pred_avg.append(np.argmax(values_avg))
            pred_owa.append(np.argmax(values_owa))
            pred_owa_without_outliers.append(np.argmax(values_owa_without_outliers))
            pred_avg_without_outliers.append(np.argmax(values_avg_without_outliers))
            pred_min_without_outliers.append(np.argmax(values_min_without_outliers))
            pred_fuzzy_removal.append(np.argmax(values_fuzzy_removal))
            pred_two_sym.append(np.argmax(values_two_sym))
            pred_wowa.append(np.argmax(values_wowa))

        if balanced:
            accuracy_rates_min.append(metrics.balanced_accuracy_score(test_y, pred_min))
            accuracy_rates_avg.append(metrics.balanced_accuracy_score(test_y, pred_avg))
            accuracy_rates_owa.append(metrics.balanced_accuracy_score(test_y, pred_owa))
            accuracy_rates_owa_without_outliers.append(
                metrics.balanced_accuracy_score(test_y, pred_owa_without_outliers))
            accuracy_rates_avg_without_outliers.append(
                metrics.balanced_accuracy_score(test_y, pred_avg_without_outliers))
            accuracy_rates_min_without_outliers.append(
                metrics.balanced_accuracy_score(test_y, pred_min_without_outliers))
            accuracy_rates_fuzzy_removal.append(metrics.balanced_accuracy_score(test_y, pred_fuzzy_removal))
            accuracy_rates_two_sym.append(metrics.balanced_accuracy_score(test_y, pred_two_sym))
            accuracy_rates_wowa.append(metrics.balanced_accuracy_score(test_y, pred_wowa))
        else:
            accuracy_rates_min.append(metrics.accuracy_score(test_y, pred_min))
            accuracy_rates_avg.append(metrics.accuracy_score(test_y, pred_avg))
            accuracy_rates_owa.append(metrics.accuracy_score(test_y, pred_owa))
            accuracy_rates_owa_without_outliers.append(metrics.accuracy_score(test_y, pred_owa_without_outliers))
            accuracy_rates_avg_without_outliers.append(metrics.accuracy_score(test_y, pred_avg_without_outliers))
            accuracy_rates_min_without_outliers.append(metrics.accuracy_score(test_y, pred_min_without_outliers))
            accuracy_rates_fuzzy_removal.append(metrics.accuracy_score(test_y, pred_fuzzy_removal))
            accuracy_rates_two_sym.append(metrics.accuracy_score(test_y, pred_two_sym))
            accuracy_rates_wowa.append(metrics.accuracy_score(test_y, pred_wowa))

    results = [accuracy_rates_min, accuracy_rates_avg,
               accuracy_rates_owa, accuracy_rates_min_without_outliers,
               accuracy_rates_avg_without_outliers, accuracy_rates_owa_without_outliers,
               accuracy_rates_fuzzy_removal,
               accuracy_rates_two_sym,
               accuracy_rates_wowa]
    accuracy = []
    for i in range(len(results)):
        accuracy.append(np.average(results[i]))

    table = Texttable()
    table.add_rows([["Dataset", "Min", "Avg", "OWA", "Mino", "Avgo", "OWAo", "FR", "TS", "WOWA"],
                   [dataset_name, accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4],
                    accuracy[5], accuracy[6], accuracy[7], accuracy[8]]])
    print(table.draw() + "\n")
    print(draw_latex(table) + "\n")
    return accuracy
