from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from sklearn import metrics

from aggregation_operators import *
from aggregation_elements import aggregation_elements, lof_score

# Library for outlier detection
from pyod.models.lof import LOF


def accuracy_test(data, target, n_splits=5, stratified=False, outlierScoreAlgorithm=LOF(), balanced=False):
    # Sets up the cross-validation
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        kf = kf.split(data, target)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True)
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

    def quantifier(t): return vague_quantifier(t, 0.2, 1)

    for train_index, test_index in kf:
        train_x = np.array([data[i] for i in train_index])
        train_y = np.array([target[i] for i in train_index])
        test_x = np.array([data[i] for i in test_index])
        test_y = np.array([target[i] for i in test_index])
        number_of_classes = len(set(train_y))

        outlier_values, outlier_labels = lof_score(train_x, train_y, outlierScoreAlgorithm)
        standard_deviations = np.std(train_x, axis=0)

        # predictions
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
                # O is the fuzzy outlier set and OL the outlier labels
                to_be_aggregated, weights, labels = aggregation_elements(x, train_x, train_y, i, outlier_values,
                                                                         outlier_labels, standard_deviations)
                to_be_aggregated_without_outliers = [j for i, j in enumerate(to_be_aggregated) if labels[i] == 0]
                partition, measure = two_symmetric_measure(labels, 0.3, quantifier)
                values_min.append(np.amin(to_be_aggregated))
                values_avg.append(np.average(to_be_aggregated))
                values_owa.append(owa(to_be_aggregated, quantifier))
                values_min_without_outliers.append(np.amin(to_be_aggregated_without_outliers))
                values_avg_without_outliers.append(np.average(to_be_aggregated_without_outliers))
                values_owa_without_outliers.append(owa(to_be_aggregated_without_outliers, quantifier))
                values_fuzzy_removal.append(fuzzy_removal_choquet_integral(to_be_aggregated, weights, np.amin))
                values_two_sym.append(k_symmetric_choquet_integral(to_be_aggregated, partition, measure))
                n = len(weights)
                som = np.sum(weights)
                po = [(1 - weights[i]) / (n - som) for i in range(n)]
                values_wowa.append(wowa(to_be_aggregated, quantifier, po))

            # Classifies it to class for which it has the greatest lower approximation value
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
    print("Min: " + str(accuracy[0]))
    print("Avg: " + str(accuracy[1]))
    print("OWA: " + str(accuracy[2]))
    print("Min without outliers: " + str(accuracy[3]))
    print("Avg without outliers: " + str(accuracy[4]))
    print("OWA without outliers: " + str(accuracy[5]))
    print("Fuzzy removal: " + str(accuracy[6]))
    print("Two symmetric: " + str(accuracy[7]))
    print("WOWA: " + str(accuracy[8]))
    return accuracy
