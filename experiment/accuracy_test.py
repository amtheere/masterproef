from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from tqdm import tqdm
from sklearn import metrics
import random

from aggregation_operators import *
from aggregation_elements import aggregation_elements, outlier_score, exp_trans
# Library for outlier detection
from pyod.models.lof import LOF

ALGORITHMS = ["Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"]


def accuracy_test(train_x, train_y, test_x, test_y,
                  outlierScoreAlgorithm=LOF(contamination=0.1), balanced=False, b=0.75):
    accuracy = []
    number_of_classes = len(set(train_y))

    outlier_values, outlier_labels = outlier_score(train_x, train_y, outlierScoreAlgorithm)
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

            weights = exp_trans(weights, b)
            values_min.append(np.amin(to_be_aggregated))
            values_avg.append(np.average(to_be_aggregated))
            values_owa.append(owa(to_be_aggregated, add_quantifier))
            values_min_without_outliers.append(np.amin(to_be_aggregated_without_outliers))
            values_avg_without_outliers.append(np.average(to_be_aggregated_without_outliers))
            values_owa_without_outliers.append(owa(to_be_aggregated_without_outliers,
                                                   add_quantifier_without_outliers))
            values_fuzzy_removal.append(fuzzy_removal_choquet_integral_min(to_be_aggregated, weights))
            values_wowa.append(wowa_outlier(to_be_aggregated, add_quantifier, weights))

            partition, measure = two_symmetric_measure(labels, 0.3, add_quantifier)
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
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_min))
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_min_without_outliers))
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_fuzzy_removal))
        accuracy.append(
            metrics.balanced_accuracy_score(test_y, pred_avg))
        accuracy.append(
            metrics.balanced_accuracy_score(test_y, pred_avg_without_outliers))
        accuracy.append(
            metrics.balanced_accuracy_score(test_y, pred_two_sym))
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_owa))
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_owa_without_outliers))
        accuracy.append(metrics.balanced_accuracy_score(test_y, pred_wowa))
    else:
        accuracy.append(metrics.accuracy_score(test_y, pred_min))
        accuracy.append(metrics.accuracy_score(test_y, pred_min_without_outliers))
        accuracy.append(metrics.accuracy_score(test_y, pred_fuzzy_removal))
        accuracy.append(metrics.accuracy_score(test_y, pred_avg))
        accuracy.append(metrics.accuracy_score(test_y, pred_avg_without_outliers))
        accuracy.append(metrics.accuracy_score(test_y, pred_two_sym))
        accuracy.append(metrics.accuracy_score(test_y, pred_owa))
        accuracy.append(metrics.accuracy_score(test_y, pred_owa_without_outliers))
        accuracy.append(metrics.accuracy_score(test_y, pred_wowa))
    return accuracy


def accuracy_test_cv(data, target, dataset_name, n_splits=5, stratified=False,
                     outlierScoreAlgorithm=LOF(contamination=0.1), balanced=False, b=0.75):
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

    for train_index, test_index in kf:
        train_x = np.array([data[i] for i in train_index])
        train_y = np.array([target[i] for i in train_index])
        test_x = np.array([data[i] for i in test_index])
        test_y = np.array([target[i] for i in test_index])
        result = accuracy_test(train_x, train_y, test_x, test_y, outlierScoreAlgorithm,
                               balanced, b)
        accuracy_rates_min.append(result[0])
        accuracy_rates_min_without_outliers.append(result[1])
        accuracy_rates_fuzzy_removal.append(result[2])
        accuracy_rates_avg.append(result[3])
        accuracy_rates_avg_without_outliers.append(result[4])
        accuracy_rates_two_sym.append(result[5])
        accuracy_rates_owa.append(result[6])
        accuracy_rates_owa_without_outliers.append(result[7])
        accuracy_rates_wowa.append(result[8])

    results = [accuracy_rates_min, accuracy_rates_min_without_outliers,
               accuracy_rates_fuzzy_removal, accuracy_rates_avg,
               accuracy_rates_avg_without_outliers, accuracy_rates_two_sym,
               accuracy_rates_owa,
               accuracy_rates_owa_without_outliers,
               accuracy_rates_wowa]
    accuracy = []
    for i in range(len(results)):
        accuracy.append(np.average(results[i]))
    return [dataset_name, accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4],
            accuracy[5], accuracy[6], accuracy[7], accuracy[8]]


def accuracy_test_combine_algo_cv(data, target, dataset_name, n_splits=5, stratified=False,
                                  outlierScoreAlgorithm=LOF(contamination=0.1), balanced=False, b=0.75):
    accuracy = []

    # Contains the number of times an algorithm performed the best on the training data
    algorithm_best = np.zeros(len(ALGORITHMS))

    # Contains the number of times an algorithm is used for the classification
    algorithm_used = np.zeros(len(ALGORITHMS))

    # Sets up the cross-validation
    seed = 7
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf = kf.split(data, target)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf = kf.split(data)

    for train_index, test_index in kf:
        train_x = np.array([data[i] for i in train_index])
        train_y = np.array([target[i] for i in train_index])
        test_x = np.array([data[i] for i in test_index])
        test_y = np.array([target[i] for i in test_index])
        best_indices = choose_best_algo(train_x, train_y, outlierScoreAlgorithm, balanced, b)
        for j in best_indices:
            algorithm_best[j] += 1
        used_index = random.choice(best_indices)
        algorithm_used[used_index] += 1
        algo = ALGORITHMS[used_index]

        number_of_classes = len(set(train_y))

        outlier_values, outlier_labels = outlier_score(train_x, train_y, outlierScoreAlgorithm)
        standard_deviations = np.std(train_x, axis=0)

        # predictions for the test instances
        pred = []

        for x in tqdm(test_x):

            # Arrays that contain the value of x for the lower approximation
            values = []

            # Calculates the lower approximations to each class
            for i in range(number_of_classes):
                # weights are the outlier scores and labels are the outlier labels
                to_be_aggregated, weights, labels = aggregation_elements(x, train_x, train_y, i, outlier_values,
                                                                         outlier_labels, standard_deviations)
                to_be_aggregated_without_outliers = [k for j, k in enumerate(to_be_aggregated) if labels[j] == 0]

                def add_quantifier(t):
                    return additive_quantifier(t, len(to_be_aggregated))

                def add_quantifier_without_outliers(t):
                    return additive_quantifier(t, len(to_be_aggregated_without_outliers))

                weights = exp_trans(weights, b)

                if algo == "Min":
                    values.append(np.amin(to_be_aggregated))
                elif algo == "Mino":
                    values.append(np.amin(to_be_aggregated_without_outliers))
                elif algo == "FR":
                    values.append(fuzzy_removal_choquet_integral_min(to_be_aggregated, weights))
                elif algo == "Avg":
                    values.append(np.average(to_be_aggregated))
                elif algo == "Avgo":
                    values.append(np.average(to_be_aggregated_without_outliers))
                elif algo == "TS":
                    partition, measure = two_symmetric_measure(labels, 0.3, add_quantifier)
                    values.append(k_symmetric_choquet_integral(to_be_aggregated, partition, measure))
                elif algo == "OWA":
                    values.append(owa(to_be_aggregated, add_quantifier))
                elif algo == "OWAo":
                    values.append(owa(to_be_aggregated_without_outliers,
                                      add_quantifier_without_outliers))
                elif algo == "WOWA":
                    values.append(wowa_outlier(to_be_aggregated, add_quantifier, weights))

            # Classifies it to the class for which it has the greatest lower approximation value
            pred.append(np.argmax(values))

        if balanced:
            accuracy.append(metrics.balanced_accuracy_score(test_y, pred))
        else:
            accuracy.append(metrics.accuracy_score(test_y, pred))
    accuracy = [dataset_name, np.average(accuracy)]
    algorithm_used = list(algorithm_used)
    algorithm_used.insert(0, dataset_name)
    algorithm_best = list(algorithm_best)
    algorithm_best.insert(0, dataset_name)
    return accuracy, algorithm_used, algorithm_best


def choose_best_algo(train_x, train_y, outlierScoreAlgorithm=LOF(contamination=0.1), balanced=False, b=0.75):
    loocv = LeaveOneOut()
    loocv = loocv.split(train_x)

    # predictions for each instance based on the rest of the dataset
    pred_min = []
    pred_fuzzy_removal = []
    pred_owa = []
    pred_owa_without_outliers = []
    pred_wowa = []
    pred_two_sym = []

    """
    pred_avg = []
    pred_avg_without_outliers = []
    pred_min_without_outliers = []
    """

    true_y = train_y

    for train_index, test_index in tqdm(loocv):
        train_data = np.array([train_x[i] for i in train_index])
        train_target = np.array([train_y[i] for i in train_index])
        test_x = np.array([train_x[i] for i in test_index])

        number_of_classes = len(set(train_target))

        outlier_values, outlier_labels = outlier_score(train_data, train_target, outlierScoreAlgorithm)
        standard_deviations = np.std(train_x, axis=0)

        # Arrays that contain the value of x for a certain lower approximator
        values_min = []
        values_fuzzy_removal = []
        values_owa = []
        values_owa_without_outliers = []
        values_wowa = []
        values_two_sym = []

        """
        values_avg = []
        values_avg_without_outliers = []
        values_min_without_outliers = []
        """

        # Calculates the lower approximations to each class
        for i in range(number_of_classes):
            # weights are the outlier scores and labels the outlier labels
            to_be_aggregated, weights, labels = aggregation_elements(test_x[0], train_data, train_target, i,
                                                                     outlier_values,
                                                                     outlier_labels, standard_deviations)
            to_be_aggregated_without_outliers = [k for j, k in enumerate(to_be_aggregated) if labels[j] == 0]

            def add_quantifier(t):
                return additive_quantifier(t, len(to_be_aggregated))

            def add_quantifier_without_outliers(t):
                return additive_quantifier(t, len(to_be_aggregated_without_outliers))

            weights = exp_trans(weights, b)
            values_min.append(np.amin(to_be_aggregated))
            values_fuzzy_removal.append(fuzzy_removal_choquet_integral_min(to_be_aggregated, weights))
            values_owa.append(owa(to_be_aggregated, add_quantifier))
            values_owa_without_outliers.append(owa(to_be_aggregated_without_outliers,
                                                   add_quantifier_without_outliers))
            values_wowa.append(wowa_outlier(to_be_aggregated, add_quantifier, weights))
            partition, measure = two_symmetric_measure(labels, 0.3, add_quantifier)
            values_two_sym.append(k_symmetric_choquet_integral(to_be_aggregated, partition, measure))

            """
            values_avg.append(np.average(to_be_aggregated))
            values_min_without_outliers.append(np.amin(to_be_aggregated_without_outliers))
            values_avg_without_outliers.append(np.average(to_be_aggregated_without_outliers))
            """

        # Classifies it to the class for which it has the greatest lower approximation value
        pred_min.append(np.argmax(values_min))
        pred_fuzzy_removal.append(np.argmax(values_fuzzy_removal))
        pred_owa.append(np.argmax(values_owa))
        pred_owa_without_outliers.append(np.argmax(values_owa_without_outliers))
        pred_wowa.append(np.argmax(values_wowa))
        pred_two_sym.append(np.argmax(values_two_sym))

        """
        pred_avg.append(np.argmax(values_avg))
        pred_avg_without_outliers.append(np.argmax(values_avg_without_outliers))
        pred_min_without_outliers.append(np.argmax(values_min_without_outliers))
        """

    accuracy = []
    if balanced:
        accuracy.append(metrics.balanced_accuracy_score(true_y, pred_min))
        # accuracy.append(metrics.balanced_accuracy_score(true_y, pred_min_without_outliers))
        accuracy.append(0)
        accuracy.append(metrics.balanced_accuracy_score(true_y, pred_fuzzy_removal))
        accuracy.append(0)
        accuracy.append(0)
        # accuracy.append(
        #     metrics.balanced_accuracy_score(true_y, pred_avg))
        # accuracy.append(
        #     metrics.balanced_accuracy_score(true_y, pred_avg_without_outliers))
        accuracy.append(
             metrics.balanced_accuracy_score(true_y, pred_two_sym))
        accuracy.append(metrics.balanced_accuracy_score(true_y, pred_owa))
        accuracy.append(metrics.balanced_accuracy_score(true_y, pred_owa_without_outliers))
        accuracy.append(metrics.balanced_accuracy_score(true_y, pred_wowa))
    else:
        accuracy.append(metrics.accuracy_score(true_y, pred_min))
        # accuracy.append(metrics.accuracy_score(true_y, pred_min_without_outliers))
        accuracy.append(0)
        accuracy.append(metrics.accuracy_score(true_y, pred_fuzzy_removal))
        accuracy.append(0)
        accuracy.append(0)
        # accuracy.append(metrics.accuracy_score(true_y, pred_avg))
        # accuracy.append(metrics.accuracy_score(true_y, pred_avg_without_outliers))
        accuracy.append(metrics.accuracy_score(true_y, pred_two_sym))
        accuracy.append(metrics.accuracy_score(true_y, pred_owa))
        accuracy.append(metrics.accuracy_score(true_y, pred_owa_without_outliers))
        accuracy.append(metrics.accuracy_score(true_y, pred_wowa))

    best_indices = np.argwhere(accuracy == np.amax(accuracy)).flatten().tolist()
    return best_indices
