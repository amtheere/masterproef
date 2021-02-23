from pyod.models.lof import LOF
import numpy as np


# R_a(x,y) for a quantitative attribute with index "feature_index"
# sd_a is the standard deviation of the attribute
def fuzzy_relation_a(x, y, sd_a, feature_index):
    if sd_a == 0:
        return 1
    a_x = x[feature_index]
    a_y = y[feature_index]
    return np.maximum(0, np.minimum((a_y - a_x + sd_a) / sd_a, (a_x - a_y + sd_a) / sd_a))


# R(x,y) assuming all features are quantitative
# standard_deviations contains the standard deviation of each attribute
def fuzzy_relation(x, y, standard_deviations):
    number_of_features = len(standard_deviations)
    result = []
    for i in range(number_of_features):
        result.append(fuzzy_relation_a(x, y, standard_deviations[i], i))
    return np.average(result)


# LOF algorithm for outlier score
# https://pyod.readthedocs.io/en/latest/index.html
# Calculates the within class outliers and returns the outlier values (scores) and labels
def outlier_score(x_train, y_train, algorithm=LOF(), method="unify"):
    number_of_instances = len(x_train)
    outlier_values = np.zeros(number_of_instances)
    outlier_labels = np.zeros(number_of_instances)
    number_of_classes = len(set(y_train))
    for i in range(number_of_classes):
        indices_of_class_members = [j for j in range(number_of_instances) if y_train[j] == i]
        class_i_x_values = np.array([x_train[j] for j in indices_of_class_members])
        algorithm.fit(X=class_i_x_values)
        partial_values = algorithm.predict_proba(X=class_i_x_values, method=method)[:, 1]
        partial_labels = algorithm.labels_
        for t in range(len(indices_of_class_members)):
            index = indices_of_class_members[t]
            outlier_values[index] = partial_values[t]
            outlier_labels[index] = partial_labels[t]
    return outlier_values, outlier_labels.astype('int')


# Returns the needed ``elements'' to calculate all the different lower approximations
# of the class with index "class_index"
def aggregation_elements(x, train_x, train_y, class_index, outlier_values, outlier_labels, standard_deviations):
    to_be_aggregated = []
    weights = []
    labels = []
    for i in range(len(train_x)):
        clss = train_y[i]
        if clss != class_index:
            to_be_aggregated.append(1 - fuzzy_relation(x, train_x[i], standard_deviations))
            weights.append(outlier_values[i])
            labels.append(outlier_labels[i])
    return to_be_aggregated, weights, labels


def exp_trans(array, b):
    return list(map(lambda x: ((((1 - b) / b) ** 2) ** x - 1) / (((1 - b) / b) ** 2 - 1), array))
