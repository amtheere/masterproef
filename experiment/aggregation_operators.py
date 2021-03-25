import numpy as np


# Calculates the Choquet integral of "function"
# with respect to the fuzzy outlier removal measure using fuzzy_set as O and a function t_norm as T
def fuzzy_removal_choquet_integral(function, fuzzy_set, t_norm):
    function_arg_sorted = np.argsort(function)
    result = function[function_arg_sorted[0]]
    to_be_tnormed = []
    for i in range(1, len(function)):
        to_be_tnormed.append(fuzzy_set[function_arg_sorted[i - 1]])
        measure_at_a_ast = t_norm(to_be_tnormed)
        result += (function[function_arg_sorted[i]] - function[function_arg_sorted[i - 1]]) * measure_at_a_ast
    return result


# Optimized version when using the minimum as the t-norm
def fuzzy_removal_choquet_integral_min(function, fuzzy_set):
    function_arg_sorted = np.argsort(function)
    result = function[function_arg_sorted[0]]
    measure_at_a_ast = fuzzy_set[function_arg_sorted[0]]
    result += (function[function_arg_sorted[1]] - function[function_arg_sorted[0]]) * measure_at_a_ast
    for i in range(2, len(function)):
        measure_at_a_ast = min(measure_at_a_ast, fuzzy_set[function_arg_sorted[i - 1]])
        result += (function[function_arg_sorted[i]] - function[function_arg_sorted[i - 1]]) * measure_at_a_ast
    return result


# Calculates the choquet integral of "function" give the partitions of indifference
# with respect to a k-symmetric measure given in the form of a function (see Proposition 3.5.)
def k_symmetric_choquet_integral(function, partition, measure):
    function_arg_sorted = np.argsort(function)
    result = function[function_arg_sorted[0]]
    a_ast = list(function_arg_sorted)

    # Initializing the vector containing a_1,\dots, a_k
    a = [len(partition[i]) for i in range(len(partition))]

    for i in range(1, len(function)):
        removed_elt = a_ast.pop()  # The element x^\ast_{i-1}

        # j is the partition (index) where removed_elt belongs to
        j = 0
        while removed_elt not in partition[j]:
            j += 1

        a[j] -= 1  # Updating the vector a
        measure_at_a_ast = measure(a)
        result += (function[function_arg_sorted[i]] - function[function_arg_sorted[i - 1]]) * measure_at_a_ast
    return result


# Returns the partition of sets of indifference and the measure for the two-symmetric measure
# described in Section 3 Example 3 and 4
def two_symmetric_measure(outlier_labels, t, quantifier):
    n = len(outlier_labels)
    outliers = []
    non_outliers = []
    for i in range(n):
        if outlier_labels[i] == 1:
            outliers.append(i)
        else:
            non_outliers.append(i)
    partition = [non_outliers, outliers]
    k = len(non_outliers)
    return partition, lambda a: quantifier((a[0] * ((1 - t) * (n / k) + t) + a[1] * t) / n)


def owa(array, quantifier):
    array = np.array(array)
    array = -1 * np.sort(-1 * array)
    result = 0
    array_length = len(array)
    for i in range(array_length):
        result += array[i] * (quantifier((i + 1) / array_length) - quantifier(i / array_length))
    return result


def wowa(array, quantifier, weights):
    array = np.array(array)
    weights = np.array(weights)
    # Indices of sorted array in decreasing order
    sigma = np.argsort(-1 * array)
    array = np.take(array, sigma)
    weights = np.take(weights, sigma)
    omega = np.zeros(len(array))
    for i in range(len(array)):
        omega[i] = quantifier(np.sum(weights[:i + 1])) - quantifier(np.sum(weights[:i]))
    return np.dot(array, omega)


# Performs WOWA with weights according to quantifier and the outlier_values
def wowa_outlier(array, quantifier, outlier_values):
    array_length = len(array)
    som = np.sum(outlier_values)
    po = [(1 - outlier_values[i]) / (array_length - som) for i in range(array_length)]
    return wowa(array, quantifier, po)


def additive_owa(array):
    return owa(array, lambda x: additive_quantifier(x, len(array)))


# The quantifier that is needed for the additive OWA weighting scheme.
# Returns Q(x) and n is the length of the array that has to be aggregated
def additive_quantifier(x, n):
    return (x * (x * n + 1)) / (n + 1)


def vague_quantifier(x, alpha, beta):
    if x <= alpha:
        return 0
    elif alpha <= x <= (alpha + beta) / 2:
        return (2 * (x - alpha) ** 2) / (beta - alpha) ** 2
    elif (alpha + beta) / 2 <= x <= beta:
        return 1 - (2 * (x - beta) ** 2) / (beta - alpha) ** 2
    else:
        return 1


def t_norm_luka(array):
    n = len(array)
    som = np.sum(array)
    result = som - n + 1
    if result >= 0:
        return result
    else:
        return 0
