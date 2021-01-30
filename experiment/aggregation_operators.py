import numpy as np


# Calculates the Choquet integral of "function"
# with respect to the fuzzy outlier removal measure using fuzzy_set as O and a function t_norm as T
def fuzzy_removal_choquet_integral(function, fuzzy_set, t_norm):
    n = len(function)  # Cardinality of X
    function_arg_sorted = np.argsort(function)
    result = function[function_arg_sorted[0]]
    for i in range(1, n):
        a_ast = function_arg_sorted[i:]
        to_be_tnormed = [fuzzy_set[j] for j in range(n) if j not in a_ast]
        measure_at_a_ast = t_norm(to_be_tnormed)
        result += (function[function_arg_sorted[i]] - function[function_arg_sorted[i - 1]]) * measure_at_a_ast
    return result


# Calculates the choquet integral of "function"
# with respect to a k-symmetric measure given in the form of a function given also the partition off indifference
def k_symmetric_choquet_integral(function, partition, measure):
    # Cardinality of X
    n = len(function)
    # Number of sets of indifference
    k = len(partition)

    function_arg_sorted = np.argsort(function)
    result = function[function_arg_sorted[0]]
    for i in range(1, n):
        a_ast = function_arg_sorted[i:]
        # Vector containing a_1,\dots, a_k
        a = np.zeros(k)
        for j in range(k):
            a_j = partition[j]
            a[j] = len([t for t in a_ast if t in a_j])
        measure_at_a_ast = measure(a)
        result += (function[function_arg_sorted[i]] - function[function_arg_sorted[i - 1]]) * measure_at_a_ast
    return result


# Returns the partition of sets of indifference and the measure
def two_symmetric_measure(outlier_labels, t, Q):
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
    return partition, lambda a: Q((a[0] * ((1 - t) * (n / k) + t) + a[1] * t) / n)


def owa(array, quantifier):
    array = np.array(array)
    array = -1 * np.sort(-1 * array)
    result = 0
    n = len(array)
    for i in range(n):
        result += array[i] * (quantifier((i + 1) / n) - quantifier(i / n))
    return result


def wowa(array, q, p):
    array = np.array(array)
    p = np.array(p)
    # Indices of sorted array in decreasing order
    sigma = np.argsort(-1 * array)
    array = np.take(array, sigma)
    p = np.take(p, sigma)
    omega = np.zeros(len(array))
    for i in range(len(array)):
        omega[i] = q(np.sum(p[:i + 1])) - q(np.sum(p[:i]))
    return np.dot(array, omega)


def vague_quantifier(x, alpha, beta):
    if x <= alpha:
        return 0
    elif alpha <= x <= (alpha + beta) / 2:
        return (2 * (x - alpha) ** 2) / (beta - alpha) ** 2
    elif (alpha + beta) / 2 <= x <= beta:
        return 1 - (2 * (x - beta) ** 2) / (beta - alpha) ** 2
    else:
        return 1
