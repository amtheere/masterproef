import numpy as np


# Calculates the Choquet integral of "function"
# with respect to the fuzzy outlier removal measure using fuzzy_set as O and a function t_norm as T
def choquet_fuzzy_removal_measure(function, fuzzy_set, t_norm):
    n = len(function)  # Cardinality of X
    argsorted = np.argsort(function)
    result = function[argsorted[0]]
    for i in range(1, n):
        aast = argsorted[i:]
        to_be_tnormed = [fuzzy_set[j] for j in range(n) if j not in aast]
        measure_at_aast = t_norm(to_be_tnormed)
        result += (function[argsorted[i]] - function[argsorted[i - 1]]) * measure_at_aast
    return result


# Calculates the choquet integral of "function"
# with respect to a k-symmetric measure given in the form of a function
def choquet_k_symmetric_measure(function, partition, measure):
    # Cardinality of X
    n = len(function)
    # Number of sets of indifference
    k = len(partition)

    argsorted = np.argsort(function)
    result = function[argsorted[0]]
    for i in range(1, n):
        Aast = argsorted[i:]
        # Vector containing a_1,\dots, a_k
        a = np.zeros(k)
        for j in range(k):
            A_j = partition[j]
            a[j] = len([t for t in Aast if t in A_j])
        measureAtAast = measure(a)
        result += (function[argsorted[i]] - function[argsorted[i - 1]]) * measureAtAast
    return result


# Returns the partition of sets of indifference and the measure
def two_symmetric_measure(outlier_labels, t, Q):
    n = len(outlier_labels)
    outliers = []
    nonOutliers = []
    for i in range(n):
        if outlier_labels[i] == 1:
            outliers.append(i)
        else:
            nonOutliers.append(i)
    partition = [nonOutliers, outliers]
    k = len(nonOutliers)
    return partition, lambda a: Q((a[0] * ((1 - t) * (n / k) + t) + a[1] * t) / n)


def OWA(array, quantifier):
    array = np.array(array)
    array = -1 * np.sort(-1 * array)
    result = 0
    n = len(array)
    for i in range(n):
        result += array[i] * (quantifier((i + 1) / n) - quantifier(i / n))
    return result


def WOWA(array, Q, P):
    array = np.array(array)
    P = np.array(P)
    # Indices of sorted array in decreasing order
    sigma = np.argsort(-1 * array)
    array = np.take(array, sigma)
    P = np.take(P, sigma)
    omega = np.zeros(len(array))
    for i in range(len(array)):
        omega[i] = Q(np.sum(P[:i + 1])) - Q(np.sum(P[:i]))
    return np.dot(array, omega)


def vague_quantifier(x, alpha, beta):
    if (x <= alpha):
        return 0
    elif (alpha <= x and x <= (alpha + beta) / 2):
        return (2 * (x - alpha) ** 2) / (beta - alpha) ** 2
    elif ((alpha + beta) / 2 <= x and x <= beta):
        return 1 - (2 * (x - beta) ** 2) / (beta - alpha) ** 2
    else:
        return 1
