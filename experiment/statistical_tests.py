from scipy import stats
import numpy as np
import pandas as pd
import scikit_posthocs as posthocs
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_results = np.loadtxt("benchmark_results_5.csv", delimiter=", ")
mini = benchmark_results[:, 0]
mino = benchmark_results[:, 3]
FR = benchmark_results[:, 6]
OWA = benchmark_results[:, 2]
OWAo = benchmark_results[:, 5]
print(np.median([mini, mino, FR, OWA, OWAo], axis=1))
print(stats.wilcoxon(mino, FR, alternative="two-sided"))
print(stats.friedmanchisquare(benchmark_results[:, 0], benchmark_results[:, 1], benchmark_results[:, 2],
                              benchmark_results[:, 3],
                              benchmark_results[:, 4], benchmark_results[:, 5], benchmark_results[:, 6],
                              benchmark_results[:, 7],
                              benchmark_results[:, 8]))
pc = posthocs.posthoc_wilcoxon(benchmark_results.T)
annotations = np.round(pd.DataFrame.to_numpy(pc), 2)


def annotation(x):
    if x < 0.01:
        return str("<0.01")
    return x


annotations = list(map(lambda x: list(map(annotation, x)), annotations))
labels = ["Min", "Avg", "OWA", "Mino", "Avgo", "OWAo", "FR", "TS", "WOWA"]
sns.heatmap(pc, annot=annotations, fmt='', xticklabels=labels, yticklabels=labels,
            vmin=0.01, vmax=0.2, cbar=False)
plt.show()
