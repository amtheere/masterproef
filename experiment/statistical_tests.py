from scipy import stats
import numpy as np
import pandas as pd
import scikit_posthocs as posthocs
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_results = np.loadtxt("benchmark_results.csv", delimiter=", ")
mini = benchmark_results[:, 0]
mino = benchmark_results[:, 1]
FR = benchmark_results[:, 2]
avg = benchmark_results[:, 3]
avgo = benchmark_results[:, 4]
TS = benchmark_results[:, 5]
OWA = benchmark_results[:, 6]
OWAo = benchmark_results[:, 7]
WOWA = benchmark_results[:, 8]
print(stats.wilcoxon(WOWA, avgo, alternative="two-sided"))
pc = posthocs.posthoc_wilcoxon([mini, mino, FR, avg, avgo, TS, OWA, OWAo, WOWA], zero_method="pratt")
annotations = np.round(pd.DataFrame.to_numpy(pc), 2)


def annotation(x):
    if x < 0.01:
        return str("<0.01")
    return x


annotations = list(map(lambda x: list(map(annotation, x)), annotations))
labels = ["Min", "Mino", "FR", "Avg", "Avgo", "TS", "OWA", "OWAo", "WOWA"]
ax = sns.heatmap(pc, annot=annotations, fmt='', xticklabels=labels, yticklabels=labels,
                 vmin=0.01, vmax=0.2, cbar=False)
ax.hlines([3, 6], 0, 9, linewidth=0.1, color="black")
ax.vlines([3, 6], 0, 9, linewidth=0.1, color="black")
plt.show()