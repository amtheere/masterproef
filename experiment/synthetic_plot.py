import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


# start van o_1 = o_2 = 5 stappen van 5 default 30
# start van omega_1 = 1 stappen van 0.1 default 1.5
# start van omega_2 = 0.5 stappen van 0.1 default 0.7
# start van sigma_1 = 0.2 stappen van 0.1 default 0.6
# start van sigma_2 = 0.1 stappen van 0.1 default 0.2
# steps van alpha_1 en 2 [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2, 2.5, 3.5, 4]

start = 0.1
step_size = 0.1
step = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.4, 3.8, 4]
parameter_name = "alpha_1"
median_accuracy = np.genfromtxt("results_synth/median_" + parameter_name + ".csv", delimiter=", ")
x = step
# x = [start + k * step_size for k in range(len(median_accuracy))]
mini = median_accuracy[:, 0]
mino = median_accuracy[:, 1]
FR = median_accuracy[:, 2]
avg = median_accuracy[:, 3]
avgo = median_accuracy[:, 4]
TS = median_accuracy[:, 5]
OWA = median_accuracy[:, 6]
OWAo = median_accuracy[:, 7]
WOWA = median_accuracy[:, 8]
min_plt = plt.plot(x, mini, alpha=0.5, ls="--", label="min", color='cyan')
mino_plt = plt.plot(x, mino, alpha=0.5, ls="-", label="mino", color="midnightblue")
FR_plt = plt.plot(x, FR, alpha=0.5, ls="-.", label="FR", color="darkmagenta")
avg_plt = plt.plot(x, avg, alpha=0.5, ls="--", label="avg", color="greenyellow")
avgo_plt = plt.plot(x, avgo, alpha=0.5, ls="-", label="avgo", color="darkgreen")
TS_plt = plt.plot(x, TS, alpha=0.5, ls="-.", label="TS", color="darkslategrey")
OWA_plt = plt.plot(x, OWA, alpha=0.5, ls="--", label="OWA", color="lightcoral")
OWAo_plt = plt.plot(x, OWAo, alpha=0.5, ls="-", label="OWAo", color="darkred")
WOWA_plt = plt.plot(x, WOWA, alpha=0.5, ls="-.", label='WOWA', color="orange")
plt.axis(ymax=0.990)
plt.xticks(x, rotation=45)
plt.xlabel(r"$\alpha_1$")
plt.ylabel('accuracy')
plt.legend(ncol=3, loc="best")
plt.savefig("graphs/" + parameter_name + ".png", dpi=300)
plt.show()
median_accuracy = np.genfromtxt("results_synth/median_" + parameter_name + "_outliers.csv", delimiter=", ")
mini = median_accuracy[:, 0]
mino = median_accuracy[:, 1]
FR = median_accuracy[:, 2]
avg = median_accuracy[:, 3]
avgo = median_accuracy[:, 4]
TS = median_accuracy[:, 5]
OWA = median_accuracy[:, 6]
OWAo = median_accuracy[:, 7]
WOWA = median_accuracy[:, 8]
min_plt = plt.plot(x, mini, alpha=0.5, ls="--", label="min", color='cyan')
mino_plt = plt.plot(x, mino, alpha=0.5, ls="-", label="mino", color="midnightblue")
FR_plt = plt.plot(x, FR, alpha=0.5, ls="-.", label="FR", color="darkmagenta")
avg_plt = plt.plot(x, avg, alpha=0.5, ls="--", label="avg", color="greenyellow")
avgo_plt = plt.plot(x, avgo, alpha=0.5, ls="-", label="avgo", color="darkgreen")
TS_plt = plt.plot(x, TS, alpha=0.5, ls="-.", label="TS", color="darkslategrey")
OWA_plt = plt.plot(x, OWA, alpha=0.5, ls="--", label="OWA", color="lightcoral")
OWAo_plt = plt.plot(x, OWAo, alpha=0.5, ls="-", label="OWAo", color="darkred")
WOWA_plt = plt.plot(x, WOWA, alpha=0.5, ls="-.", label='WOWA', color="orange")
plt.legend(ncol=3, loc="best")
plt.axis(ymax=0.96)
plt.xticks(x, rotation=45)
plt.xlabel(r"$\alpha_1$")
plt.ylabel('accuracy')
plt.savefig("graphs/" + parameter_name + "_outliers.png", dpi=300)
plt.show()
