import matplotlib.pyplot as plt
import numpy as np


def plot_errorbars_by_model(mean_mk, std_mk,
        labels_m=["all languages model", "english only"],
        colors_m=["red", "green"],
        xticklabels_k=["average", "en", "ru", "hi"],
        axis=None):
    assert len(mean_mk) == len(std_mk) == len(labels_m) == len(colors_m)
    if axis is None:
        _, axis = plt.subplots(1, 1)
    
    n_m = len(labels_m)
    shift_m = np.linspace(0, 0.4, n_m)


    def add_errorbars(mean_k, std_k, shift, *, label=None, color=None):
        axis.errorbar(np.arange(mean_k.size) + shift,
                    mean_k, yerr=std_k, capsize=5,
                    fmt='o', label=label, color=color,
                    ecolor=color)

    axis.set_title("Comparison of models")
    axis.set_xticks(range(len(xticklabels_k)))
    axis.set_xticklabels(labels=xticklabels_k)


    for mean_k, std_k, shift, label, color in\
            zip(mean_mk, std_mk, shift_m, labels_m, colors_m):
        add_errorbars(mean_k, std_k, shift, label=label, color=color)

    axis.grid()
    axis.legend()

    return axis
