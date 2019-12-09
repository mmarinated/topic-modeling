import matplotlib.pyplot as plt
import numpy as np


def get_mean_std_k(learner, *, num_splits, dict_wiki_tensor_dataset, metric_name="f1_micro",
                   keys=["val", "val_en", "val_ru", "val_hi"]):
    mean_k = []
    std_k  = []
    
    for key in keys:
        dataset = dict_wiki_tensor_dataset[key]
        cur_df = learner.get_test_metrics_kfold(num_splits, dataset)
        mean_k.append(cur_df.mean()[metric_name])
        std_k.append(cur_df.std()[metric_name])
    
    return np.array(mean_k), np.array(std_k)

def plot_errorbars_by_model(mean_mk, std_mk,
        labels_m=["all languages model", "english only"],
        colors_m="bgryck",
        xticklabels_k=["average", "en", "ru", "hi"],
        axis=None):
    assert len(mean_mk) == len(std_mk) == len(labels_m)
    assert len(colors_m) >= len(mean_mk)
    if axis is None:
        _, axis = plt.subplots(1, 1)
    
    n_m = len(labels_m)
    shift_m = np.linspace(0, 0.2, n_m)


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
