import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (multilabel_confusion_matrix,
                             precision_recall_fscore_support)

def get_true_and_pred_labels(loader, model, device, threshold=0.5):
    """Returns true labels and predictions."""
    model.eval()
    outputs_list = []
    true_list = []
    with torch.no_grad():
        for data, length, labels in loader:
            data_batch, length_batch, label_batch = data.to(device), length.to(device), labels.float()
            outputs_bc = torch.sigmoid(model(data_batch, length_batch))
            outputs_bc = outputs_bc.detach().cpu().numpy()
            outputs_bc = (outputs_bc > threshold)
            outputs_list.append(outputs_bc)
            true_list.append(label_batch)
    return np.vstack(true_list), np.vstack(outputs_list)

def create_per_class_tables(loader, model, device, class_names, threshold=0.5):
    """
    Function that creates per class tables with count, TN, FN, TP, FP, precision, recall, f1.
    @param: loader - data loader for the dataset to test against
    """
    model.eval()
    outputs_list_nc = []
    true_list_nc = []
    with torch.no_grad():
        for data, length, labels in loader:
            data_batch, length_batch, label_batch = data.to(device), length.to(device), labels.float()
            outputs_bc = torch.sigmoid(model(data_batch, length_batch))
            outputs_bc = outputs_bc.detach().cpu().numpy().astype(np.float)
            outputs_bc = (outputs_bc > threshold)
            outputs_list_nc.append(outputs_bc)
            true_list_nc.append(label_batch.detach().cpu().numpy().astype(np.float))
    # to np.array
    outputs_list_nc = np.vstack(outputs_list_nc)
    true_list_nc = np.vstack(true_list_nc)
    
    # per class counts
    counts_c = true_list_nc.sum(axis=0)
    
    # per class confusion matrix: TN, FN, TP, FP
    confusion_matrix_c22 = multilabel_confusion_matrix(
        true_list_nc,
        outputs_list_nc,
    )
    confusion_matrix_c4 = confusion_matrix_c22.reshape(-1, 4)
    
    # per class precision, recall, f-score
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        true_list_nc,
        outputs_list_nc,
        average=None
    )

    # combine all metrics in a dict
    per_class_metrics = {
        "class_name": class_names,
        "count": counts_c,
        "TN": confusion_matrix_c4[:, 0],
        "FN": confusion_matrix_c4[:, 2],
        "TP": confusion_matrix_c4[:, 3],
        "FP": confusion_matrix_c4[:, 1],
        "precision": precision_c,
        "recall": recall_c,
        "f1": f1_c
    }
    return pd.DataFrame(per_class_metrics)


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
        fig, axis = plt.subplots(1, 1)
#         fig.set_size_inches(10, 6)
    
    n_m = len(labels_m)
    shift_m = np.linspace(0, 0.2, n_m)


    def add_errorbars(mean_k, std_k, shift, *, label=None, color=None):
        axis.errorbar(np.arange(mean_k.size) + shift,
                    mean_k, yerr=std_k, capsize=5,
                    fmt='o', label=label, color=color,
                    ecolor=color)

#     axis.set_title("Comparison of models")
    axis.set_xticks(range(len(xticklabels_k)))
    axis.set_xticklabels(labels=xticklabels_k)


    for mean_k, std_k, shift, label, color in\
            zip(mean_mk, std_mk, shift_m, labels_m, colors_m):
        add_errorbars(mean_k, std_k, shift, label=label, color=color)

    axis.grid()
    axis.legend()

    return axis
