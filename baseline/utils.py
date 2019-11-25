import io

import torch
import numpy as np
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = torch.tensor(list(map(float, tokens[1:])))
    return data

def create_embeddings_matrix(word_to_index, embeddings):
    vocab_size = len(word_to_index)
    embed_dim = len(list(embeddings.values())[0])
    weights_matrix_ve = np.zeros((vocab_size, embed_dim))

    words_found = 0
    for i, word in enumerate(word_to_index):
        if word in embeddings.keys():
            weights_matrix_ve[i] = embeddings[word]
            words_found += 1
    weights_matrix_ve = torch.FloatTensor(weights_matrix_ve)
    
    print("Total words in vocab: {}".format(vocab_size))
    print("No. of words from vocab found in embeddings: {}".format(words_found))
    return weights_matrix_ve

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

# Function for testing the model
def test_model(loader, model, device, threshold=0.5):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
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
    
    # macro precision, recall, f-score
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        np.vstack(true_list),
        np.vstack(outputs_list),
        average="macro"
    )
    # micro precision, recall, f-score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        np.vstack(true_list),
        np.vstack(outputs_list),
        average="micro"
    )
    # combine all metrics in a dict
    dict_metrics = {
        "precision_macro": precision_macro, 
        "recall_macro": recall_macro, 
        "f1_macro": f1_macro,
        "precision_micro": precision_micro, 
        "recall_micro": recall_micro, 
        "f1_micro": f1_micro
    }
    return dict_metrics

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
