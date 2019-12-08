import io
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
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

def create_embeddings_matrix(index_to_word, embeddings):
    vocab_size = len(index_to_word)
    embed_dim = len(list(embeddings.values())[0])
    weights_matrix_ve = np.zeros((vocab_size, embed_dim))

    words_found = 0
    for i, word in enumerate(index_to_word):
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

def print_results(metrics_dict):
    """Prettily prints metrics dict."""
    metrics_dict = {key: round(value, 4) for key, value in metrics_dict.items()}
    print("Precision macro: {}, Recall macro: {}, F1 macro: {} ".format(
        metrics_dict["precision_macro"], metrics_dict["recall_macro"], metrics_dict["f1_macro"]
    ))
    print("Precision micro: {}, Recall micro: {}, F1 micro: {} ".format(
        metrics_dict["precision_micro"], metrics_dict["recall_micro"], metrics_dict["f1_micro"]
    ))
    
def get_train_val_loader(train_dataset, val_dataset, *,
                         batch_size=8,
                         collate_fn=None):
    loader_kw = {
        "batch_size" : batch_size,
        "collate_fn" : collate_fn,
    }
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **loader_kw,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        **loader_kw,
    )
    
    return train_loader, val_loader
    
def train_model(train_loader, val_loader, model, criterion, optimizer, options, device,
                num_epochs=10, model_name="model", save_model=False):
    best_val_f1_micro = 0
    best_metrics_dict = {}
    plot_cache = []
    for epoch in range(num_epochs):
        print(epoch, "epoch")
        runnin_loss = 0.0
        for i, (data, length, labels) in enumerate(train_loader):        
            model.train()
            data_batch, length_batch, label_batch = data.to(device),length.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            runnin_loss += loss.item()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            if i>0 and i % 100 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Train_loss: {}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), runnin_loss / i))
            # validate every 300 iterations
            if i > 0 and i % 800 == 0:
                # optimizer.update_swa()
                metrics_dict = test_model(val_loader, model, device=device)
                print_results(metrics_dict)
                if metrics_dict["f1_micro"] > best_val_f1_micro:
                    best_val_f1_micro = metrics_dict["f1_micro"]
                    best_metrics_dict = metrics_dict
                    if save_model:
                        # optimizer.swap_swa_sgd()
                        # torch.save(model.state_dict(), f"{PATH_TO_MODELS_FOLDER}{model_name}.pth")
                        torch.save({
                            'state_dict': model.state_dict(),
                            'options': options,
                            'plot_cache': plot_cache,
                        },
                            f'{PATH_TO_MODELS_FOLDER}{model_name}.pth')
     
                        print('Model Saved')
                        print()
    # optimizer.swap_swa_sgd()
    return best_metrics_dict

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
