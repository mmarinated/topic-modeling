import io

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = torch.tensor(list(map(float, tokens[1:])))
    return data

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