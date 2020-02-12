from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.model import FinalModel
from baseline.MY_PATHS import *


def get_train_val_loader(train_dataset, list_val_dataset, *,
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
    
    assert isinstance(list_val_dataset, list)

    list_val_loaders = [
        DataLoader(val_dataset, shuffle=False, **loader_kw)
        for val_dataset in list_val_dataset
    ]
    
    return train_loader, list_val_loaders

class ClassifierLearner:
    def __init__(self, options, model_name, *, device, criterion=None, optimizer=None):
        self.options = options
        self.model_name = model_name
        self.device = device
        self.model = FinalModel(self.options).to(device)
        self.criterion = criterion if criterion is not None else torch.nn.BCEWithLogitsLoss()
        # we change lr later
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=3e-3)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10)

        self.logger = SummaryWriter(PATH_TO_TENSORBOARD_RUNS, comment=self.model_name)
        # self.logger.add_text("model_description", self._model.__repr__())

        self.best_epoch = -1
        self.best_val_f1_micro = -1
        self.best_metrics_dict = {}
        self.list_metrics_dict_by_epoch = []
        self.plot_cache = []
        self.losses = []


    def set_loaders(self, train_loader, name_to_val_loader: Dict[str, Any]):
        self.train_loader = train_loader
        self.name_to_val_loader = name_to_val_loader
        assert "val" in self.name_to_val_loader.keys(), \
               "it is main val loader that should be always present"

    def _set_optim_lr(self, lr):
        assert len(self.optimizer.param_groups) == 1
        self.optimizer.param_groups[0]["lr"] = lr


    def train_model(self, num_epochs=10, lr=3e-3,  save_model=False):
        assert hasattr(self, "train_loader"), "call set_loaders first"
        self._set_optim_lr(lr)

        for epoch in range(num_epochs):
            print(epoch, "epoch")
            runnin_loss = 0.0
            for batch_idx, (data, length, labels) in enumerate(self.train_loader):        
                self.model.train()
                data_batch, length_batch, label_batch =\
                    data.to(self.device),length.to(self.device), labels.float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data_batch, length_batch)
                loss = self.criterion(outputs, label_batch)
                loss.backward()
                self.optimizer.step()

                runnin_loss += loss.item()
                self.losses.append(loss.item())
                print(f"Loss: {loss.item()}")
                #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                # if i>0 and i % 100 == 0:
                #     print('Epoch: [{}/{}], Step: [{}/{}], Train_loss: {}'.format(
                #         epoch+1, num_epochs, i+1, len(self.train_loader), runnin_loss / i))

                # # validate every 300 iterations
                # if i > 0 and i % 800 == 0:
                #     # optimizer.update_swa()
                #     self.validate(loader=self.val_loader, epoch=epoch, save_model=save_model)


            # updating
            new_f1_score = self.validate(loader=self.name_to_val_loader["val"], epoch=epoch, save_model=save_model)
            self.lr_scheduler.step(new_f1_score)
            # logging
            self.logger.add_scalar("train/bce_loss", runnin_loss, epoch)
            _tmp_f1_micro = self.get_test_metrics(self.train_loader, device=self.device)["f1_micro"]
            self.logger.add_scalar("f1_micro/train", _tmp_f1_micro, epoch)
            for val_loader_name, val_loader in self.name_to_val_loader.items():
                _tmp_f1_micro = self.get_test_metrics(val_loader, device=self.device)["f1_micro"]
                self.logger.add_scalar(f"f1_micro/{val_loader_name}", _tmp_f1_micro, epoch)

        # optimizer.swap_swa_sgd()
        return self.best_metrics_dict, self.best_epoch

    def get_test_metrics(self, loader, *, device, threshold=0.5):
        """
        Help function that tests the model's performance on a dataset
        @param: loader - data loader for the dataset to test against
        """
        self.model.eval()
        outputs_list = []
        true_list = []
        with torch.no_grad():
            for data, length, labels in loader:
                data_batch, length_batch, label_batch = data.to(device), length.to(device), labels.float()
                outputs_bc = torch.sigmoid(self.model(data_batch, length_batch))
                outputs_bc = outputs_bc.detach().cpu().numpy()
                outputs_bc = (outputs_bc > threshold)
                outputs_list.append(outputs_bc)
                true_list.append(label_batch)
        
        true_nc = np.vstack(true_list)
        pred_nc = np.vstack(outputs_list)
        n_c = true_nc.shape[1]
        present_classes_C = list(np.arange(n_c)[true_nc.sum(axis=0) > 0])
        # macro precision, recall, f-score
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_nc,
            pred_nc,
            average="macro",
            labels=present_classes_C
        )
        # micro precision, recall, f-score
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_nc,
            pred_nc,
            average="micro",
            labels=present_classes_C
        )
        # combine all metrics in a dict
        dict_metrics = {
            "precision_macro": precision_macro, 
            "recall_macro": recall_macro, 
            "f1_macro": f1_macro,
            "precision_micro": precision_micro, 
            "recall_micro": recall_micro, 
            "f1_micro": f1_micro,
        }
        return dict_metrics

    def get_test_metrics_kfold(self, num_splits, dataset):
        indices_splits = [
            np.arange(start, len(dataset), num_splits) 
            for start in range(num_splits)
        ]
        
        list_of_metrics = [
            self.get_test_metrics(data.Subset(dataset, indices), device=self.device)
            for indices in indices_splits
        ]
        
        return pd.DataFrame(list_of_metrics)

    def validate(self, loader, epoch, save_model) -> float:
        """
        Updates best_val_score
        
        @returns
            f1_micro score
        """
        metrics_dict = self.get_test_metrics(loader, device=self.device)
        self.print_results(metrics_dict)
        self.list_metrics_dict_by_epoch.append(metrics_dict)
        if metrics_dict["f1_micro"] > self.best_val_f1_micro:
            self.best_epoch = epoch
            self.best_val_f1_micro = metrics_dict["f1_micro"]
            self.best_metrics_dict = metrics_dict
            if save_model:
                self.save_model()
                
        return metrics_dict["f1_micro"]

    def save_model(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'options': self.options,
            'plot_cache': self.plot_cache,
        },
            f'{PATH_TO_MODELS_FOLDER}{self.model_name}.pth')

        print('Model Saved\n')

    @staticmethod
    def print_results(metrics_dict):
        """Prettily prints metrics dict."""
        metrics_dict = {key: round(value, 4) for key, value in metrics_dict.items()}
        print("Precision macro: {}, Recall macro: {}, F1 macro: {} ".format(
            metrics_dict["precision_macro"], metrics_dict["recall_macro"], metrics_dict["f1_macro"]
        ))
        print("Precision micro: {}, Recall micro: {}, F1 micro: {} ".format(
            metrics_dict["precision_micro"], metrics_dict["recall_micro"], metrics_dict["f1_micro"]
        ))


###
##  utils to write concise for-loop
###

from functools import partial
from baseline.data_creation.preprocess import pad_collate_fn


def get_model_name_from_options(options):
    return "_".join([f"{k}_{v}" for k, v in options.items()])

def fit_model_given_options(
        options, dict_wiki_tensor_dataset, *,
        num_epochs, lr,
        pad_idx, device, TRAIN_SET_NAME,
        VAL_KEYS):

    model_name = get_model_name_from_options(options)
    learner = ClassifierLearner(options, model_name, device=device)

    train_loader, dict_val_loader = get_train_val_loader(
        dict_wiki_tensor_dataset[TRAIN_SET_NAME], 
        [dict_wiki_tensor_dataset[key] for key in VAL_KEYS], 
        collate_fn=partial(pad_collate_fn, pad_token=pad_idx)
    )

    dict_val_loader = {
        key : val_loader 
        for key, val_loader in zip(VAL_KEYS, dict_val_loader)
    } 

    learner.set_loaders(train_loader, dict_val_loader)
    learner.train_model(num_epochs=num_epochs, lr=lr)

    return learner
