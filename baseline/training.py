import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from .model import FinalModel
from .MY_PATHS import *


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

class ClassifierLearner:
    def __init__(self, options, model_name, *, device, criterion=None, optimizer=None):
        self.options = options
        self.model_name = model_name
        self.model = FinalModel(self.options).to(device)
        self.criterion = criterion or torch.nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=3e-3)


        self.best_val_f1_micro = 0
        self.best_metrics_dict = {}
        self.plot_cache = []


    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader


    def set_optim_lr(self, lr):
        assert len(self.optimizer.param_groups) == 1
        self.optimizer.param_groups[0]["lr"] = lr


    def train_model(self, num_epochs=10, lr=3e-3,  save_model=False):
        assert hasattr(self, "train_loader"), "call set_loaders first"
        self.set_optim_lr(lr)

        for epoch in range(num_epochs):
            print(epoch, "epoch")
            runnin_loss = 0.0
            for i, (data, length, labels) in enumerate(self.train_loader):        
                self.model.train()
                data_batch, length_batch, label_batch =\
                    data.to(self.device),length.to(self.device), labels.float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data_batch, length_batch)
                loss = self.criterion(outputs, label_batch)
                loss.backward()
                self.optimizer.step()

                runnin_loss += loss.item()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                if i>0 and i % 100 == 0:
                    print('Epoch: [{}/{}], Step: [{}/{}], Train_loss: {}'.format(
                        epoch+1, num_epochs, i+1, len(self.train_loader), runnin_loss / i))
                # validate every 300 iterations
                if i > 0 and i % 800 == 0:
                    # optimizer.update_swa()
                    self.validate(save_model)


        # optimizer.swap_swa_sgd()
        return self.best_metrics_dict


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


    def validate(self, save_model):
        metrics_dict = self.get_test_metrics(self.val_loader, device=self.device)
        self.print_results(metrics_dict)
        if metrics_dict["f1_micro"] > self.best_val_f1_micro:
            self.best_val_f1_micro = metrics_dict["f1_micro"]
            self.best_metrics_dict = metrics_dict
            if save_model:
                self.save_model()
                

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