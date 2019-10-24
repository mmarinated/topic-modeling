# load the embeddings
import io

# import dependencies
import nltk
import json
import io
import gzip
import torch
import string
import random
import jsonlines
import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('stopwords')

def remove_stop_words(tokens, language="english"):
    stop_words = nltk.corpus.stopwords.words(language)
    result = []
    for token in tokens:
        if not token in stop_words:
            result.append(token)
    return result

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def tokenize_dataset(dataset, word_to_index):
    _current_dictified = []
    for l in tqdm(dataset['tokens']):
        encoded_l = [word_to_index[i] if i in word_to_index else word_to_index['<unk>'] for i in l]
        _current_dictified.append(encoded_l)
    return _current_dictified

class TensoredDataset(Dataset):
    def __init__(self, list_of_lists_of_tokens,targets):
        self.input_tensors = []
        self.target_tensors = []
        self.input_len = []
        
        for i in range(len(list_of_lists_of_tokens)):
            self.input_tensors.append(torch.LongTensor(list_of_lists_of_tokens[i]))
            self.target_tensors.append(torch.LongTensor(targets[i]))
            self.input_len.append(torch.FloatTensor([len(list_of_lists_of_tokens[i])]))
    
    def __len__(self):
        return len(self.input_tensors)
    
    def __getitem__(self, idx):
        # return a (input, target) tuple
        return (self.input_tensors[idx], self.input_len[idx], self.target_tensors[idx])
    
def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    
    for t in list_of_tensors:
        #print(t.reshape(1, -1).shape)
        #print(torch.tensor([[pad_token]*(max_length - t.size(-1))])[0].shape)
        padded_tensor = torch.cat([t.reshape(1, -1), torch.LongTensor([[pad_token]*(max_length - t.size(-1))])], dim = -1)
        padded_list.append(padded_tensor)
    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor

def pad_collate_fn(batch, word_to_index):
    # batch is a list of sample tuples
    input_list = [s[0] for s in batch]
    length_list = [s[1] for s in batch]
    target_list = [s[2] for s in batch]
    
    #pad_token = persona_dict.get_id('<pad>')
    pad_token = word_to_index['<pad>']
    
    input_tensor = pad_list_of_tensors(input_list, pad_token)    
    length_tensor = torch.stack(length_list)
    target_tensor = torch.stack(target_list)
    
    return input_tensor, length_tensor, target_tensor