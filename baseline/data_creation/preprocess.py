import io
from collections import namedtuple
from typing import List, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..MY_PATHS import *

np.random.seed(57)

###
# Remove rows in dfs.
###

def remove_rows_with_empty_column(wiki_df_nk : pd.DataFrame, *, column : str) -> None:
    """inplace"""
    valid_n = wiki_df_nk[column].apply(lambda x: len(x) > 0)
    print(f"Percentage of articles with no {column}: {1 - valid_n.mean():.2}"
          f" ({(1 - valid_n).sum()} articles)")
    wiki_df_nk.drop(wiki_df_nk.index[~valid_n], inplace=True)
    wiki_df_nk.reset_index(drop=True, inplace=True)

def _get_common_set_of_QIDs(list_of_lists_of_QIDs : List[List[str]]) -> Set[str]:
    list_of_sets_of_QIDs = [set(list_of_QIDs) for list_of_QIDs in list_of_lists_of_QIDs]
    common_set = list_of_sets_of_QIDs[0].intersection(*list_of_sets_of_QIDs[1:])
    return common_set

def remove_non_common_articles_and_sort_by_QID(languages_dict):
    """Modifies wiki_dfs in languages_dict inplace"""
    list_of_lists_of_QIDs = [cur_dict["wiki_df"].QID for cur_dict in languages_dict.values()]
    print(f"Num of articles initially: \n {languages_dict.keys()} \n {[len(l) for l in list_of_lists_of_QIDs]}")
    common_set = _get_common_set_of_QIDs(list_of_lists_of_QIDs)
    print(f"Num of articles after intersection: \n {len(common_set)}")
    for cur_dict in languages_dict.values():
        mask = cur_dict["wiki_df"].QID.isin(common_set)
        cur_dict["wiki_df"] = cur_dict["wiki_df"][mask]
        cur_dict["wiki_df"].sort_values(by=["QID"], inplace=True)

###
# Vocab
###

def _flatten(tokens : List[List[str]]) -> List[str]:
    return [word for text in list(tokens) for word in text]

def create_vocab_from_tokens(tokens):
    """Returns list of unique tokens given tokens = list of lists."""
    all_tokens = _flatten(tokens)
    # sort in alphabetical order
    vocab = sorted(list(set(all_tokens)))
    return vocab

def create_lookups_for_vocab(vocab, add_tokens_list=[]):
    """
    Can add add_tokens_list in the begining, e.g. ["<pad>", "<unk>"] (will be mapped to indices [0, 1]).
    
    Output
    ------
    index_to_word, word_to_index
    """
    index_to_word = add_tokens_list + vocab
    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
    return index_to_word, word_to_index

###
# Embeddings
###

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

###
# Create tensor dataset
###

def create_dict_of_tensor_datasets(dict_of_dfs, word_to_index, max_num_tokens=None):
    """
    Creates dict of tensor datasets for each df in dict_of_dfs.
    
    Each df in dict_of_dfs should have columns 'tokens', 'labels'.
    """
    wiki_tensor_dataset = {
        name_df : TensoredDataset(
            tokenize_dataset(df, word_to_index, max_num_tokens=max_num_tokens),
            list(df.labels),
        ) 
        for name_df, df  in dict_of_dfs.items()
    }    
    
    return wiki_tensor_dataset

def tokenize_dataset(dataset, word_to_index, max_num_tokens=None):
    """
    If max_num_tokens=None, all tokens are taken.
    """
    _current_dictified = []
    for l in tqdm(dataset['tokens']):
        encoded_l = [
            word_to_index[i] if i in word_to_index 
            else word_to_index['<unk>'] 
            for i in l[:max_num_tokens]
        ]
        _current_dictified.append(encoded_l)
    return _current_dictified


TextData = namedtuple("TextData", ("tokens", "len", "target"))

class TensoredDataset(Dataset):
    def __init__(self, list_of_lists_of_tokens, targets):
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
        return TextData(self.input_tensors[idx], self.input_len[idx], self.target_tensors[idx])
    
    def __repr__(self):
        return "return TextData(self.input_tensors[idx], self.input_len[idx], self.target_tensors[idx])"
    
def _pad_list_of_tensors(list_of_tensors, *, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    
    for t in list_of_tensors:
        #print(t.reshape(1, -1).shape)
        #print(torch.tensor([[pad_token]*(max_length - t.size(-1))])[0].shape)
        padded_tensor = torch.cat([t.reshape(1, -1),
                                   torch.LongTensor([[pad_token]*(max_length - t.size(-1))])], dim=-1)
        padded_list.append(padded_tensor)
    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor

def pad_collate_fn(batch, *, pad_token):
    """pad_token = word_to_index['<pad>']"""
    input_list, length_list, target_list = zip(*batch)
    text_data = TextData(
        tokens = _pad_list_of_tensors(input_list, pad_token=pad_token),
        len    = torch.stack(length_list),
        target = torch.stack(target_list),
    )
    return text_data



# def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
#     np.random.seed(seed)
#     perm = np.random.permutation(df.index)
#     m = len(df.index)
#     train_end = int(train_percent * m)
#     validate_end = int(validate_percent * m) + train_end
#     train, val, test = df.iloc[perm[:train_end]], df.iloc[perm[train_end:validate_end]], df.iloc[perm[validate_end:]]
#     return train, val, test
