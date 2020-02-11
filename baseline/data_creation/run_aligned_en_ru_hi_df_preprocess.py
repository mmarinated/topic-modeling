import pickle as pkl
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import sys
sys.path.append("../../")

from baseline.MY_PATHS import *

from baseline.data_creation.preprocess import (create_lookups_for_vocab, create_vocab_from_tokens,
                          remove_non_common_articles_and_sort_by_QID,
                          remove_rows_with_empty_column)
from baseline.utils import get_classes_list

def get_dict_of_split_sizes_and_QIDs(QIDs, LANGUAGES_LIST, train_size=0.8, val_size=0.1):
    """
    Assumes there are: 
    splits = ["full", "monolingual_train", "multilingual_train", "val", "test"].
    
    """
    splits = ["full", "monolingual_train", "multilingual_train", "val", "test"]
    SPLIT_DICT = {split: {} for split in splits}
    SPLIT_DICT["full"]["size"]               = len(QIDs)
    SPLIT_DICT["monolingual_train"]["size"]  = int(train_size * SPLIT_DICT["full"]["size"])
    SPLIT_DICT["multilingual_train"]["size"] = SPLIT_DICT["monolingual_train"]["size"] // len(LANGUAGES_LIST)
    SPLIT_DICT["val"]["size"]                = int(val_size * SPLIT_DICT["full"]["size"])
    SPLIT_DICT["test"]["size"] = (
        SPLIT_DICT["full"]["size"]
        - SPLIT_DICT["monolingual_train"]["size"]
        - SPLIT_DICT["val"]["size"]
    )
    print(*SPLIT_DICT.items(), sep="\n")
    
    SPLIT_DICT["monolingual_train"]["QIDs"], val_and_test_QIDs = train_test_split(
        QIDs, 
        train_size=SPLIT_DICT["monolingual_train"]["size"], 
        random_state=SEED
    )
    SPLIT_DICT["multilingual_train"]["QIDs"], _ = train_test_split(
        SPLIT_DICT["monolingual_train"]["QIDs"], 
        train_size=SPLIT_DICT["multilingual_train"]["size"], 
        random_state=SEED
    )
    SPLIT_DICT["val"]["QIDs"], SPLIT_DICT["test"]["QIDs"] = train_test_split(
        val_and_test_QIDs, 
        train_size=SPLIT_DICT["val"]["size"], 
        random_state=SEED
    )
    return SPLIT_DICT

# Load list of classes
classes_list = get_classes_list(PATH_TO_DATA_FOLDER + "classes.txt")

SAVE = False
LOAD = True

SEED = 57

LANGUAGES_LIST = ["english", "russian", "hindi"]
print(LANGUAGES_LIST)
LANGUAGES_DICT = defaultdict(dict)

for language in LANGUAGES_LIST:
    print(language)
    # Get paths to files
    LANGUAGES_DICT[language]["FILE_NAMES_DICT"] = get_paths(language)

    # Load wiki_df
    wiki_df = pkl.load(open(LANGUAGES_DICT[language]["FILE_NAMES_DICT"]["wiki_df"], "rb"))
    LANGUAGES_DICT[language]["wiki_df"] = wiki_df

    # Remove rows with empty labels/tokens
    remove_rows_with_empty_column(LANGUAGES_DICT[language]["wiki_df"], column="mid_level_categories")
    remove_rows_with_empty_column(LANGUAGES_DICT[language]["wiki_df"], column="tokens")

# This step should be done BEFORE the splits
remove_non_common_articles_and_sort_by_QID(LANGUAGES_DICT)

for cur_dict in LANGUAGES_DICT.values():
    # Binarize labels
    mlb = MultiLabelBinarizer(classes_list)
    cur_dict["wiki_df"]["labels"] =\
        list(mlb.fit_transform(cur_dict["wiki_df"].mid_level_categories))
    assert (mlb.classes_ == classes_list).all()
    
    # Create and save OR load vocabulary
    if SAVE:
        vocab = create_vocab_from_tokens(cur_dict["wiki_df"]["tokens"])
        torch.save(vocab, cur_dict["FILE_NAMES_DICT"]["vocab"])
        print("Saved: ", cur_dict["FILE_NAMES_DICT"]["vocab"])
    if LOAD:
        vocab = torch.load(cur_dict["FILE_NAMES_DICT"]["vocab"])

    index_to_word, word_to_index = create_lookups_for_vocab(vocab)
    cur_dict["index_to_word"], cur_dict["word_to_index"] = index_to_word, word_to_index

# train/val/test sizes and QIDs
splits = ["monolingual_train", "multilingual_train", "val", "test"]
QIDs = LANGUAGES_DICT["english"]["wiki_df"].QID
SPLIT_DICT = get_dict_of_split_sizes_and_QIDs(QIDs, LANGUAGES_LIST)

# Create and save OR load splitted dfs
for cur_dict in LANGUAGES_DICT.values():
    dict_of_dfs = defaultdict()
    if SAVE:
        for split in ["monolingual_train", "multilingual_train", "val", "test"]:
            dict_of_dfs[split] = cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(SPLIT_DICT[split]["QIDs"])]
            # save
            torch.save(dict_of_dfs[split], cur_dict["FILE_NAMES_DICT"][split])
            print("Saved:, ", cur_dict["FILE_NAMES_DICT"][split])
    if LOAD:
        for split in ["monolingual_train", "multilingual_train", "val", "test"]:
            dict_of_dfs[split] = torch.load(cur_dict["FILE_NAMES_DICT"][split])


    cur_dict["dict_of_dfs"] = dict_of_dfs