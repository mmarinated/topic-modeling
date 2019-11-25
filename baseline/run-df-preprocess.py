import pickle as pkl
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from preprocess import (create_dict_of_tensor_datasets,
                        create_lookups_for_vocab, create_vocab_from_tokens,
                        remove_non_common_articles_and_sort_by_QID,
                        remove_rows_with_empty_column)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH_TO_DATA_FOLDER = "/scratch/mz2476/wiki/data/aligned_datasets/"

# Load list of classes
classes_list = torch.load(PATH_TO_DATA_FOLDER + '45_classes_list.pt')

SAVE = False
DEBUG = True
LOAD = True

test_size = 0.1
train_size = 10000
val_size = 1000

SEED = 57

LANGUAGES_LIST = ["english", "russian", "hindi"]
LANGUAGES_DICT = defaultdict(dict)

for language in LANGUAGES_LIST:
    language_code = language[:2]
    FILE_NAMES_DICT = {
        "json": f"wikitext_topics_{language_code}_filtered.json",
        "wiki_df": f"wikitext_tokenized_text_sections_outlinks_{language_code}.p",
        "vocab": f"vocab_all_{language_code}.pt",
        "train": f"df_wiki_train_{train_size}_{language_code}.pt",
        "val": f"df_wiki_valid_{val_size}_{language_code}.pt",
        "test": f"df_wiki_test_{test_size}_{language_code}.pt",
#         "tensor_dataset": f"wiki_tensor_dataset_{language_code}.pt",
    }
    LANGUAGES_DICT[language]["FILE_NAMES_DICT"] = FILE_NAMES_DICT
    print(FILE_NAMES_DICT)

# Load wiki_df and remove rows with empty labels/tokens
for language in LANGUAGES_DICT.keys():
    wiki_df = pkl.load(open(PATH_TO_DATA_FOLDER + LANGUAGES_DICT[language]["FILE_NAMES_DICT"]["wiki_df"], "rb"))
    LANGUAGES_DICT[language]["wiki_df"] = wiki_df

    remove_rows_with_empty_column(LANGUAGES_DICT[language]["wiki_df"], column="mid_level_categories")
    remove_rows_with_empty_column(LANGUAGES_DICT[language]["wiki_df"], column="tokens")

remove_non_common_articles_and_sort_by_QID(LANGUAGES_DICT)

# Binarize labels, create vocabulary
for cur_dict in LANGUAGES_DICT.values():
    mlb = MultiLabelBinarizer(classes_list)
    cur_dict["wiki_df"]["labels"] =\
        list(mlb.fit_transform(cur_dict["wiki_df"].mid_level_categories))
    assert (mlb.classes_ == classes_list).all()

    if LOAD:
        vocab = torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["vocab"])

    if SAVE:
        vocab = create_vocab_from_tokens(cur_dict["wiki_df"]["tokens"])
        torch.save(vocab, PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["vocab"])
        print("Saved: ", cur_dict["FILE_NAMES_DICT"]["vocab"])

    index_to_word, word_to_index = create_lookups_for_vocab(vocab)
    cur_dict["index_to_word"], cur_dict["word_to_index"] = index_to_word, word_to_index

# train/val/test split by QID
QIDs = LANGUAGES_DICT["english"]["wiki_df"].QID
rest_QIDs, test_QIDs = train_test_split(QIDs, test_size=test_size, random_state=SEED)
train_QIDs, val_QIDs = train_test_split(QIDs, train_size=train_size, test_size=val_size, random_state=SEED)

for cur_dict in LANGUAGES_DICT.values():
    dict_of_dfs = defaultdict()

    if LOAD:
        dict_of_dfs["train"], dict_of_dfs["val"], dict_of_dfs["test"] =\
            (torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["train"]),
             torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["val"]),
             torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["test"]))

    if SAVE:
        dict_of_dfs["train"], dict_of_dfs["val"], dict_of_dfs["test"] =\
            (cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(train_QIDs)],
             cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(val_QIDs)],
             cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(test_QIDs)])

        torch.save(dict_of_dfs["train"], PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["train"])
        torch.save(dict_of_dfs["val"], PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["val"])
        torch.save(dict_of_dfs["test"], PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["test"])
        print("Saved:, ", cur_dict["FILE_NAMES_DICT"]["train"], cur_dict["FILE_NAMES_DICT"]["val"], cur_dict["FILE_NAMES_DICT"]["test"])
    
    cur_dict["dict_of_dfs"] = dict_of_dfs

# # Tokenized datasets
# for cur_dict in LANGUAGES_DICT.values():
#     create_dict_of_tensor_datasets(dict_of_dfs, word_to_index, max_num_tokens=None)
# ADD
