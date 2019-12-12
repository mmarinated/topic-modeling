import pickle as pkl
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from ..preprocess import (create_lookups_for_vocab, create_vocab_from_tokens,
                          remove_non_common_articles_and_sort_by_QID,
                          remove_rows_with_empty_column)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH_TO_DATA_FOLDER = "/scratch/mz2476/wiki/data/aligned_datasets/"
PATH_TO_SAVE_FOLDER = "/scratch/mz2476/wiki/data/aligned_datasets/data_for_model" # ADD it to paths

# Load list of classes
classes_list = torch.load(PATH_TO_DATA_FOLDER + '45_classes_list.pt')

SAVE = True
DEBUG = True
LOAD = False

monolingual_train_size = 30000
multilingual_train_size = 10000
val_size = 1000

SEED = 57

LANGUAGES_LIST = ["english", "russian", "hindi"]
LANGUAGES_DICT = defaultdict(dict)

for language in LANGUAGES_LIST:
    language_code = language[:2]
    FILE_NAMES_DICT = {
        "json"      : f"wikitext_topics_{language_code}_filtered.json",
        "wiki_df"   : f"wikitext_tokenized_text_sections_outlinks_{language_code}.p",
        "vocab"     : f"data_for_model/vocab_all_{language_code}.pt",
        "monolingual_train"     : f"data_for_model/df_wiki_monolingual_train_{monolingual_train_size}_{language_code}.pt",
        "multilingual_train"    : f"data_for_model/df_wiki_multilingual_train_{multilingual_train_size}_{language_code}.pt",
        "val"       : f"data_for_model/df_wiki_valid_{val_size}_{language_code}.pt",
        "test"      : f"data_for_model/df_wiki_test_{language_code}.pt",
#         "tensor_dataset": f"wiki_tensor_dataset_{language_code}.pt",
    }
    LANGUAGES_DICT[language]["FILE_NAMES_DICT"] = FILE_NAMES_DICT
    print(language, "\n", FILE_NAMES_DICT)

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
monolingual_train_QIDs, val_and_test_QIDs = train_test_split(QIDs, train_size=monolingual_train_size, random_state=SEED)
multilingual_train_QIDs, _ = train_test_split(monolingual_train_QIDs, train_size=multilingual_train_size, random_state=SEED)
val_QIDs, test_QIDs = train_test_split(val_and_test_QIDs, train_size=val_size, random_state=SEED)
test_size = len(test_QIDs)
print(f"monolingual_train size \t{len(monolingual_train_QIDs)} \n"
      f"multilingual_train size \t{len(multilingual_train_QIDs)} \n"
      f"val size \t{len(val_QIDs)} \n"
      f"test size \t{len(test_QIDs)} \n")

for cur_dict in LANGUAGES_DICT.values():
    dict_of_dfs = defaultdict()

    if LOAD:
        dict_of_dfs["monolingual_train"], dict_of_dfs["multilingual_train"], dict_of_dfs["val"], dict_of_dfs["test"] =\
            (torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["monolingual_train"]),
             torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["multilingual_train"]),
             torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["val"]),
             torch.load(PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"]["test"]))

    if SAVE:
        dict_of_dfs["monolingual_train"], dict_of_dfs["multilingual_train"], dict_of_dfs["val"], dict_of_dfs["test"] =\
            (cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(monolingual_train_QIDs)],
             cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(multilingual_train_QIDs)],
             cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(val_QIDs)],
             cur_dict["wiki_df"][cur_dict["wiki_df"].QID.isin(test_QIDs)])
        for name in dict_of_dfs.keys():
            torch.save(dict_of_dfs[name], PATH_TO_DATA_FOLDER + cur_dict["FILE_NAMES_DICT"][name])
            print("Saved:, ", cur_dict["FILE_NAMES_DICT"][name])
    
    cur_dict["dict_of_dfs"] = dict_of_dfs
