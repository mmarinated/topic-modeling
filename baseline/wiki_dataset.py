
from collections import defaultdict

import numpy
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer

import preprocess
import utils
from MY_PATHS import *
from preprocess import (TensoredDataset, create_dict_of_tensor_datasets,
                        create_lookups_for_vocab, create_vocab_from_tokens,
                        pad_collate_fn, tokenize_dataset)

SEED = 57


# these values cannot be changed
monolingual_train_size = 30000
multilingual_train_size = 10000
val_size = 1000


def get_mixed_datasets(LANGUAGES_LIST=("english", "russian", "hindi"), SAVE=False, LOAD=True):
    """
    @returns
        index_to_word, word_to_index, dict_wiki_tensor_dataset, weights_matrix_ve, classes
    """
    LANGUAGES_DICT = defaultdict(dict)

    # assuming the data is in PATH_TO_DATA_FOLDER
    for language in LANGUAGES_LIST:
        language_code = language[:2]
        LANGUAGES_DICT[language]["language_code"] = language_code
        FILE_NAMES_DICT = {
            "vocab": f"{PATH_TO_DATA_FOR_MODEL_FOLDER}vocab_all_{language_code}.pt",
            "monolingual_train": f"{PATH_TO_DATA_FOR_MODEL_FOLDER}df_wiki_monolingual_train_{monolingual_train_size}_{language_code}.pt",
            "multilingual_train": f"{PATH_TO_DATA_FOR_MODEL_FOLDER}df_wiki_multilingual_train_{multilingual_train_size}_{language_code}.pt",
            "val": f"{PATH_TO_DATA_FOR_MODEL_FOLDER}df_wiki_valid_{val_size}_{language_code}.pt",
            "test": f"{PATH_TO_DATA_FOR_MODEL_FOLDER}df_wiki_test_{language_code}.pt",
            "fasttext_embeddings": f"{PATH_TO_EMBEDDINGS_FOLDER}wiki.{language_code}.align.vec",
            "embed_matrix": f'{PATH_TO_SAVED_EMBED_FOLDER}embeddings_matrix_with_idx_to_word_{language_code}.pt',
        }
        # ADD check that these files exist
        LANGUAGES_DICT[language]["FILE_NAMES_DICT"] = FILE_NAMES_DICT

    # LOAD vocab, tensor dataset, classes
    classes = torch.load(PATH_TO_DATA_FOLDER + "45_classes_list.pt")
    mlb = MultiLabelBinarizer(classes)

    for language, lang_dict in LANGUAGES_DICT.items():
        vocab = torch.load(lang_dict["FILE_NAMES_DICT"]["vocab"])
        print(f"{language} vocab size is:", len(vocab))
    #     LANGUAGES_DICT[language]["vocab"] = vocab
        LANGUAGES_DICT[language]["index_to_word"], LANGUAGES_DICT[language]["word_to_index"] =\
            create_lookups_for_vocab(vocab)

    # Create combined vocab, index_to_word, word_to_index
    # 0 - <pad>, 1 - <unk> 
    vocab = ["<pad>", "<unk>"]
    print("Order:", LANGUAGES_DICT.keys())
    for language, lang_dict in LANGUAGES_DICT.items(): # .keys() keep same order in Python version >= 3.7
        assert lang_dict["index_to_word"][0] != "<pad>"
        vocab += lang_dict["index_to_word"]
        
    index_to_word, word_to_index = create_lookups_for_vocab(vocab)
    assert len(set(word_to_index)) == len(word_to_index)

    wiki_train, wiki_valid = [], []

    dict_of_dfs = defaultdict()

    for language, lang_dict in LANGUAGES_DICT.items():
        language_code = lang_dict["language_code"]
        dict_of_dfs[f"monolingual_train_{language_code}"], dict_of_dfs[f"multilingual_train_{language_code}"] =\
                (torch.load(lang_dict["FILE_NAMES_DICT"]["monolingual_train"]),
                torch.load(lang_dict["FILE_NAMES_DICT"]["multilingual_train"]))
        dict_of_dfs[f"val_{language_code}"] = torch.load(lang_dict["FILE_NAMES_DICT"]["val"])
        wiki_train.append(dict_of_dfs[f"multilingual_train_{language_code}"])
        wiki_valid.append(dict_of_dfs[f"val_{language_code}"])

    wiki_train = pd.concat(wiki_train).sample(frac=1, random_state=SEED).reset_index(drop=True)
    wiki_valid = pd.concat(wiki_valid).sample(frac=1, random_state=SEED).reset_index(drop=True)

    dict_of_dfs["train"] = wiki_train
    dict_of_dfs["val"] = wiki_valid

    print(f"Combined train size: {wiki_train.shape[0]} \nCombined val size: {wiki_valid.shape[0]}")
    # wiki_train.head()

 
    dict_wiki_tensor_dataset = create_dict_of_tensor_datasets(dict_of_dfs, word_to_index, max_num_tokens=None)

    for language, lang_dict in LANGUAGES_DICT.items():
        if LOAD:
            embed_info_dict = torch.load(lang_dict["FILE_NAMES_DICT"]["embed_matrix"])
            LANGUAGES_DICT[language]["weights_matrix_ve"] = embed_info_dict["weights_matrix_ve"]
        if SAVE:
            language_code = lang_dict["language_code"]
            # 2.5 million
            embeddings = utils.load_vectors(lang_dict["FILE_NAMES_DICT"]["fasttext_embeddings"])
            #Creating the weight matrix for pretrained word embeddings
            weights_matrix_ve = utils.create_embeddings_matrix(lang_dict["index_to_word"], embeddings)
            LANGUAGES_DICT[language]["weights_matrix_ve"] = weights_matrix_ve
            # SAVE embeddings matrix together with index_to_word
            torch.save({
                "index_to_word" : lang_dict["index_to_word"],
                "weights_matrix_ve" : weights_matrix_ve,
            }, lang_dict["FILE_NAMES_DICT"]["embed_matrix"])
            print("Saved.") 

    #Creating the weight matrix for pretrained word embeddings
    # 0 - <pad>, 1 - <unk> 
    weights_matrix_ve = torch.zeros(len(index_to_word), LANGUAGES_DICT["english"]["weights_matrix_ve"].shape[1])
    start_idx = 2
    for language, lang_dict in LANGUAGES_DICT.items():
        end_idx = start_idx + len(lang_dict["index_to_word"])
        assert index_to_word[start_idx:end_idx] == lang_dict["index_to_word"]
        assert index_to_word[start_idx] == lang_dict["index_to_word"][0]
        assert index_to_word[end_idx-1] == lang_dict["index_to_word"][-1]
        weights_matrix_ve[start_idx:end_idx] = lang_dict["weights_matrix_ve"]
        start_idx = end_idx

    print(f"Embeddings matrix shape: {weights_matrix_ve.shape}, \nVocab size: {len(vocab)}")

    return index_to_word, word_to_index, dict_wiki_tensor_dataset, weights_matrix_ve, classes


# class WikiData:
#     def __init__(self, languages_list, FILE_NAMES_DICT): # ADD specify FILE_NAMES_DICT in txt file
#     self.FILE_NAMES_DICT = FILE_NAMES_DICT
#     self.vocab = 


# class MultilingualWikiData:
#     def __init__(self, languages_list, FILE_NAMES_DICT):


# class MonolingualWikiData:
#     def __init__(self, language, FILE_NAMES_DICT):
