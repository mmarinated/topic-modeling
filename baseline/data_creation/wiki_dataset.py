from collections import defaultdict

import numpy
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from baseline.MY_PATHS import *
from baseline.data_creation.preprocess import (TensoredDataset, create_dict_of_tensor_datasets,
                        create_lookups_for_vocab, create_vocab_from_tokens,
                        pad_collate_fn, tokenize_dataset,
                        load_vectors, create_embeddings_matrix)
from baseline.utils import get_classes_list

SEED = 57

def get_mixed_datasets(LANGUAGES_LIST=("english", "russian", "hindi"), SAVE=False, LOAD=True):
    """
    @returns
        index_to_word, word_to_index, dict_wiki_tensor_dataset, weights_matrix_ve, classes
    """
    LANGUAGES_DICT = defaultdict(dict)

    # assuming the data is in PATH_TO_DATA_FOLDER
    for language in LANGUAGES_LIST:
        LANGUAGES_DICT[language]["FILE_NAMES_DICT"] = get_paths(language)

    # LOAD vocab, tensor dataset, classes
    classes = get_classes_list()
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
        language_code = language[:2]
        dict_of_dfs[f"monolingual_train_{language_code}"], dict_of_dfs[f"multilingual_train_{language_code}"] =\
                (torch.load(lang_dict["FILE_NAMES_DICT"]["monolingual_train"]),
                torch.load(lang_dict["FILE_NAMES_DICT"]["multilingual_train"]))
        dict_of_dfs[f"val_{language_code}"] = torch.load(lang_dict["FILE_NAMES_DICT"]["val"])
        wiki_train.append(dict_of_dfs[f"multilingual_train_{language_code}"])
        wiki_valid.append(dict_of_dfs[f"val_{language_code}"])

    wiki_train = pd.concat(wiki_train).sample(frac=1, random_state=SEED).reset_index(drop=True)
    wiki_valid = pd.concat(wiki_valid).sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Add bilingual datasets
    wiki_train_en_ru = pd.concat([
        dict_of_dfs[f"multilingual_train_en"], dict_of_dfs[f"multilingual_train_ru"],
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    wiki_train_en_hi = pd.concat([
        dict_of_dfs[f"multilingual_train_en"], dict_of_dfs[f"multilingual_train_hi"],
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    wiki_train_ru_hi = pd.concat([
        dict_of_dfs[f"multilingual_train_ru"], dict_of_dfs[f"multilingual_train_hi"],
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)


    dict_of_dfs["train_en_ru"] = wiki_train_en_ru
    dict_of_dfs["train_en_hi"] = wiki_train_en_hi
    dict_of_dfs["train_ru_hi"] = wiki_train_ru_hi
    dict_of_dfs["train"] = wiki_train
    dict_of_dfs["val"] = wiki_valid

    print(f"Combined train size: {wiki_train.shape[0]} \nCombined val size: {wiki_valid.shape[0]}")
    # wiki_train.head()

 
    dict_wiki_tensor_dataset = create_dict_of_tensor_datasets(dict_of_dfs, word_to_index, max_num_tokens=None)

    for language, lang_dict in LANGUAGES_DICT.items():
        if SAVE:
            # 2.5 million
            embeddings = load_vectors(lang_dict["FILE_NAMES_DICT"]["fasttext_embeddings"])
            #Creating the weight matrix for pretrained word embeddings
            weights_matrix_ve = create_embeddings_matrix(lang_dict["index_to_word"], embeddings)
            LANGUAGES_DICT[language]["weights_matrix_ve"] = weights_matrix_ve
            # SAVE embeddings matrix together with index_to_word
            torch.save({
                "index_to_word" : lang_dict["index_to_word"],
                "weights_matrix_ve" : weights_matrix_ve,
            }, lang_dict["FILE_NAMES_DICT"]["embed_matrix"])
            print("Saved", lang_dict["FILE_NAMES_DICT"]["embed_matrix"])
        if LOAD:
            embed_info_dict = torch.load(lang_dict["FILE_NAMES_DICT"]["embed_matrix"])
            LANGUAGES_DICT[language]["weights_matrix_ve"] = embed_info_dict["weights_matrix_ve"]

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
