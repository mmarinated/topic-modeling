_parent_path = "/scratch/mz2476/wiki"

PATH_TO_HINDI_STOPWORDS = f"{_parent_path}/topic-modeling/baseline/data_creation/hindi_stopwords.txt"

PATH_TO_EMBEDDINGS_FOLDER = f"{_parent_path}/embeddings/"
PATH_TO_DATA_FOLDER = f"{_parent_path}/data/202001_dumps/"
PATH_TO_MODELS_FOLDER = f"{_parent_path}/models/"
PATH_TO_TENSORBOARD_RUNS = f"{PATH_TO_MODELS_FOLDER}/runs/"

# PATH_TO_SAVED_EMBED_FOLDER = f"{_parent_path}/data/aligned_datasets/mix_en_hi_ru/"
# PATH_TO_DATA_FOR_MODEL_FOLDER = f"{_parent_path}/data/aligned_datasets/data_for_model/"

# PATH_TO_SAVED_ALIGNED_DATA_FOLDER = "/scratch/mz2476/wiki/data/aligned_datasets/data_for_model" # ADD it to paths

def get_paths(language):
    """
    Returns FILE_NAMES_DICT -- dict, where paths to 
    - wiki_df
    - vocab
    - monolingual_train
    - multilingual_train
    - val
    - test
    - fasttext_embeddings
    - embed_matrix (embed_matrix and idx_to_word)
    are stored for given language.
    """
    language_code = language[:2]
    FILE_NAMES_DICT = {
        "wiki_df"   : PATH_TO_DATA_FOLDER + f"{language_code}_df_full.pkl",
        "vocab"     : PATH_TO_DATA_FOLDER + f"{language_code}_vocab_all.pt",
        "monolingual_train"     : PATH_TO_DATA_FOLDER + f"{language_code}_df_wiki_monolingual_train.pt",
        "multilingual_train"    : PATH_TO_DATA_FOLDER + f"{language_code}_df_wiki_multilingual_train.pt",
        "val"       : PATH_TO_DATA_FOLDER + f"{language_code}_df_wiki_valid.pt",
        "test"      : PATH_TO_DATA_FOLDER + f"{language_code}_df_wiki_test.pt",
        "fasttext_embeddings": f"{PATH_TO_EMBEDDINGS_FOLDER}wiki.{language_code}.align.vec",
        "embed_matrix": f'{PATH_TO_DATA_FOLDER}{language_code}_embeddings_matrix_with_idx_to_word.pt',
    }
    
    return FILE_NAMES_DICT