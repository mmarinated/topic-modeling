# import dependencies
import io
import re
import nltk
import json
import gzip
import torch
import spacy
import string
import jsonlines
import pandas as pd
import pickle as pkl
import numpy as np
import mwparserfromhell
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm, tqdm_notebook

from spacy.lang.en import English
import spacy

# Load Russian tokenizer, tagger, parser, NER and word vectors
tokenizer = English()
punctuations = string.punctuation

# downloading and setting stop word list from NLTK
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# lowercase and remove punctuation
def tokenize(sent):
    tokens = tokenizer(sent)
    return [token.text.lower() for token in tokens if (token.text not in punctuations)]

# clean text using regex - similar to what is used in FastText paper
def clean(text):
    text = text.lower()
    patterns = [
        "\[\[category:.*?\]\]", # # EDITED remove categories [[Category:Far-left politics]]
        "{{.*}}"
        ,"&amp;"
        ,"&lt;"
        ,"&gt;"
        ,"<ref[^<]*<\/ref>"
        ,"<[^>]*>"
        ,"\|left"
        ,"\|\d+px"
        ,"\[\[category:"
        ,r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b"
        ,"\|thumb"
        ,"\|right"
        ,"\[\[image:[^\[\]]*"
        ,"\[\[category:([^|\]]*)[^]]*\]\]"
        ,"\[\[[a-z\-]*:[^\]]*\]\]"
        ,"\["
        ,"\]"
        ,"\{[^\}]*\}"
        ,r"\n"
#         ,r"[^а-яА-Я0-9 ]" # EDITED
#         ,r"\b[а-яА-Я]\b" # EDITED
        ,r"[^a-zA-Z0-9 ]"
        ,r"\b[a-zA-Z]\b"
        ," +"
    ]
    
    for pattern in patterns:
        cleanr = re.compile(pattern)
        text = re.sub(cleanr, ' ', text)
    return text

# covert numerals to their text equivalent
# EDITED
def subsitute(text):
    return text.strip().replace('0', ' zero') \
                        .replace('1',' one') \
                        .replace('2',' two') \
                        .replace('3',' three') \
                        .replace('4',' four') \
                        .replace('5',' five') \
                        .replace('6',' six') \
                        .replace('7',' seven') \
                        .replace('8',' eight') \
                        .replace('9',' nine')

# remove empty token generated from inserting blank spaces
def remove_empty_token(tokens):
    result = []
    for token in tokens:
        if not token.strip() == '':
            result.append(token)
    return result

# optional - remove other common stop words 
# get the stop words from NLTK package
def remove_stop_words(tokens):
    result = []
    for token in tokens:
        if not token in STOP_WORDS:
            result.append(token)
    return result

# optional - remove words less than 3 character long
def remove_short_words(tokens):
    result = []
    for token in tokens:
        if len(token) >= 3:
            result.append(token)
    return result

# get the tokenized dataframe containing - QID, Word Tokens & Categories
def get_wiki_tokenized_dataset(FILE_NAME, extract_section=False):
#     global wiki_dict
    wiki_dict = []
    i = 0
    with open(FILE_NAME) as file:
         for line in tqdm_notebook(file):
            i += 1
            wiki_row = {}
            line = json.loads(line.strip())
            wikitext = mwparserfromhell.parse(line['wikitext'])
            wiki_row['QID'] = line['qid'] # EDITED
            wiki_row['mid_level_categories'] = line['mid_level_categories']
            if extract_section:
                assert False, "not implemented"
                sections = wikitext.filter_headings()  
                wiki_row['tokens'] = tokenize(subsitute(clean(str(sections))))
            else:
                raw_tokens = tokenize(subsitute(clean(str(wikitext))))
                wiki_row['raw_tokens'] = raw_tokens
                wiki_row['tokens'] = remove_short_words(
                                     remove_stop_words(
                                     remove_empty_token(raw_tokens)))

            wiki_dict.append(wiki_row)
#             if i > 4:
#                 break

    wiki_df = pd.DataFrame(wiki_dict)
    return wiki_df