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
from tqdm import tqdm_notebook as tqdm

import spacy

###
# All language-specific constants defined here.
###
def _get_tokenizer_obj(language):
    if language == "english":
        from spacy.lang.en import English as Tokenizer
    elif language == "russian":
        from spacy.lang.ru import Russian as Tokenizer
    else:
        raise NotImplementedError

    tokenizer = Tokenizer()
    return tokenizer

_digits_to_words_dict = {
    "english" : {
        '0': ' zero',
        '1': ' one',
        '2': ' two',
        '3': ' three',
        '4': ' four',
        '5': ' five',
        '6': ' six',
        '7': ' seven',
        '8': ' eight',
        '9': ' nine',
    },
    "russian" : {
        '0': ' ноль',
        '1': ' один',
        '2': ' два',
        '3': ' три',
        '4': ' четыре',
        '5': ' пять',
        '6': ' шесть',
        '7': ' семь',
        '8': ' восемь',
        '9': ' девять',
    },
}

_common_forbidden_patterns =  [
    "{{.*}}"
    ,"&amp;"
    ,"&lt;"
    ,"&gt;"
    ,r"<ref[^<]*<\/ref>"
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
    ," +"
]

_forbidden_patterns_dict = {
    "english" : 
    _common_forbidden_patterns \
    + [
        "\[\[category:.*?\]\]", # EDITED remove categories [[Category:Far-left politics]]
        r"[^a-zA-Z0-9 ]",
        r"\b[a-zA-Z]\b",
    ],       
    "russian" :
    _common_forbidden_patterns \
    + [
        "\[\[категория:.*?\]\]", # EDITED: remove category for Russian]
        r"[^а-яА-Я0-9 ]",
        r"\b[а-яА-Я]\b",
    ],
}
 
###
# Parser.
###
  
class Parser:
    def __init__(self, language):
        """
        Parameters:
        -----------
        language : str, "english" or "russian"
        """
        self.LANGUAGE = language
        ## SAME for all languages
        # Load punctuations
        self.PUNCTUATIONS = set(string.punctuation)

        ## SPECIFIC to language
        # Patterns for regex
        self.PATTERNS = _forbidden_patterns_dict[language]
        # Digits to words
        self.DIGITS_TO_WORDS_DICT = _digits_to_words_dict[language]
        # Load tokenizer
        self.TOKENIZER = _get_tokenizer_obj(language)
        # Download and set stop word list from NLTK
        nltk.download('stopwords')
        self.STOP_WORDS = set(stopwords.words(language))
        self.STOP_WORDS.update({"", " "})
        
    def _tokenize(self, sent):
        """
        Lowercase and remove punctuation.

        Uses self.PUNCTUATIONS, self.TOKENIZER
        """
        tokens = self.TOKENIZER(sent)
        return [token.text for token in tokens if (token.text not in self.PUNCTUATIONS)]

    def _clean_patterns(self, text):
        """ 
        Clean text using regex - similar to what is used in FastText paper.

        Uses self.PATTERNS
        """
        for pattern in self.PATTERNS:
            cleanr = re.compile(pattern)
            text = re.sub(cleanr, ' ', text)
        return text

    def _substitute_digits_with_words(self, text):
        """ 
        Convert digits to their names. 
        
        Uses self.DIGITS_TO_WORDS_DICT
        """
        chars = text.strip()
        new_sentence = [self.DIGITS_TO_WORDS_DICT.get(char, char) for char in chars]
        return ''.join(new_sentence)

    def _remove_stop_words(self, tokens):
        """
        Removes stop words.

        Uses self.STOP_WORDS
        """
        return [token for token in tokens if not token in self.STOP_WORDS]

    def _preprocess_pipeline(self, wikitext) -> str:
        """ Combines all text transformations in a pipeline. """
        wikitext = str(wikitext).lower()
        wikitext = self._clean_patterns(wikitext)
        wikitext = self._substitute_digits_with_words(wikitext)
        wikitext = self._tokenize(wikitext)
        tokens   = self._remove_stop_words(wikitext)
        return tokens

    def get_wiki_tokenized_dataset(self, fname, *,
             extract_section=False, extract_outlinks=False,
             debug=False):
        """
        Get the tokenized dataframe containing - QID, Word Tokens & Categories.
        
        Parameters:
        ----------
        extract_section : bool (default: False)
            If True, section titles are also extracted and tokenized.
        extract_outlinks : bool (default: False)
            If True, raw outlinks (names of cited articles) are also extracted.
        """
        wiki_list_of_dicts = []
        with open(fname) as file:
            for line_idx, line in enumerate(tqdm(file)):
                wiki_row = {}
                line = json.loads(line.strip())
                wikitext = mwparserfromhell.parse(line['wikitext'])
                wiki_row['QID'] = line['qid'] # EDITED
                wiki_row['mid_level_categories'] = line['mid_level_categories']
                wiki_row['tokens'] = self._preprocess_pipeline(wikitext)

                if extract_section:
                    sections = wikitext.filter_headings()  
                    wiki_row['sections_tokens'] = self._preprocess_pipeline(sections)
                if extract_outlinks:
                    raise NotImplementedError
                    outlinks = 0

                wiki_list_of_dicts.append(wiki_row)
                if debug and line_idx > 4:
                    break

        wiki_df = pd.DataFrame(wiki_list_of_dicts)
        return wiki_df