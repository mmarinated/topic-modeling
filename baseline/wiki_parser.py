"""
Parser class -- for parsing json file with Wikipedia articles to get
- tokens of text of the article,
- tokens of sections of the article,
- list of outlinks.

Main function is 
    - get_wiki_tokenized_dataset.
"""
import json
import re
import string

import mwparserfromhell
import nltk
import pandas as pd
from nltk.corpus import stopwords
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
    elif language == "hindi":
        from spacy.lang.hi import Hindi as Tokenizer
    else:
        raise NotImplementedError

    tokenizer = Tokenizer()
    return tokenizer

def _get_stopwords(language):
    try:
        # Download and set stop word list from NLTK
        nltk.download('stopwords')
        stop_words = set(stopwords.words(language))
    except OSError:
        # Load words from file
        if language == "hindi":
            file = open('hindi_stopwords.txt')
            text = file.read()
            stop_words = set([word for word in text.split("\n") if word != ""])
        else:
            raise NotImplementedError
    return stop_words

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
    "hindi" : {
        '0': ' शून्य',
        '1': ' एक',
        '2': ' दो',
        '3': ' तीन',
        '4': ' चार',
        '5': ' पांच',
        '6': ' छह',
        '7': ' सात',
        '8': ' आठ',
        '9': ' नौ',
        '०': ' शून्य',
        '१': ' एक',
        '२': ' दो',
        '३': ' तीन',
        '४': ' चार',
        '५': ' पांच',
        '६': ' छह',
        '७': ' सात',
        '८': ' आठ',
        '९': ' नौ',
    },
}

_common_forbidden_patterns =  [
    "\[\[category:.*?\]\]", # EDITED remove categories [[Category:Far-left politics]]
    "\[\[категория:.*?\]\]", # EDITED: remove category for Russian
    "\[\[श्रेणी:.*?\]\]", # EDITED: remove category for Hindi
    "{{.*}}" # put a star?
    ,"&amp;"
    ,"&lt;"
    ,"&gt;"
    ,r"<ref[^<]*<\/ref>"
    ,"<[^>]*>"
    ,"\|left"
    ,"\|\d+px"
#     ,"\[\[category:"
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

# NOTE: category pattern should be added in common pattern
# because it should be removed before other patterns.
_forbidden_patterns_dict = {
    "english": [
        r"[^a-zA-Z0-9 ]",
        r"\b[a-zA-Z]\b",
    ],
    "russian": [
        r"[^а-яА-Я0-9 ]",
        r"\b[а-яА-Я]\b",
    ],
    "hindi": [
        r'[^\u0900-\u097F0-9 ]',  # remove all special symbols
        r'।',  # remove full stop
    ]
}

###
# Parser.
###
  
class Parser:
    def __init__(self, language):
        """
        Parameters:
        -----------
        language : str, "english" or "russian" or "hindi"
        """
        self.LANGUAGE = language
        ## SAME for all languages
        # Load punctuations
        self.PUNCTUATIONS = set(string.punctuation)

        ## SPECIFIC to language
        # Patterns for regex
        self.PATTERNS = _common_forbidden_patterns + _forbidden_patterns_dict[language]
        # Digits to words
        self.DIGITS_TO_WORDS_DICT = _digits_to_words_dict[language]
        # Load tokenizer
        self.TOKENIZER = _get_tokenizer_obj(language)
        # Set stop word list
        self.STOP_WORDS = _get_stopwords(language)
    
    ###
    # Parse wikitext.
    ###
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

        Example:
        >>> parser = Parser("english")
        >>> parser._substitute_digits_with_words("1 boy, 23 girls")
        " one boy,  two three girls"
        """
        chars = text.strip()
        new_sentence = [self.DIGITS_TO_WORDS_DICT.get(char, char) for char in chars]
        return ''.join(new_sentence)
    
    def _tokenize(self, sent):
        """
        Lowercase and remove punctuation.

        Uses self.PUNCTUATIONS, self.TOKENIZER
        """
        tokens = self.TOKENIZER(sent)
        return [token.text for token in tokens if (token.text not in self.PUNCTUATIONS)]
    
    def _remove_empty_tokens(self, tokens):
        """ Removes empty tokens that consist of several spaces. """
        return [token for token in tokens if not token.strip() == '']

    def _remove_stop_words(self, tokens):
        """
        Removes stop words.

        Uses self.STOP_WORDS
        """
        return [token for token in tokens if not token in self.STOP_WORDS]

    def _preprocess_pipeline(self, wikitext) -> str:
        """ Combines all text transformations in a pipeline and returns list of tokens. """
        wikitext = str(wikitext).lower()
        wikitext = self._clean_patterns(wikitext)
        wikitext = self._substitute_digits_with_words(wikitext)
        wikitext = self._tokenize(wikitext)
        wikitext = self._remove_empty_tokens(wikitext)
        wikitext = self._remove_stop_words(wikitext)
        return wikitext

    ###
    # Parse wiki outlinks.
    ###

    def _links_to_str(self, raw_links):
        return [str(raw_link) for raw_link in raw_links]

    def _clean_links(self, raw_links):
        links = []
        for raw_link in raw_links:
            link = raw_link[2:-2].split("|")[0]
            links.append(link)
        return links

    def get_wiki_tokenized_dataset(
            self, fname, *,
            extract_title=True, extract_tokens=True, extract_categories=True,
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
                
                if extract_title:
                    wiki_row["title"] = line["entitle"]
                if extract_tokens:
                    wiki_row['tokens'] = self._preprocess_pipeline(wikitext)
                if extract_categories:
                    wiki_row['mid_level_categories'] = line['mid_level_categories']
                if extract_section:
                    sections = wikitext.filter_headings()  
                    wiki_row['sections_tokens'] = self._preprocess_pipeline(sections)
                if extract_outlinks:
                    outlinks = wikitext.filter_wikilinks()
                    outlinks = self._links_to_str(outlinks)
                    wiki_row['raw_outlinks'] = outlinks
                    wiki_row['outlinks'] = self._clean_links(outlinks)

                wiki_list_of_dicts.append(wiki_row)
                if debug and line_idx > 4:
                    break

        wiki_df = pd.DataFrame(wiki_list_of_dicts)
        return wiki_df

