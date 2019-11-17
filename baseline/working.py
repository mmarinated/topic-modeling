_number_to_word_dict = {
    "english" : {1 : "one", ... }
    "russian" : ...
}

_forbidden_patterns_dict = {
    "eng" : ...
    "rus" ...
}

def __init__(language):
    global LANGUAGE, tokenizer, punctuations, STOP_WORDS
    

    
    LANGUAGE = language
    self.FORBIDDEN_PATTERNS = _forbidden_patterns_dict[LANGUAGE]
    self.AJSDK = _ASDJKL_dict[LANGUAGE]

    if LANGUAGE == "english":
        from spacy.lang.en import English as Language
    else:
        from spacy.lang.ru import Russian as Language

        # class A:
#     def __init__(self, lang):
#         if lang ...
        
class BaseParser:
    def __init__(self):
        # create global consts all languages have to share
        # e.g. common_patterns
        
    # all methods defined
    # NOTE:
    # they can use self.tokenizer defined in a child class
        
class EnglishParser(BaseParser):
    def __init__(self):
        self.super().__init__()
        
        self.tokenizer = EnglishTokenizer()
        
      