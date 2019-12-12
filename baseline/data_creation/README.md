This module is to run once to create data.

To create df with cleaned tokenized texts, sections, links from raw `.json` run:
```python
import wiki_parser

parser = wiki_parser.Parser("english")
wiki_df = parser.get_wiki_tokenized_dataset(
    path_to_json_file,
    extract_title=True, extract_tokens=True, extract_categories=True,
    extract_section=False, extract_outlinks=False,
)
```

To create the train/val/test splits df for experiments for aligned articles in English, Russian, Hindi, run `run_aligned_en_ru_hi_df_preprocess.py`.

To create `index_to_word, word_to_index, dict_wiki_tensor_dataset, weights_matrix_ve, classes` for aligned articles in English, Russian, Hindi,
run `wiki_dataset.get_mixed_datasets()`.