
## Data
### Classes
- (initially) 44 classes: `data/classes_list.pt` 
    ['Culture.Arts', 'Culture.Broadcasting',
    'Culture.Crafts and hobbies', 'Culture.Entertainment',
    'Culture.Food and drink', 'Culture.Games and toys',
    'Culture.Internet culture', 'Culture.Language and literature',
    'Culture.Media', 'Culture.Music', 'Culture.Performing arts',
    'Culture.Philosophy and religion', 'Culture.Plastic arts',
    'Culture.Sports', 'Culture.Visual arts', 'Geography.Africa',
    'Geography.Americas', 'Geography.Antarctica', 'Geography.Asia',
    'Geography.Bodies of water', 'Geography.Europe',
    'Geography.Landforms', 'Geography.Maps', 'Geography.Oceania',
    'Geography.Parks', 'History_And_Society.Business and economics',
    'History_And_Society.Education',
    'History_And_Society.History and society',
    'History_And_Society.Military and warfare',
    'History_And_Society.Politics and government',
    'History_And_Society.Transportation', 'STEM.Biology',
    'STEM.Chemistry', 'STEM.Engineering', 'STEM.Geosciences',
    'STEM.Information science', 'STEM.Mathematics', 'STEM.Medicine',
    'STEM.Meteorology', 'STEM.Physics', 'STEM.Science', 'STEM.Space',
    'STEM.Technology', 'STEM.Time'] 
- (for aligned en-ru-hi articles) 45 classes: 
    initial list + ['Culture.People', 'Culture.Architecture'] - ['Culture.Plastic arts']

### Embeddings
- `embeddings/wiki.en.align.vec`
- `embeddings/wiki.ru.align.vec`
- `embeddings/wiki.hi.align.vec`

### Wikitext
#### Source file
- (initial) articles in English: `data/wikitext_tokenized.p` (Peeyush preprocessed the json.)
- articles in Russian: `data/wikitext_ru_sample.json`
- Aligned en-ru-hi articles: 33711 articles for each language: 
    `data/aligned_datasets/wikitext_topics_en_filtered.json`, `data/aligned_datasets/wikitext_topics_ru_filtered.json`, `data/aligned_datasets/wikitext_topics_hi_filtered.json`

## Models
ADD model files to Google Drive. Currently they are on prince.

### EN
- best 1-layer: `models/en_optimizer_SWA_num_hidden_1_dim_hidden_150_dropout_rate_0_learning_rate_0.01_num_epochs_10.pth`
- best 2-layer: `models/en_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10.pth`

### RU
- frozen (ru embeddings + frozen layer_out from en): `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10_frozen.pth`,
- finetuned: `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10_init_pretrained.pth`,   
- trained from scratch: `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10.pth`,


## TODO
- load model from state dict: load options dict, so not to difine it all the time.
- standardize code for data preprocessing for different languages.
- ADD model files to Google Drive. Currently they are on prince.