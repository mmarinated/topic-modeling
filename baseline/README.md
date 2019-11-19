
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
- Aligned en-ru-hi articles: 33K articles for each language: (Actually: en 33823, ru 33711)
    `data/aligned_datasets/wikitext_topics_en_filtered.json`, `data/aligned_datasets/wikitext_topics_ru_filtered.json`, `data/aligned_datasets/wikitext_topics_hi_filtered.json`

## Models
ADD model files to Google Drive. Currently they are on prince.
[Link to Google Drive.](https://drive.google.com/drive/u/2/folders/1_fhco5kWR8uAZ7pnlzAFLkzeYEwV_QYX)

### EN
- best 1-layer: `models/en_optimizer_SWA_num_hidden_1_dim_hidden_150_dropout_rate_0_learning_rate_0.01_num_epochs_10.pth`
- best 2-layer: `models/en_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10.pth`

### RU
- frozen (ru embeddings + frozen layer_out from en): `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10_frozen.pth`,
- finetuned: `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10_init_pretrained.pth`,   
- trained from scratch: `models/ru_optimizer_SWA_num_hidden_2_dim_hidden_150_dropout_rate_0.2_learning_rate_0.01_num_epochs_10.pth`,

## Results
- Multilingual

|idx|experiment                   |precision_macro|recall_macro                                 |f1_macro|precision_micro|recall_micro|f1_micro|
|------|-----------------------------|---------------|---------------------------------------------|--------|---------------|------------|--------|
|0     |Train on 20K EN articles, validate on 2K.|0.5811         |0.4114                                       |0.4622  |0.8467         |0.705       |0.7694  |
|1     |Train on 10K EN articles and 10K RU articles, validate on 1K RU and 1000 EN.|0.6146         |0.4238                                       |0.4815  |0.8418         |0.7318      |0.783   |
|2     |Train on 10K EN articles and 10K RU articles, validate on 1K EN.|0.5893         |0.4714                                       |0.5078  |0.8343         |0.7579      |0.7942  |
|3     |Train on 10K EN articles and 10K RU articles, validate on 1K RU.|0.5631         |0.3612                                       |0.4168  |0.8493         |0.7082      |0.7724  |


## TODO
- load model from state dict: load options dict, so not to define it all the time.
- standardize code for data preprocessing for different languages. (might want to specify len of train, val. Divide to train,val, save test.)
- clean code for models.
- ADD model files to Google Drive. Currently they are on prince.