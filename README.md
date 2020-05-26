# topic-modeling
Inferring the topics of Wikipedia articles in different languages.   
Capstone project, Fall 2019.

## Research directions
- Improving the architecture of the model for English articles currently deployed.
    - bag-of-words models with fastText embeddings
    - LSTM, LSTM with self attention, LSTM with IDF self attention weights, transformer
-  Transferring the model to articles in other languages (Hindi, Russian).
    - using fastText multilingual word embeddings, we experiment on using model trained only on English articles vs model trained on several languages simulteneously.
- Exploring language agnostic models based on links between articles.
    - bag-of-words model
    - graph CNN model (GraphSAGE)

[Report](report/Capstone_Report_2019.pdf)  
[Poster](report/Capstone_Poster_2019.pdf) - top-3 best capstone poster among 36 teams.

![Poster](report/poster_image.png)

*Created by Marina Zavalina, Peeyush Jain, Sarthak Agarwal, Chinmay Singhal in Fall 2019.   
Advisors: Isaac Johnson (Wikimedia Foundation), Anastasios Noulas (NYU CDS).     
Project for DS-GA 1006, NYU Center for Data Science.*



<!-- 
1. Data

a) Pickle file for Wikitext (contains tokens)

https://drive.google.com/open?id=1bgkuTbN-eRlKLiPsbK8fbtqlFGD-iK1Z

b) Pickle file for Wikisections (contains tokens)

https://drive.google.com/open?id=1OWbzrvSpiibS5xEuhltB65nXMxZvnSRg



2. Pre-trained word embeddings (fastText)

https://drive.google.com/open?id=1vfoiWQkEjyNXRyi0JzA8Aq5Zzjfcpo2w -->
