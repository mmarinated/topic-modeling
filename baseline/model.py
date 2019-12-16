from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# c - num classifiers
# b - batch
# l - length (padded)

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, dim_e, pretrained_embeddings=None):
        """
        @param vocab_size: size of the vocabulary. 
        @param dim_e: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        if pretrained_embeddings is None:
            # pay attention to padding_idx
            self.embed_e = nn.Embedding(vocab_size, dim_e, padding_idx=0)
            
        else:
            # pay attention to padding_idx
            self.embed_e = nn.Embedding(vocab_size, dim_e, padding_idx=0).from_pretrained(pretrained_embeddings)
#             self.embed_e.weight.requires_grad=False

    def forward(self, data_bl, length_b):
        """
        Take average of all words in the text.
        
        @param data_bl: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length_b: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        embed_ble = self.embed_e(data_bl)
        out_be = torch.sum(embed_ble, dim=-2)
        out_be /= length_b.float()     
        return out_be

    
def FeedForward(in_features, mid_features, out_features=44, num_layers=1, activation=nn.ReLU(), dropout_rate=0.2):
    """
    Function that creates sequential model (nn.Module) with specified number of layers.
    If 1 layer, returns linear model.
    """
    if num_layers == 1:
        return nn.Linear(in_features, out_features)
    return nn.Sequential(
        nn.Linear(in_features, mid_features),
        *([activation, nn.Dropout(dropout_rate), nn.Linear(mid_features, mid_features)] * max(0, (num_layers - 2))),
        *[activation, nn.Linear(mid_features, out_features)]
    )


class FinalModel(nn.Module):
    """
    Final model that combines embeddings of words in an article (average) and puts it through layer_out.
    """
    def __init__(self, options):
        super(FinalModel, self).__init__()

        self.layer_bag_of_words = BagOfWords(options["VOCAB_SIZE"], options["dim_e"], options["pretrained_embeddings"])
        self.layer_out = FeedForward(
            in_features=options["dim_e"],
            mid_features=options["mid_features"],
            out_features=options["num_classes"], 
            num_layers=options["num_layers"],
            dropout_rate=options["dropout_rate"],
            activation=options["activation"]
        )

    def forward(self, data_bl, length_b):
        # get embeddings
        embed_article_be = self.layer_bag_of_words(data_bl, length_b)
        # use layer_out
        out_bc = self.layer_out(embed_article_be)
        return out_bc