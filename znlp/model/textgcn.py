import torch
import torch.nn as nn

from ..layers import GraphConvolution

class TextGCN(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(TextGCN,self).__init__()
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        
        self.gc1 = GraphConvolution(config.nfeat_dim, config.nhid_dim)
        self.gc2 = GraphConvolution(config.nhid_dim, config.ohid_dim)
    def forward(self,words2ids,i_mask,paris_mat,**kwargs):
        pass

