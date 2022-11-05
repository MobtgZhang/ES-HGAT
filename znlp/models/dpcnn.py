import torch
import torch.nn as nn

from ..layers import DPCNN

class DPCNNModel(nn.Module):
    def __init__(self,config,c_embedding=None,**kwargs):
        super(DPCNNModel,self).__init__()
        if c_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.c_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = c_embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        self.dpcnn = DPCNN(self.embedding_dim,config.class_size)
    def forward(self,content_chars,**kwargs):
        c_emb = self.c_embedding(content_chars)
        return self.dpcnn(c_emb)

