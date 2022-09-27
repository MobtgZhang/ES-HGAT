import torch 
import torch.nn as nn
import torch.nn.functional as F

from ..layers import GraphConvolution
from ..layers import MatchSimpleNet

class TextGCN(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(TextGCN,self).__init__()
        self.single = single
        self.nhid_dim = config.nhid_dim
        self.ohid_dim = config.ohid_dim
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.gc1 = GraphConvolution(self.embedding_dim, self.nhid_dim)
        self.gc2 = GraphConvolution(self.nhid_dim, self.ohid_dim)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.ohid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.ohid_dim,self.label_size,self.class_size)
        self.gelu = nn.GELU()
    def forward(self,words2ids,i_mask,paris_mat,**kwargs):
        w_emb = self.w_embedding(words2ids)
        w_mask = self.mask_embedding(i_mask)
        w_emb = w_emb*torch.sigmoid(w_mask)
        w_hid = self.gc1(w_emb,paris_mat)
        w_hid = self.gelu(w_hid)
        w_hid = self.gc2(w_hid,paris_mat)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(w_hid)
        return logits
    
