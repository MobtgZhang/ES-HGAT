import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import HyperGraphAttentionLayerSparse
from ..layers import MatchSimpleNet

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, input_size, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
        self.lin = nn.Linear(input_size,n_hid)
        self.gelu = nn.GELU()
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, n_hid, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)        
    def forward(self, x, adj_mat):
        
        x = self.gat1(x, adj_mat)
        x = self.gelu(self.lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj_mat)
        return x

class HyperGAT(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(HyperGAT,self).__init__()
        self.single = single
        self.dropout = config.dropout
        self.n_hid = config.n_hid
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.gat = HGNN_ATT(self.embedding_dim,self.n_hid,self.dropout)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.n_hid,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.n_hid,self.label_size,self.class_size)
    def forward(self,words2ids,paris_mat,**kwargs):
        w_emb = self.w_embedding(words2ids)
        gat_hid = self.gat(w_emb,paris_mat)
        
        if self.single:
            logits = self.pred(gat_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(gat_hid)
        return logits

