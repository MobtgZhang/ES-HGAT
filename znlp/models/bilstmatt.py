import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MutiAttentionLayer
from ..layers import MatchSimpleNet

class BiLSTMAtt(nn.Module):
    def __init__(self,config,single=False,embedding = None):
        super(BiLSTMAtt,self).__init__()
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        self.single = single
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,bidirectional=True)
        self.att = MutiAttentionLayer(self.hidden_dim*2,self.hidden_dim*2,self.hidden_dim*2)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hidden_dim*2,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hidden_dim*2,self.label_size,self.class_size)
        self.gelu = nn.GELU()
    def forward(self,content_words,w_mask,**kwargs):
        w_emb = self.w_embedding(content_words)
        w_mask = self.mask_embedding(w_mask)
        w_hid = w_emb*torch.sigmoid(w_mask)
        w_hid,_ = self.lstm(w_hid)
        w_hid,w_att = self.att(w_hid)
        w_hid = self.gelu(w_hid)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(w_hid)
        return logits,w_att
    
    