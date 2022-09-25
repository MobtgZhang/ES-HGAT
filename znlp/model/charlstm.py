import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MutiSelfAttLayer
from ..layers import MatchNetwork

class CharBiLSTM(nn.Module):
    def __init__(self,config,single=False,embedding = None):
        super(CharBiLSTM,self).__init__()
        self.hidden_dim = config.hidden_dim
        self.single = single
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,bidirectional=True,batch_first=True)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hidden_dim*2,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchNetwork()
        self.gelu = nn.GELU()
    def forward(self,content_chars,c_mask,**kwargs):
        c_emb = self.w_embedding(content_chars)
        c_mask = self.mask_embedding(c_mask)
        c_hid = c_emb*torch.sigmoid(c_mask)
        c_hid,_ = self.lstm(c_hid)
        c_hid = self.gelu(c_hid)
        if self.single:
            logits = self.pred(c_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(c_hid)
        return logits
    
    