import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MatchSimpleNet

class TextRCNN(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(TextRCNN,self).__init__()
        self.single = single
        self.num_layers = config.num_layers
        self.r_dropout = config.r_dropout
        self.hidden_dim = config.hidden_dim
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,self.num_layers,bidirectional=True,
                            batch_first=True,dropout=self.r_dropout)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hidden_dim*2+self.embedding_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hidden_dim*2+self.embedding_dim,self.label_size,self.class_size,self.r_dropout)
        self.gelu = nn.GELU()
    def forward(self,content_words,w_mask,**kwargs):
        w_emb = self.w_embedding(content_words)
        w_mask = self.mask_embedding(w_mask)
        c_emb_mask = w_emb*torch.sigmoid(w_mask)
        c_hid, _ = self.lstm(c_emb_mask)
        c_hid = torch.cat((c_emb_mask, c_hid), 2)
        c_hid = self.gelu(c_hid)
        if self.single:
            logits = self.pred(c_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(c_hid)
        return logits
