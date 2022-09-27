import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MultiCNNLayer,MatchSimpleNet

class TextCNN(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(TextCNN,self).__init__()
        self.single = single
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        if embedding is None:
            self.num_chars = config.num_chars
            self.embedding_dim = config.embedding_dim
            self.c_embedding = nn.Embedding(self.num_chars,self.embedding_dim)
        else:
            self.num_chars,self.embedding_dim = embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.cnn = MultiCNNLayer(self.embedding_dim,self.n_filters,self.filter_sizes,self.out_dim,self.dropout)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.out_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.out_dim,self.label_size,self.class_size)
    def forward(self,content_chars,c_mask,**kwargs):
        c_emb = self.c_embedding(content_chars)
        c_mask = self.mask_embedding(c_mask)
        c_hid = c_emb*torch.sigmoid(c_mask)
        c_hid = self.cnn(c_hid)
        if self.single:
            logits = self.pred(c_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(c_hid)
        return logits
        
