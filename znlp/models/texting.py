import torch
import torch.nn as nn
from ..layers import GraphGNN
from ..layers import MatchSimpleNet,ReadoutLayer


class TextING(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(TextING,self).__init__()
        self.single = single
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        
        self.layers_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.layers_list.append(GraphGNN(self.embedding_dim,self.hidden_dim))
            else:
                self.layers_list.append(GraphGNN(self.hidden_dim,self.hidden_dim))
        if single:
            self.class_size = config.class_size
            self.pred = ReadoutLayer(self.hidden_dim,self.hidden_dim,self.class_size,self.dropout)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hidden_dim,self.label_size,self.class_size)
    def forward(self,words2ids,i_mask,paris_mat,**kwargs):
        w_emb = self.w_embedding(words2ids)
        i_mask_emb = self.mask_embedding(i_mask)
        w_out = w_emb*torch.sigmoid(i_mask_emb)
        for k in range(self.num_layers):
            w_out = self.layers_list[k](w_out,paris_mat)
        if self.single:
            logits = self.pred(w_out,i_mask)
        else:
            logits = self.pred(w_out)
        return logits
        
        
