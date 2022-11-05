import torch
import torch.nn as nn
from ..layers import GraphGNN
from ..layers import MatchSimpleNet,ReadoutLayer


class TextING(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(TextING,self).__init__()
        self.single = config.single
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.layers_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.layers_list.append(GraphGNN(self.embedding_dim,self.hidden_dim))
            else:
                self.layers_list.append(GraphGNN(self.hidden_dim,self.hidden_dim))
        if self.single:
            self.class_size = config.class_size
            self.pred = ReadoutLayer(self.hidden_dim,self.hidden_dim,self.class_size,self.dropout)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hidden_dim,self.label_size,self.class_size)
    def forward(self,words2ids,i_mask,paris_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_out = self.w_embedding(words2ids)
        for k in range(self.num_layers):
            if k==0:
                t_mask = i_mask.float().unsqueeze(-1).repeat(1,1,self.embedding_dim)
            else:
                t_mask = i_mask.float().unsqueeze(-1).repeat(1,1,self.hidden_dim)
            w_out = self.layers_list[k](w_out,paris_mat,t_mask)
        if self.single:
            logits = self.pred(w_out,i_mask)
        else:
            logits = self.pred(w_out)
        return logits
        
        
