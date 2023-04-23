import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from ..layers import MultiCNNLayer

class TextCNNPre(nn.Module):
    def __init__(self,config):
        super(TextCNNPre,self).__init__()
        self.config = config
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.hid_dim = config.hid_dim
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        self.cnn = MultiCNNLayer(self.hid_dim,self.n_filters,self.filter_sizes,self.out_dim,self.dropout)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,input_ids,attention_mask,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        dc_hid = self.cnn(dc_hid)
        # output layer
        output = self.pred(dc_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits
    