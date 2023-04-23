import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from ..layers import HyperGraphAttentionLayerSparse

class HyperGATPre(nn.Module):
    def __init__(self,config):
        super(HyperGATPre,self).__init__()
        self.config = config
        self.hid_dim = config.n_hid
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        self.graph = HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,alpha=0.1,transfer=True)
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hid_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,att_output=False,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pooled_out['last_hidden_state']
        g_hid = self.encode(outs)
        g_hid = self.graph(g_hid,adjmat)
        output = self.pred(g_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits
    