import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from ..layers import GraphConvolution,MutiAttentionLayer,SFU

class TextGCN(nn.Module):
    def __init__(self):
        super(TextGCN,self).__init__()
    def forward(self):
        pass
class TextGCNPre(nn.Module):
    def __init__(self,config):
        super(TextGCNPre,self).__init__()
        self.dropout = config.dropout
        self.nhid_dim = config.nhid_dim
        self.ohid_dim = config.ohid_dim
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        self.gc1 = GraphConvolution(self.embedding_dim, self.nhid_dim)
        self.gc2 = GraphConvolution(self.nhid_dim, self.ohid_dim)
        # funsion layer
        self.att = MutiAttentionLayer(self.ohid_dim,self.ohid_dim,self.ohid_dim)
        self.sfu = SFU(self.ohid_dim,self.ohid_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.ohid_dim),
            nn.GELU()
        )
        self.class_size = config.class_size
        self.pred = nn.Linear(self.ohid_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,att_output=False,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        
        w_hid = self.gc1(dc_hid,adjmat)
        w_hid = self.gelu(w_hid)
        w_hid = self.gc2(w_hid,adjmat)
        # fusion layer
        datt_hid,datts = self.att(dc_hid,w_hid)
        dsfu_hid = self.sfu(datt_hid,dc_hid)
        
        # output layer
        output = self.pred(dsfu_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        if att_output:
            return logits,datts
        else:
            return logits
