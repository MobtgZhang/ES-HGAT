import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from ..layers import CapsuleLayer

class CapsuleNetPre(nn.Module):
    def __init__(self,config):
        super(CapsuleNetPre,self).__init__()
        self.config = config
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        
        self.num_capsules = config.num_capsules
        self.capsules_dim = config.capsules_dim
        self.num_routing = config.num_routing
        
        self.caps = CapsuleLayer(self.num_capsules,self.pretrain.config.hidden_size,self.capsules_dim,self.num_routing)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.capsules_dim,self.class_size)
    def forward(self,input_ids,attention_mask,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.caps(dc_hid)
        # output layer
        output = self.pred(dc_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits
