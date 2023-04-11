import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

class PretrainingModel(nn.Module):
    def __init__(self,config):
        super(PretrainingModel,self).__init__()
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        self.hid_dim = self.pretrain.config.hidden_size
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hid_dim,self.class_size)
    def forward(self,input_ids,attention_mask,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        pooled_out = pooled_out['last_hidden_state']
        # output layer
        output = self.pred(pooled_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits

