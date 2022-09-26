import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutiAttentionLayer(nn.Module):
    def __init__(self,in_size,dim_k,dim_v):
        super(MutiAttentionLayer, self).__init__()
        self.in_size = in_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q_mat = nn.Linear(in_size,dim_k)
        self.k_mat = nn.Linear(in_size,dim_k)
        self.v_mat = nn.Linear(in_size,dim_v)
        self._norm_fact = 1.0/math.sqrt(dim_k)

    def forward(self,query,key=None,value=None):
        if key is None:
            key = query
        if value is None:
            value = key
        q_mat = self.q_mat(query) # (batch_size,seq_len,dim_k)
        k_mat = self.k_mat(key) # (batch_size,seq_len,dim_k)
        v_mat = self.v_mat(value) # (batch_size,seq_len,dim_v)
        atten = torch.bmm(q_mat,k_mat.permute(0,2,1))*self._norm_fact # (batch_size,seq_lenq,seq_lenk)
        atten = F.softmax(atten,dim=-1)
        o_mat = torch.bmm(atten,v_mat) # (batch_size,seq_lenq,dim_v)
        return o_mat,atten
