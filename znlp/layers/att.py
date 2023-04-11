import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutiAttentionLayer(nn.Module):
    def __init__(self,dim_q,dim_k,dim_v):
        super(MutiAttentionLayer, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q_mat = nn.Linear(dim_q,dim_v)
        self.k_mat = nn.Linear(dim_k,dim_v)
        self.v_mat = nn.Linear(dim_v,dim_v)
        self._norm_fact = 1.0/math.sqrt(dim_k)

    def forward(self,query,key=None,value=None):
        if key is None:
            assert self.dim_k == self.dim_q
            key = query
        if value is None:
            assert self.dim_k == self.dim_v
            value = key
        q_mat = self.q_mat(query) # (batch_size,seq_lenk,dim_k)
        k_mat = self.k_mat(key) # (batch_size,seq_lenq,dim_k)
        v_mat = self.v_mat(value) # (batch_size,seq_lenv,dim_v)
        atten = torch.bmm(q_mat,k_mat.permute(0,2,1))*self._norm_fact # (batch_size,seq_lenq,seq_lenk)
        atten = F.softmax(atten,dim=-1)
        o_mat = torch.bmm(atten,v_mat) # (batch_size,seq_lenq,dim_v)
        return o_mat,atten

class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o
