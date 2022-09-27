import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import CapsuleLayer,MatchSimpleNet

class CapsuleNet(nn.Module):
    def __init__(self,config,single=False,embedding=None):
        super(CapsuleNet,self).__init__()
        self.single = single
        self.num_capsules = config.num_capsules
        self.capsules_dim = config.capsules_dim
        self.num_routing = config.num_routing
        if embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.caps = CapsuleLayer(self.num_capsules,self.embedding_dim,self.capsules_dim,self.num_routing)
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.capsules_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.capsules_dim,self.label_size,self.class_size)
    def forward(self,content_words,w_mask,**kwargs):
        w_emb = self.w_embedding(content_words)
        w_mask = self.mask_embedding(w_mask)
        w_hid = w_emb*torch.sigmoid(w_mask)
        w_hid = self.caps(w_hid)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(w_hid)
        return logits

    