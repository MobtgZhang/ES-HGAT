import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import CapsuleLayer,MatchSimpleNet

from transformers import XLNetModel,XLNetTokenizer
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from transformers import AlbertModel,AlbertTokenizer



class CapsuleNetChars(nn.Module):
    def __init__(self,config,c_embedding=None,**kwargs):
        super(CapsuleNetChars,self).__init__()
        self.single = config.single
        self.num_capsules = config.num_capsules
        self.capsules_dim = config.capsules_dim
        self.num_routing = config.num_routing
        if c_embedding is None:
            self.num_chars = config.num_chars
            self.embedding_dim = config.embedding_dim
            self.c_embedding = nn.Embedding(self.num_chars,self.embedding_dim)
        else:
            self.num_chars,self.embedding_dim = c_embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        self.caps = CapsuleLayer(self.num_capsules,self.embedding_dim,self.capsules_dim,self.num_routing)
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.capsules_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.capsules_dim,self.label_size,self.class_size)
    def forward(self,content_chars,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_emb = self.c_embedding(content_chars)
        w_hid = self.caps(w_emb)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=-1)
        else:
            logits = self.pred(w_hid)
        return logits

class CapsuleNetWords(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(CapsuleNetWords,self).__init__()
        self.single = config.single
        self.num_capsules = config.num_capsules
        self.capsules_dim = config.capsules_dim
        self.num_routing = config.num_routing
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.caps = CapsuleLayer(self.num_capsules,self.embedding_dim,self.capsules_dim,self.num_routing)
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.capsules_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.capsules_dim,self.label_size,self.class_size)
    def forward(self,content_words,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_emb = self.w_embedding(content_words)
        w_hid = self.caps(w_emb)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=-1)
        else:
            logits = self.pred(w_hid)
        return logits
class PrtrainCapsModel(nn.Module):
    def __init__(self,config,**kwargs):
        super(PrtrainCapsModel,self).__init__()
        # chars embedding
        if "bert" in config.pretrain_path:
            self.pretrain = BertModel.from_pretrained(config.pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)
        elif "roberta" in config.pretrain_path:
            self.pretrain = RobertaModel.from_pretrained(config.pretrain_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(config.pretrain_path)
        elif "albert" in config.pretrain_path:
            self.pretrain = AlbertModel.from_pretrained(config.pretrain_path)
            self.tokenizer = AlbertTokenizer.from_pretrained(config.pretrain_path)
        elif "xlnet" in config.pretrain_path:
            self.pretrain = XLNetModel.from_pretrained(config.pretrain_path)
            self.tokenizer = XLNetTokenizer.from_pretrained(config.pretrain_path)
        else:
            raise ValueError("Unknown model name: %s"%config.pretrain_path)
        
        self.num_capsules = config.num_capsules
        self.capsules_dim = config.capsules_dim
        self.num_routing = config.num_routing
        self.single = config.single
        
        self.caps = CapsuleLayer(self.num_capsules,self.pretrain.config.hidden_size,self.capsules_dim,self.num_routing)
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.capsules_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.capsules_dim,self.label_size,self.class_size)
    def forward(self,content,**kwargs):
        device = next(self.parameters()).device
        outputs = self.tokenizer(content,return_tensors="pt",padding=True,truncation=True)
        outputs = outputs.to(device)
        pooled_out = self.pretrain(**outputs)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.caps(dc_hid)
        # output layer
        if self.single:
            output = self.pred(dc_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=-1)
        else:
            logits = self.pred(dc_hid)
        return logits
    