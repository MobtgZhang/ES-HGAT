import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MatchSimpleNet

class TextRCNN(nn.Module):
    def __init__(self,config,c_embedding=None,**kwargs):
        super(TextRCNN,self).__init__()
        self.num_layers = config.num_layers
        self.r_dropout = config.r_dropout
        self.hidden_dim = config.hidden_dim
        if c_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.c_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = c_embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,self.num_layers,bidirectional=True,
                            batch_first=True,dropout=self.r_dropout)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hidden_dim*2+self.embedding_dim,self.class_size)
        self.gelu = nn.GELU()
    def forward(self,content_chars,c_mask,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        c_emb = self.c_embedding(content_chars)
        c_mask = self.mask_embedding(c_mask)
        c_emb_mask = c_emb*torch.sigmoid(c_mask)
        c_hid, _ = self.lstm(c_emb_mask)
        c_hid = torch.cat((c_emb_mask, c_hid), 2)
        c_hid = self.gelu(c_hid)
        logits = self.pred(c_hid.sum(dim=1))
        logits = F.log_softmax(logits,dim=1)
        return logits
