import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
    def forward(self,x):
        x = x * torch.tanh(F.softplus(x))
        return x
class MultiCNNLayer(nn.Module):
    def __init__(self, hidden_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super(MultiCNNLayer,self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, hidden_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.mish = Mish()

    def forward(self, hidden_output):  # input shape of [batch_size,seq_len,hidden_size]
        hidden = hidden_output.unsqueeze(1)  #  [batch_size, 1, seq_len, hidden_size]
        # for number of len(filter_sizes) elements，and the shape of every elemnets is that 
        # [batch_size, n_filters, seq_len - filter_sizes[n] + 1]
        conved = [self.mish(conv(hidden)).squeeze(3) for conv in self.convs]
        # for number of len(filter_sizes) elements，and the shape of every elemnets is that [batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=-1))  # [batch size, n_filters * len(filter_sizes)]
        cat = cat.transpose(2,1)
        return self.fc(cat)
