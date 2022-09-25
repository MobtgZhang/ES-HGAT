import torch
import torch.nn as nn


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

class CapsuleLayer(nn.Module):
    def __init__(self,num_capsules,in_dim,capsules_dim,num_routing):
        super(CapsuleLayer,self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.capsules_dim = capsules_dim
        self.num_routing = num_routing
        self.W = nn.Parameter(0.001*torch.randn(num_capsules,in_dim,capsules_dim),
                              requires_grad=True)
    def forward(self,x_tensor):
        """
        x_tensor: (batch_size,seq_len,in_dim)
        out_tensor: (batch_size,num_capsules,capsules_dim)
        """
        batch_size = x_tensor.size(0)
        device = x_tensor.device 
        x_tensor = x_tensor.unsqueeze(1) # (batch_size,1,seq_len,embedding_dim)
        u_hat = torch.matmul(x_tensor,self.W) 
        # W @ x = (batch_size, 1, seq_len, embedding_dim) @ (num_capsules,embedding_dim,capsules_dim) =
        # (batch_size, num_capsules, seq_len, capsules_dim)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        in_caps = temp_u_hat.shape[2]
        b = torch.rand(batch_size, self.num_capsules, in_caps, 1).to(device)
        for route_iter in range(self.num_routing - 1):
            c = b.softmax(dim=1)
            # element-wise multiplication
            c_extend = c.expand_as(temp_u_hat)
            s = (c_extend * temp_u_hat).sum(dim=2)
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_capsules, seq_len, capsules_dim) @ (batch_size, num_capsules, capsules_dim, 1)
            # -> (batch_size, num_capsules, seq_len, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        c_extend = c.expand_as(u_hat)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)
        return v


