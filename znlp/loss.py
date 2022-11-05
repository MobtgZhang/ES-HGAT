import torch
import torch.nn as nn
class MultipleLoss(nn.Module):
    def __init__(self):
        super(MultipleLoss,self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,y_pred,y_true):
        assert (y_pred.ndim == 3 and y_true.ndim == 2) or (y_pred.ndim == 2 and y_true.ndim == 1)
        if y_pred.ndim == 3 and y_true.ndim == 2:
            batch_size,num_attributes,num_classes = y_pred.shape
            #device = y_pred.device
            loss = sum([self.loss_fn(y_pred[:,k,:],y_true[:,k]) for k in range(num_attributes)])/num_attributes
            return loss
        else:
            return self.loss_fn(y_pred,y_true)



    