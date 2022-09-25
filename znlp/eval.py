from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from znlp.data import to_var

class FBetaLoss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7,beta=1.0):
        super(FBetaLoss,self).__init__()
        self.epsilon = epsilon
        self.beta = beta
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 3
        assert y_true.ndim == 2
        batch_size,second_class_num,first_class_num = y_pred.shape
        y_true = torch.cat([F.one_hot(item,second_class_num).reshape(1,-1,second_class_num) for item in y_true],dim=0)
        y_true = y_true.transpose(2,1)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = (1+self.beta**2)* (precision*recall) / (self.beta**2*precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1-f1
def f1_score(y_pred,y_true,beta=1.0,epsilon = 1e-7):
    """_summary_
    Args:
        y_pred (b,s,f): _description_
        y_true (b,f): _description_
    """
    batch_size,second_class_num,first_class_num = y_pred.shape
    y_true = torch.cat([F.one_hot(item,second_class_num).reshape(1,-1,second_class_num) for item in y_true],dim=0)
    y_true = y_true.transpose(2,1)
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (1+beta**2)* (precision*recall) / (beta**2*precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return 1-f1
class ExtractMatch(nn.Module):
    def __init__(self):
        super(ExtractMatch,self).__init__()
    def forward(self,predicts,targets):
        """The Extract Match Class
        Args:
            predicts (batch_size,second_class_num,first_class_num): fine graind for two channels
            targets (batch_size,first_class_num): name2ids
        """
        predicts = torch.argmax(predicts,dim=1)
        out = predicts.eq(targets).float()
        out = out.mean(dim=0)
        return out
def extract_match(predicts,targets):
    """_summary_
    Args:
        predicts (batch_size,second_class_num,first_class_num): _description_
        targets (batch_size,first_class_num): _description_
    """
    predicts = torch.argmax(predicts,dim=1)
    out = predicts.eq(targets).float()
    out = out.mean(dim=0)
    return out

def evaluate(dataloader,model,loss_fn,device,name):
    model.eval()
    time_bar = tqdm(enumerate(dataloader),total=len(dataloader),leave = True)
    all_loss = 0.0
    all_f1val = 0.0
    all_emval = 0.0
    for idx,item in time_bar:
        item = to_var(item,device)
        re_dict,targets = item
        predicts = model(**re_dict)
        loss = loss_fn(predicts,targets)
        f1value = f1_score(predicts,targets)
        emvalue = extract_match(predicts,targets)
        all_loss += loss.item()
        all_emval += emvalue.mean().item()
        all_f1val += f1value.mean().item()
        time_bar.set_description("%s epoch %d"%(name,idx))
    all_loss /= len(dataloader)
    all_f1val /= len(dataloader)
    all_emval /= len(dataloader)
    return all_loss,all_f1val,all_emval
