from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import to_var
from sklearn.metrics import f1_score, accuracy_score

def evaluate(dataloader,model,loss_fn,device,name,save_file=None):
    model.eval()
    time_bar = tqdm(enumerate(dataloader),total=len(dataloader),leave = True)
    all_loss = 0.0
    
    y_pred = []
    y_true = []
    if save_file is not None:
        save_attentions_list = []
    for idx,item in time_bar:
        item = to_var(item,device)
        re_dict,targets = item
        predicts = model(**re_dict)
        if save_file is not None:
            _,atts = model.get_features(**re_dict)
            atts = atts.detach().cpu().numpy()
            save_attentions_list += [v for v in atts]
        loss = loss_fn(predicts,targets)
        predicts = torch.argmax(predicts,dim=-1)
        all_loss += loss.item()
        y_pred.append(predicts)
        y_true.append(targets)
        time_bar.set_description("%s epoch %d"%(name,idx))
    if save_file is not None:
        np.savez(save_file,save_attentions_list=np.array(save_attentions_list,dtype=object))
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()
    all_f1val = 0.0
    all_emval = 0.0
    if len(y_pred.shape) == 2:
        _,num_attributes = y_pred.shape
        for k in range(num_attributes):
            all_f1val += f1_score(y_true[:,k], y_pred[:,k], average='macro')
            all_emval += accuracy_score(y_true[:,k],y_pred[:,k])
        all_f1val /= num_attributes
        all_emval /= num_attributes
    else:
        all_f1val = f1_score(y_true, y_pred, average='macro')
        all_emval = accuracy_score(y_true,y_pred)
    all_loss /= len(dataloader)
    return all_loss,all_f1val,all_emval
