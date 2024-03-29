import os
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger()
from config import check_args,get_args,get_models
from znlp.data import ContentReviewDataset
from znlp.data import batchfy
from znlp.data import to_var
from znlp.utils import DataSaver,Timer
from znlp.eval import evaluate

def main(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # preparing the dataset
    train_dataset = ContentReviewDataset(result_dir,"train",args.pretrain_path,args.window,args.mat_type)
    train_loader = DataLoader(train_dataset,batch_size= args.batch_size,shuffle=True,collate_fn=batchfy)
    dev_dataset = ContentReviewDataset(result_dir,"dev",args.pretrain_path,args.window,args.mat_type)
    dev_loader = DataLoader(dev_dataset,batch_size= args.batch_size,shuffle=False,collate_fn=batchfy)
    args.pretrain_path = train_dataset.tokenizer_file
    # preparing the model
    model = get_models(args,pretrain_path=args.pretrain_path,class_size=train_dataset.class_size,limits_len=train_dataset.limits_len)
    logger.info(model.config.__dict__)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    if args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    elif args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(),lr=args.learning_rate)
    else:
        raise ValueError("unknow optimizer %s"%str(args.optim))
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30,40], gamma=args.gamma)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    # save result
    model_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    if args.train_evaluate:
        save_model_file = os.path.join(model_dir,args.time_step + "-train.csv")
        train_saver = DataSaver(save_model_file)
    train_timer = Timer()
    save_model_file = os.path.join(model_dir,args.time_step + "-valid.csv")
    valid_saver = DataSaver(save_model_file)
    valid_timer = Timer()
    save_model_file = os.path.join(model_dir,args.time_step + "-test.csv")
    best_em = 0.0
    for epoch in range(args.epoches):
        model.train()
        time_bar = tqdm(enumerate(train_loader),total=len(train_loader),leave = True)
        train_timer.reset()
        for idx,item in time_bar:
            item = to_var(item,device)
            optimizer.zero_grad()
            re_dict,targets = item
            predicts = model(**re_dict)
            loss = loss_fn(predicts,targets)            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            time_bar.set_description("epoch %d"%epoch)  # the information bar for the left     
            time_bar.set_postfix(loss="%0.4f"%loss.item(),learning_rate="%0.4e"%(optimizer.param_groups[0]['lr']))  # the information bar for the right
        train_t = train_timer.time()
        if (epoch+1)%args.optim_step == 0:
            scheduler.step()
        if args.train_evaluate:
            train_loss,train_f1val,train_emval = evaluate(train_loader,model,loss_fn,device,"trainset test")
            train_saver.add_values(train_f1val,train_emval,train_loss,train_t)
        valid_timer.reset()
        valid_loss,valid_f1val,valid_emval = evaluate(dev_loader,model,loss_fn,device,"validset test")
        valid_t = valid_timer.time()
        valid_saver.add_values(valid_f1val,valid_emval,valid_loss,valid_t)
        if args.train_evaluate:
            logger.info("epoch:%d,train loss:%0.4f,valid loss:%0.4f, valid f1 score :%0.4f, valid em score :%0.4f"%
                        (epoch,train_loss,valid_loss,valid_f1val,valid_emval))
        else:
            logger.info("epoch:%d,valid loss:%0.4f, valid f1 score :%0.4f, valid em score :%0.4f"%
                        (epoch,valid_loss,valid_f1val,valid_emval))
        # save the best model
        if best_em<valid_emval:
            best_em = valid_emval
            save_model_file = os.path.join(model_dir,args.time_step + ".pth")
            torch.save(model.state_dict(), save_model_file)
if __name__ == "__main__":
    args = get_args()
    check_args(args)
    # First step,create a `logger`
    logger.setLevel(logging.INFO)  # The main switch log level.
    # Second step,create a `handler`,which is used to write the log file.
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    args.time_step = rq
    model_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    if args.save_attention:
        args.save_attention_file = os.path.join(model_dir,rq + '.npz')
    logfile = os.path.join(model_dir,rq + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # The switch of the output files for different log levels.
    # Create a `streamhandler` to print the messagae into the terminal, the level is above `error`.
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Third step, define the format of the output `handler`.
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # Four step,add the `logger` into the `handler`.
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(str(args))
    main(args)
    