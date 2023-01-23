import copy
import os
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from znlp.data import Dictionary, to_var
from znlp.data import ContentReviewDataset,batchfy
from znlp.data import Timer,DataSaver
from znlp.eval import evaluate
from znlp.loss import MultipleLoss

logger = logging.getLogger()
from config import get_models,check_args,get_args
  
def main(args):
    device = torch.device("cpu" if not args.cuda and torch.cuda.is_available() else "cuda:%d"%args.device_id)
    # preparing the dataset
    words_dict_file_name = os.path.join(args.result_dir,args.dataset,"words-dict.json")
    words_dict = Dictionary.load(words_dict_file_name)
    chars_dict_file_name = os.path.join(args.result_dir,args.dataset,"chars-dict.json")
    chars_dict = Dictionary.load(chars_dict_file_name)
    # preparing the dataset
    load_train_dataset_file = os.path.join(args.result_dir,args.dataset,"train-graph.pkl")
    load_valid_dataset_file = os.path.join(args.result_dir,args.dataset,"valid-graph.pkl")
    load_test_dataset_file = os.path.join(args.result_dir,args.dataset,"test-graph.pkl")
    train_dataset = ContentReviewDataset(load_train_dataset_file,words_dict,chars_dict)
    train_loader = DataLoader(train_dataset,batch_size= args.batch_size,shuffle=True,collate_fn=batchfy)
    valid_dataset = ContentReviewDataset(load_valid_dataset_file,words_dict,chars_dict)
    valid_loader = DataLoader(valid_dataset,batch_size= args.batch_size,shuffle=False,collate_fn=batchfy)
    if os.path.exists(load_test_dataset_file):
        test_dataset = ContentReviewDataset(load_test_dataset_file,words_dict,chars_dict)
        test_loader = DataLoader(test_dataset,batch_size= args.batch_size,shuffle=False,collate_fn=batchfy)
    else:
        test_dataset = None
        test_loader = None
    # preparing the model
    model = get_models(args)
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
    test_saver = DataSaver(save_model_file)
    test_timer = Timer()
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
        valid_loss,valid_f1val,valid_emval = evaluate(valid_loader,model,loss_fn,device,"validset test",args.save_attention_file)
        valid_t = valid_timer.time()
        valid_saver.add_values(valid_f1val,valid_emval,valid_loss,valid_t)
        if test_loader is not None:
            test_timer.reset()
            test_loss,test_f1val,test_emval = evaluate(test_loader,model,loss_fn,device,"testset test")
            test_t = test_timer.time()
            test_saver.add_values(test_f1val,test_emval,test_loss,test_t)
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
    ch.setLevel(logging.ERROR)
    # Third step, define the format of the output `handler`.
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # Four step,add the `logger` into the `handler`.
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(str(args))
    main(args)


