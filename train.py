import torch 
from main import device
import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx 
import random 
from sklearn.metrics import f1_score

def train_step(tr_loader, model, optimizer, cls_criterion, args):
    model.train()
    all_count = 0
    L=0
    for data in tr_loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            pass
        else:
            pred = model(data)
            optimizer.zero_grad()
            is_labeled = data.y == data.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            L+=loss.item()
            all_count += 1
    return L/all_count    

def eval_step(loader, model, cls_criterion, compute_loss=False, args=None, prints=False):
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    all_count = 0
    L=0
    for data in loader:
        data = data.to(device)

        if data.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(data)
            if compute_loss:
                is_labeled = data.y == data.y
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                L+=loss.item()
                all_count += 1
            y_true.append(data.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    if compute_loss:
        loss = L/all_count

    y_true = torch.cat(y_true, dim = 0).numpy()
    soft_y_pred = torch.cat(y_pred, dim = 0).numpy()
    hard_y_pred = (torch.cat(y_pred, dim = 0).sigmoid() > 0.5).int().numpy()
    if prints:
        print(hard_y_pred)
    
    return eval_rocauc(y_true, soft_y_pred), loss, f1_score(y_true, hard_y_pred), accuracy_score(y_true, hard_y_pred)

def train_data(args, tr_loader, val_loader, te_loader_1, te_loader_2, model, optimizer, pth=""):

   
    cls_criterion = torch.nn.BCEWithLogitsLoss()

    bval=args.worst_case
    btest_1=0
    btest_2=0
    btest_1_f1 = 0
    btest_2_f1 = 0
    btest_1_acc = 0
    btest_2_acc = 0 
    bval_f1 = 0 
    bval_roc = 0 
    bval_acc = 0 
    patience = args.patience
    count = 0
    for epoch in range(1, 1+bval):
        tr_loss = train_step(tr_loader, model, optimizer, cls_criterion, args)
        tr_roc, _, tr_f1, tr_acc = eval_step(tr_loader, model, cls_criterion, compute_loss=False, args=args)
        val_roc, val_loss, val_f1, val_acc = eval_step(val_loader, model, cls_criterion, True, args)
        if epoch == 80:
            te_roc_1, te_loss_1, te_f1_small, te_acc_1= eval_step(te_loader_1, model, cls_criterion,  True, args)
        else:
            te_roc_1, te_loss_1, te_f1_small, te_acc_1 = eval_step(te_loader_1, model, cls_criterion,  True, args)
        te_roc_2, te_loss_2, te_f1_large, te_acc_2 = eval_step(te_loader_2, model, cls_criterion,  True, args)
        if bval>val_loss:
            bval=val_loss
            bval_f1 = val_f1 
            bval_roc = val_roc 
            bval_acc = val_acc
            btest_1=te_roc_1
            btest_2=te_roc_2
            btest_1_f1 = te_f1_small 
            btest_2_f1 = te_f1_large
            btest_1_acc = te_acc_1
            btest_2_acc = te_acc_2
            count = 0
            if pth:
                torch.save(model.state_dict(), pth)
        else:
            count += 1
            if count == patience:
                break
        print('Epoch: {:02d}, trloss: {:.4f}, trroc: {:.4f}, trf1: {:.4f}, tracc: {:.4f}, , valloss: {:.4f}, valroc: {:.4f}, valf1: {:.4f}, valacc: {:.4f},testloss_1: {:.4f}, testroc_1: {:.4f}, testf1_1: {:.4f}, testacc_1: {:.4f}, testloss_2: {:.4f}, testroc_2: {:.4f}, testf1_2: {:.4f}, testacc_2: {:.4f}.'.format(epoch,tr_loss,tr_roc, tr_f1, tr_acc, val_loss,val_roc, val_f1, val_acc, te_loss_1, te_roc_1, te_f1_small, te_acc_1, te_loss_2, te_roc_2, te_f1_large, te_acc_2))
    print("Best test roc/f1: test_roc_1:{:.4f}, test_f1_1:{:.4f}, test_acc_1:{:.4f}, test_roc_2:{:.4f}, test_f1_2:{:.4f}, test_acc_2:{:.4f}, val_f1:{:.6f}, val_roc:{:.4f}, val_acc:{:.4f}".format(btest_1, btest_1_f1, btest_1_acc, btest_2, btest_2_f1, btest_2_acc, bval_f1, bval_roc, bval_acc))
    return btest_1, btest_2, btest_1_f1, btest_2_f1, btest_1_acc, btest_2_acc, bval_f1, bval_roc, bval_acc

def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)