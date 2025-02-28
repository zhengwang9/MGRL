'''
使用多中心进行训练
每个中心拥有自己的分类头
'''

import os
os.environ["OMP_NUM_THREADS"]="8"
import argparse
import setproctitle
setproctitle.setproctitle(f"Run")
import warnings
warnings.filterwarnings("ignore")
import pickle
import logging
import random
import os
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader

from model import ABMIL_MH
from Datasets_multiC import Survival_Dataset_MultiCenter_TtumorDFS as dataset
from Datasets_multiC import Survival_Dataset_SingleCenter_external_TannoDFS as exdataset
import torch
import numpy as np
from lifelines.utils import concordance_index as ci
import matplotlib.pyplot as plt



try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

def get_params():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--label_path', type=str, default='DFS.xlsx')
    parser.add_argument('--feat_path_center1', type=str, default='')
    parser.add_argument('--feat_path_center2', type=str, default='')
    parser.add_argument('--feat_path_center3', type=str, default='')
    parser.add_argument('--feat_path_center4', type=str, default='')
    parser.add_argument("--fold_num", type=int, default=0)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--center", type=str, default='MultiCenter')
    parser.add_argument("--center_id", type=str, default='')
    parser.add_argument("--model", type=str, default='abmil')
    parser.add_argument("--version", type=str, default='test')
    parser.add_argument("--feat_label", type=str, default='ctrans')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--split_path', type=str,default='/multi_center_5fold_DFS.pkl')
    parser.add_argument('--test_split_path', type=str,default='/data_split/multi_center_test_DFS.pkl')
    parser.add_argument('--log_path', type=str,default='')
    parser.add_argument('--model_path', type=str,default='')
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of model training")
    parser.add_argument("--l2_reg_alpha", type=float, default=1e-8)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--cox_loss", type=int, default=1)
    parser.add_argument("--reduction", type=str, default='max')
    args, _ = parser.parse_known_args()
    return args
args = get_params()
device = torch.device(f"cuda:{args.device}")

if not os.path.exists(os.path.join(args.model_path,args.center,args.model)):
    os.makedirs(os.path.join(args.model_path,args.center,args.model))
if not os.path.exists(os.path.join(args.log_path,args.center,args.model)):
    os.makedirs(os.path.join(args.log_path,args.center,args.model))
date = str(datetime.now().date())
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
file_handler = logging.FileHandler(f'{args.log_path}/{args.center}/{args.model}/{args.feat_label}_{args.seed}_{args.center_id}_fold{args.fold_num}_{date}-v{args.version}.log')
file_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(file_handler)

def _neg_partial_log(hazards, time, event):
    current_batch_len = len(hazards)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = time[j] >= time[i]
    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.to(device)
    train_ystatus =  event
    theta = hazards.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)
    return loss_nn




def eval_model(model,loader,loader_type,batch_size,cuda=True):
    with torch.no_grad():
        model.eval()
        loss_total = 0.
        hazards_all = torch.Tensor()
        censorship_all = torch.Tensor()
        survtime_all = torch.Tensor()
        censorship_batch = torch.Tensor()
        hazards_batch = torch.Tensor()
        survtime_batch = torch.Tensor()
        for batch_idx, dict in enumerate(tqdm(loader)):
            feat = dict['feat']
            censorship = dict['censorship']
            survtime = dict['survtime']
            center=dict['center']
            hazards_max=torch.tensor([[0.]]).to(device)
            with torch.no_grad():
                feat = feat.to(device)
                censorship = censorship.to(device)
                survtime = survtime.to(device)
            hazards_1,hazards_2,hazards_3 = model(feat,center)
            if loader_type=='Test' and args.reduction=='mean':
                stacked_tensors = torch.stack([hazards_1, hazards_2, hazards_3])
                hazards=torch.mean(stacked_tensors,dim=0,keepdim=True).squeeze(0)
            elif loader_type=='Test' and args.reduction=='max':
                hazards_max=torch.maximum(hazards_max,hazards_1)
                hazards_max=torch.maximum(hazards_max,hazards_2)
                hazards_max=torch.maximum(hazards_max,hazards_3)
                hazards=hazards_max
            else:
                if center==1:
                    hazards=hazards_1
                elif center==2:
                    hazards=hazards_2
                else:
                    hazards=hazards_3
            if hazards_batch.shape[0] == 0:
                hazards_batch = hazards
                hazards = hazards.detach().cpu()
            else:
                hazards_batch = torch.cat((hazards_batch, hazards), dim=0)
                hazards = hazards.detach().cpu()
            if survtime_batch.shape[0] == 0:
                survtime_batch = survtime
                survtime = survtime.detach().cpu()
            else:
                survtime_batch = torch.cat((survtime_batch, survtime), dim=0)
                survtime = survtime.detach().cpu()

            if censorship_batch.shape[0] == 0:
                censorship_batch = censorship
                censorship = censorship.detach().cpu()
            else:
                censorship_batch = torch.cat((censorship_batch, censorship), dim=0)
                censorship = censorship.detach().cpu()
            if hazards_all.shape[0] == 0:
                hazards_all = -1 * hazards
            else:
                hazards_all = torch.cat((hazards_all, -1 * hazards), dim=0)
            if survtime_all.shape[0] == 0:
                survtime_all = survtime
            else:
                survtime_all = torch.cat((survtime_all, survtime), dim=0)
            if censorship_all.shape[0] == 0:
                censorship_all = censorship.detach().cpu()
            else:
                censorship_all = torch.cat((censorship_all, censorship), dim=0)
            if torch.sum(censorship_batch) > 0. and (batch_idx + 1)%batch_size==0:
                loss = _neg_partial_log(hazards_batch, survtime_batch, censorship_batch)
                loss = args.cox_loss * loss
                loss_total+=loss.item()
                censorship_batch = torch.Tensor()
                hazards_batch = torch.Tensor()
                survtime_batch = torch.Tensor()
            elif (batch_idx + 1) % batch_size == 0 and torch.sum(censorship_batch) == 0:
                censorship_batch = torch.Tensor()
                hazards_batch = torch.Tensor()
                survtime_batch = torch.Tensor()
        c_index = ci(survtime_all.data.cpu(), hazards_all.data.cpu(), censorship_all.data.cpu())
        logging.info('{} Loss:{}; {} Ci:{}'.format(loader_type,loss_total, loader_type,c_index))
        return loss_total,c_index

def train_model(n_epochs, model, optimizer, batch_size,train_loader,val_loader,test_loader,cuda=True):
    if cuda: model = model.to(device)
    best_ci = 0.
    best_epoch = 0
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        print('\n')
        logging.info('\n')
        train_num = len(train_loader)
        train_loss=0.
        with torch.no_grad():
            hazards_all=torch.Tensor()
            censorship_all=torch.Tensor()
            survtime_all=torch.Tensor()
        censorship_batch = torch.Tensor()
        hazards_batch = torch.Tensor()
        survtime_batch = torch.Tensor()
        for batch_idx, dict in enumerate(tqdm(train_loader)):
            feat=dict['feat']
            censorship=dict['censorship']
            survtime=dict['survtime']
            center=dict['center']
            with torch.no_grad():
                feat = feat.to(device)
            censorship = censorship.to(device)
            survtime=survtime.to(device)
            hazards_1,hazards_2,hazards_3 = model(feat,center)
            if center==1:
                hazards=hazards_1
            elif center==2:
                hazards=hazards_2
            else:
                hazards=hazards_3
            del feat
            if hazards_batch.shape[0]==0:
                hazards_batch=hazards
                hazards=hazards.detach().cpu()
            else:
                hazards_batch=torch.cat((hazards_batch,hazards),dim=0)
                hazards=hazards.detach().cpu()
            if survtime_batch.shape[0]==0:
                survtime_batch=survtime
                survtime=survtime.detach().cpu()
            else:
                survtime_batch=torch.cat((survtime_batch,survtime),dim=0)
                survtime=survtime.detach().cpu()
            if censorship_batch.shape[0]==0:
                censorship_batch=censorship
                censorship=censorship.detach().cpu()
            else:
                censorship_batch=torch.cat((censorship_batch,censorship),dim=0)
                censorship=censorship.detach().cpu()
            if (((batch_idx+1)%batch_size==0) or (batch_idx==len(train_loader)-1)) and torch.sum(censorship_batch)>0.:
                optimizer.zero_grad()
                loss = _neg_partial_log(hazards_batch, survtime_batch, censorship_batch)
                loss = args.cox_loss * loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                censorship_batch = torch.Tensor()
                hazards_batch = torch.Tensor()
                survtime_batch = torch.Tensor()
            elif (batch_idx+1)%batch_size==0 and torch.sum(censorship_batch)==0:
                censorship_batch = torch.Tensor()
                hazards_batch = torch.Tensor()
                survtime_batch = torch.Tensor()
            if hazards_all.shape[0]==0: hazards_all=-1*hazards
            else: hazards_all=torch.cat((hazards_all,-1*hazards),dim=0)
            if survtime_all.shape[0]==0:
                survtime_all=survtime
            else:
                survtime_all=torch.cat((survtime_all,survtime),dim=0)
            if censorship_all.shape[0]==0:
                censorship_all=censorship
            else:
                censorship_all=torch.cat((censorship_all,censorship),dim=0)
        c_index= ci(survtime_all.data,hazards_all.data,censorship_all.data)
        logging.info('Epoch:{}'.format(epoch))
        logging.info('Train Loss:{}; Train Ci:{}'.format(train_loss, c_index))
        train_losses.append(train_loss)
        print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch,train_loss,c_index))
        val_loss,val_ci=eval_model(model,val_loader,'Val',batch_size,cuda=True)
        val_losses.append(val_loss)
        if val_ci>best_ci:
            best_ci = val_ci
            best_epoch = epoch
            torch.save(model, f'{args.model_path}/{args.center}/{args.model}/{args.feat_label}_{args.seed}_{args.center_id}_fold{args.fold_num}_{date}_v{args.version}.pth')
            logging.info('Best epoch:{}; Best Val ci:{}'.format(best_epoch, best_ci))
    model_test=torch.load(f'{args.model_path}/{args.center}/{args.model}/{args.feat_label}_{args.seed}_{args.center_id}_fold{args.fold_num}_{date}_v{args.version}.pth')
    model_test=model_test.to(device)
    _,test_ci=eval_model(model_test,test_loader,'Test',batch_size,cuda=True)
    del model_test
    logging.info('Best epoch:{}; Val Ci:{}; Test Ci:{}'.format(best_epoch,best_ci,test_ci))


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def main(args):
    n_epochs = args.epochs
    batch_size = args.batch_size
    split_path = args.split_path
    seed = args.seed
    print("seed:{}".format(seed))
    test_center=args.center_id.split('wo_')[1]
    with open(split_path, 'rb') as file:
        split_file = pickle.load(file)
    train_list=[]
    val_list=[]
    for center in split_file.keys():
        if center!=test_center:
            train_list+=split_file[center][f'fold_{args.fold_num}']['train']
            val_list+=split_file[center][f'fold_{args.fold_num}']['val']
    with open(args.test_split_path, 'rb') as file:
        test_split_file = pickle.load(file)
    test_list = test_split_file[f'{test_center}']
    setup_seed(seed)
    feat_dim=769
    model = ABMIL_MH(feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {'num_workers': 0, } if device.type == "cuda" else {}
    if test_center=='center2':
        train_dataset=dataset(train_list,args.feat_path_center1,args.feat_path_center3,args.feat_path_center4,args.label_path)
        val_dataset=dataset(val_list,args.feat_path_center1,args.feat_path_center3,args.feat_path_center4,args.label_path)
        test_dataset=exdataset(test_list, args.feat_path_center2,test_center,f'{test_center}.pkl',f'{test_center}_pT.xlsx',
                                 f'{test_center}_dfs.xlsx')
    elif test_center=='center3':
        train_dataset = dataset(train_list, args.feat_path_center1, args.feat_path_center2, args.feat_path_center4,
                                args.label_path)
        val_dataset = dataset(val_list, args.feat_path_center1, args.feat_path_center2, args.feat_path_center4,
                              args.label_path)
        test_dataset = exdataset(test_list, args.feat_path_center3, test_center,
                                 f'{test_center}.pkl',
                                 f'{test_center}_pT.xlsx',
                                 f'{test_center}_dfs.xlsx')
    else:
        train_dataset = dataset(train_list, args.feat_path_center1, args.feat_path_center2, args.feat_path_center3,
                                args.label_path)
        val_dataset = dataset(val_list, args.feat_path_center1, args.feat_path_center2, args.feat_path_center3,
                              args.label_path)
        test_dataset = exdataset(test_list, args.feat_path_center4, test_center,
                                 f'{test_center}.pkl',
                                 f'{test_center}_pT.xlsx',
                                 f'{test_center}_dfs.xlsx')

    train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True,**kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False,**kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False,**kwargs)
    train_model(n_epochs, model, optimizer, batch_size,train_loader,val_loader,test_loader,cuda=True)





if __name__ == '__main__':
    try:
        main(args)
    except Exception as exception:
        raise

