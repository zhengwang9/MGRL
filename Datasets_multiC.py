import itertools
import os
import pickle
import random
import collections
import h5py
import numpy
import pandas as pd
from torch.utils.data import Dataset
import torch
class Survival_Dataset_MultiCenter_TtumorDFS(Dataset):
    '''
    center 1
    feat_path 1: center 2
    feat_path 2: center 3

    '''
    def __init__(self, id_list,feat_path_center1,feat_path_exc1,feat_path_exc2,label_path,tumor_index_file='/data19/wz/annotation/whole_center.pkl',T_file='/data19/wz/processed_excel/whole_pT.xlsx'):
        super(Survival_Dataset_MultiCenter_TtumorDFS, self).__init__()
        self.label_file = pd.read_excel(label_path)
        self.T_file=pd.read_excel(T_file)
        self.feat_path_center1 =feat_path_center1
        self.feat_path_exc1=feat_path_exc1
        self.feat_path_exc2=feat_path_exc2
        self.feat_id_list=[]
        self.id_list=id_list

        with open(tumor_index_file,'rb') as file:
            self.tumor_index=pickle.load(file)

        self.feat_id_list=os.listdir(self.feat_path_center1)+os.listdir(self.feat_path_exc1)+os.listdir(self.feat_path_exc2)
        self.centers=[1]*len(os.listdir(self.feat_path_center1))+[2]*len(os.listdir(self.feat_path_exc1))+[3]*len(os.listdir(self.feat_path_exc2))

    def __len__(self):
        return len(self.feat_id_list)

    def __getitem__(self, idx):
        feat_id=self.feat_id_list[idx]
        center=self.centers[idx]
        for id in self.id_list:
            if id in feat_id:
                excel_id=id

        id_index = self.label_file['id'][self.label_file['id'] == excel_id].index[0]
        censorship=int(self.label_file['status'][id_index])
        survtime=float(self.label_file['DFS'][id_index])
        t_index = self.T_file['id'][self.T_file['id'] == excel_id].index[0]
        T=int(self.T_file['T'][t_index])
        if center==1:
            feat_file=os.path.join(self.feat_path_center1,feat_id)
        elif center==2:
            feat_file=os.path.join(self.feat_path_exc1,feat_id)
        elif center==3:
            feat_file=os.path.join(self.feat_path_exc2,feat_id)
        with h5py.File(feat_file, 'r') as f:
            feat = f['features'][:]
            coords=f['coords'][:]
        #用index筛选feat
        index=self.tumor_index[feat_id.split('.h5')[0]]
        # index=sorted(index)
        feat=torch.from_numpy(feat[index])
        if T==3:
            T_vector=torch.zeros((feat.shape[0],1))
        else:
            T_vector=torch.ones((feat.shape[0],1))
        feat=torch.cat((feat,T_vector),dim=1)
        return {'feat':feat,'censorship':censorship,'survtime':survtime,'id':excel_id,'center':center,'feat_id':feat_id,'index':index,'coords':coords}

class Survival_Dataset_SingleCenter_external_TannoDFS(Dataset):
    def __init__(self, id_list, feat_path, center,tumor_index_file='',T_file='',dfs_info=''):
        super(Survival_Dataset_SingleCenter_external_TannoDFS, self).__init__()
        self.label_file = pd.read_excel(dfs_info)
        self.T_file=pd.read_excel(T_file)
        self.feat_path = feat_path
        self.feat_id_list = []
        self.id_list = id_list
        self.center = center
        with open(tumor_index_file, 'rb') as file:
            self.tumor_index = pickle.load(file)
        self.feat_id_list = os.listdir(self.feat_path)
    def __len__(self):
        return len(self.feat_id_list)

    def __getitem__(self, idx):
        feat_id = self.feat_id_list[idx]
        excel_id=''
        for index,id in enumerate(self.id_list):
            if id in feat_id:
                excel_id = id
        id_index = self.label_file['id'][self.label_file['id'] == excel_id].index[0]
        censorship = int(self.label_file['status'][id_index])
        survtime = float(self.label_file['DFS'][id_index])
        t_index = self.T_file['id'][self.T_file['id'] == excel_id].index[0]
        T=int(self.T_file['T'][t_index])
        feat_file = os.path.join(self.feat_path, feat_id)
        with h5py.File(feat_file, 'r') as f:
            feat = f['features'][:]
            coords=f['coords'][:]
        # 用index筛选feat
        index = self.tumor_index[feat_id.split('.h5')[0]]
        # index=sorted(index)
        feat = torch.from_numpy(feat[index])
        if T == 3:
            T_vector = torch.zeros((feat.shape[0], 1))
        else:
            T_vector = torch.ones((feat.shape[0], 1))
        feat = torch.cat((feat, T_vector), dim=1)
        return {'feat': feat, 'censorship': censorship, 'survtime': survtime, 'id': excel_id,'feat_id':feat_id,'center':self.center,'coords':coords,'index':index}







