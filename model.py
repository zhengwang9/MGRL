import sys

import torch
from model.model_utils import *
################################
# Attention MIL Implementation #
################################
import pickle
class ABMIL_MH(nn.Module):

    def __init__(self, feat_dim,size_arg = "small", dropout=0.25, n_classes=1):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(ABMIL_MH, self).__init__()
        self.size_dict_path = {"small": [feat_dim, 512, 256], "big": [2048, 512, 384]}
        ##############
        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier_center1 = nn.Linear(size[2], n_classes)
        self.classifier_center2 = nn.Linear(size[2], n_classes)
        self.classifier_center3 = nn.Linear(size[2], n_classes)
    def relocate(self):
        self.attention_net=self.attention_net.to(device[0])
        self.rho = self.rho.to(device[0])
        self.classifier = self.classifier.to(device[0])

    def forward(self, x,center):
        if len(x.shape) == 3:
            x = x.squeeze(0)
        A, h_path = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path # [256] vector
        if center==1:
            logits_1  = self.classifier_center1(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        elif center==2:
            logits_2  = self.classifier_center2(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        else:
            logits_3  = self.classifier_center3(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        return torch.sigmoid(logits_1),torch.sigmoid(logits_2),torch.sigmoid(logits_3)

