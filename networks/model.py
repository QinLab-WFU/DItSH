import torch
import torch.nn as nn
from torchvision.models import alexnet
# from test import FusionModel , SpatialAttention
from networks.relative_similarity import RelativeSimilarity
from networks.ca_net import *
from utils.attention_zoom import *
class RelaHash(nn.Module):
    def __init__(self,
                 nbit, nclass, batchsize,
                 init_method='M',
                 pretrained=True, freeze_weight=False,
                 device='cuda',
                 **kwargs):
        super(RelaHash, self).__init__()

        # self.backbone = FusionModel(hash_bit=nbit)
        self.backbone = CANet(bit=nbit,nclass=nclass)

        self.hash_fc = nn.Sequential(
            nn.BatchNorm1d(self.backbone.num_ftrs // 2 * 3, affine=True),
            nn.Linear(self.backbone.num_ftrs // 2 * 3, self.backbone.feature_size),
            nn.BatchNorm1d(self.backbone.feature_size, affine=True),
            nn.ELU(inplace=True),
            nn.Linear(self.backbone.feature_size, nbit),
        )
        # nn.init.normal_(se.weight, std=0.01)

        self.relative_similarity = RelativeSimilarity(nbit, nclass, batchsize, init_method=init_method, device=device)

    def get_hash_params(self):
        return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())
    
    def get_backbone_params(self):
        return self.backbone.get_features_params()
    
    def get_centroids(self):
        return self.relative_similarity.centroids
        
    def forward(self, x,if_zoom=False):
        if if_zoom == True:
            _,_,_,y_zoom,_ = self.backbone(x)
            return y_zoom
        f11, f22, f33, y33, feats = self.backbone(x)
        f44 = torch.cat((f11, f22, f33), -1)
        z = self.hash_fc(f44)
        logits = self.relative_similarity(z) # hashcode
        return logits, z , y33 ,feats



# net = RelaHash(nclass=10 , nbit= 16 , batchsize= 32)
# print(net)