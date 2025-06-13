import torch.nn as nn
from torch.nn.functional import normalize
import torch
import copy
import sys


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):

        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)  
        self.copy_encoder = copy.deepcopy(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)  
        self.feature_cross_v_dec = nn.ModuleList([MLP(feature_dim, feature_dim).to(device) for i in range(view)])
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )


        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.view = view

        self.cl = ContraLoss(0.5)

    def forward(self, xs):
        hs = []
        qs = []
        xcs = []
        zs = []
        zks = []
        for v in range(self.view):
            x = xs[v]                                                    
            z = self.encoders[v](x)
            z_k = self.copy_encoder[v](x)                                      
            h = normalize(self.feature_contrastive_module(z), dim=1)     
            q = self.label_contrastive_module(z)                           
            xc = self.feature_cross_v_dec[v](z)                                
            hs.append(h)                                                
            zs.append(z)  
            zks.append(z_k)                                               
            qs.append(q)                                            
            xcs.append(xc)                                               
        return hs, qs, xcs, zs, zks

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]                                                    
            z = self.encoders[v](x)                                       
            zs.append(z)                                                 
            h = self.feature_contrastive_module(z)                       
            hs.append(h)                                                 
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]                                                   
            z = self.encoders[v](x)                                      
            q = self.label_contrastive_module(z)                         
            pred = torch.argmax(q, dim=1)                                
            qs.append(q)                                                 
            preds.append(pred)                                           
        return qs, preds
    
    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = Norm(z)
        # clamp将张量中的所有值限制为不小于给定的最小值
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x

Norm = nn.functional.normalize

class ContraLoss(nn.Module):
    def __init__(self, temp=1.0):
        super(ContraLoss, self).__init__()
        self.temp = temp

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = Norm(x_q)
        x_k = Norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temp)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss