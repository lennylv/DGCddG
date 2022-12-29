import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, NODES):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        self.convs_wild = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        # for _ in range(nlayers):
        #     self.convs_wild.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.NODES = NODES
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.fc_aux = nn.Linear( 2, 4 )
        self.fc_cross = nn.Linear( 600, nhidden )

        self.fc = nn.Linear(nhidden , nhidden // 32)
        self.fc_2 = nn.Linear(nhidden // 32 + 2, 1)

    def forward(self, x, adj, wild_adj, wild_feature, nodes, mutaion_site, aux):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        wild_x = F.dropout(wild_feature, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        wild_layer_inner = self.act_fn(self.fcs[0](wild_x))

        _layers.append(layer_inner)
        _layers.append(wild_layer_inner)
        
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i+1))
            wild_layer_inner = F.dropout(wild_layer_inner, self.dropout, training=self.training)
            wild_layer_inner = self.act_fn(con(wild_layer_inner, wild_adj, _layers[1], self.lamda, self.alpha, i+1))  

        # for i, con in enumerate(self.convs_wild):

    
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        wild_layer_inner =  F.dropout(wild_layer_inner, self.dropout, training=self.training)

        layer_inner_m = layer_inner[mutaion_site]
        wild_layer_inner_m = wild_layer_inner[mutaion_site]
        
        layer_inner_sum = torch.sum(layer_inner_m, 0)
        wild_layer_inner_sum = torch.sum(wild_layer_inner_m, 0)

        layer_inner_mean = torch.mean(layer_inner_m, 0)
        wild_layer_inner_mean = torch.mean(wild_layer_inner_m, 0)
        
        layer_inner_differ_sum =  layer_inner_sum - wild_layer_inner_sum
        
        layer_inner_differ_mean = - layer_inner_mean + wild_layer_inner_mean
        layer_inner_differ = torch.cat((layer_inner_differ_sum, layer_inner_differ_mean,), 0)

        k = [i for i in range(17)]
        q = [i for i in range(25, 65)]
        k.extend(q)
        # aux = aux.float()[-3].unsqueeze(0)
        # -4: secondary structure
        # -3: asa
        # -2, -1: phi, psi
        aux = aux.float()[-4:-2]
        aux = aux * len(mutaion_site)
        # aux = self.act_fn(aux)
        
        layer_inner_differ = self.fc( layer_inner_differ_sum )

        output = self.act_fn( layer_inner_differ )
        output= F.dropout(output, self.dropout, training=self.training)
        output = torch.cat((output, aux), 0)
        output = self.fc_2( output )

        return output, layer_inner_differ

if __name__ == '__main__':
    pass
