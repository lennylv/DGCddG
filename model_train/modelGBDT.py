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

    def forward(self, input, adj , h0 , lamda, alpha, l):
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
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.fc_aux = nn.Linear( 57 , nhidden)
        self.fc = nn.Linear(nhidden + 57, nhidden // 2)
##        self.fc = nn.Linear(nhidden, nclass)
        self.fc_2 = nn.Linear(nhidden // 2 , nhidden // 4)

        self.fc_3 = nn.Linear(nhidden // 4, 1)

    def forward(self, x, adj, wild_adj, wild_feature, nodes, mutaion_site, aux):
        _layers = []
        #print(x)
        x = F.dropout(x, self.dropout, training=self.training)
        wild_x = F.dropout(wild_feature, self.dropout, training=self.training)
        
##        layer_inner = self.act_fn(self.fcs[0](x))
##        wild_layer_inner = self.act_fn(self.fcs[0](wild_feature))
        #print(x)

##        for p in self.fcs[0].parameters():
##            print(p.grad)
        layer_inner = self.act_fn(self.fcs[0](x))
        wild_layer_inner = self.act_fn(self.fcs[0](wild_x))
        
        _layers.append(layer_inner)
        _layers.append(wild_layer_inner)

        for i,con in enumerate(self.convs):
##            for p in con.parameters():
##                print(p)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))

            wild_layer_inner = F.dropout(wild_layer_inner, self.dropout, training=self.training)
            wild_layer_inner = self.act_fn(con(wild_layer_inner, wild_adj, _layers[1], self.lamda, self.alpha, i+1))
    
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        wild_layer_inner =  F.dropout(wild_layer_inner, self.dropout, training=self.training)
        #select the atoms-description of wild and mutant
        #print(mut_array)
        #print(wild_layer_inner)
        # print(mutaion_site)

        layer_inner = layer_inner[mutaion_site]
        wild_layer_inner = wild_layer_inner[mutaion_site]

        # layer_inner = layer_inner[:nodes, :]
        # wild_layer_inner = wild_layer_inner[:nodes, :]

##        layer_inner = layer_inner.transpose(1, 0)
##        wild_layer_inner = wild_layer_inner.transpose(1, 0)
##        layer_inner = nn.MaxPool1d()
        
        # layer_inner = torch.max(layer_inner, 0)[0]
        # wild_layer_inner = torch.max(wild_layer_inner, 0)[0]
        
        layer_inner = torch.mean(layer_inner, 0)
        wild_layer_inner = torch.mean(wild_layer_inner, 0)

        # layer_inner = torch.sum(layer_inner[:nodes, :], 0)
        # wild_layer_inner = torch.sum(wild_layer_inner[:nodes, :], 0)

        # _ = torch.max(layer_inner)
        # _ = torch.max(wild_layer_inner)

        # print(layer_inner.shape, wild_layer_inner.shape)
##        layer = torch.cat((layer_inner,wild_layer_inner), 0)
##        layer_inner = layer.flatten()
        
        layer_inner_differ = layer_inner - wild_layer_inner
        
        # layer_inner_differ = torch.cat((layer_inner,wild_layer_inner,layer_inner_differ), 0)
        
        gbdt = torch.cat((layer_inner,wild_layer_inner,layer_inner_differ), 0)
        #layer_inner = self.fc(layer_inner)
        # fully connect
##        layer_inner = self.fcs[-1](layer_inner)
##        gbdt = layer_inner
##        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        # print(layer_inner_differ)
        # aux_fc = self.fc_aux(aux.float())
        # layer_inner_differ = torch.cat((layer_inner_differ, aux.float()), 0)

        # layer_inner_differ = aux_fc
        layer_inner_differ = self.act_fn(layer_inner_differ)
        
        # gbdt = torch.cat((layer_inner,wild_layer_inner,layer_inner_differ, residue_fc), 0)
        # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
##        layer_inner = self.fc(layer_inner)
        layer_inner_differ = torch.cat((layer_inner_differ, aux.float()), 0)
        layer_inner_differ = self.fc(layer_inner_differ)
        layer_inner_differ = F.dropout(layer_inner_differ, self.dropout, training=self.training)

        # layer_inner_differ = torch.cat((layer_inner_differ, residue_fc), 0)
        
        output = self.act_fn(layer_inner_differ)
        
        output = F.dropout(output, self.dropout, training=self.training)
        # gbdt = output
        output = self.fc_2(output)
        
        # output = self.sig(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.act_fn(output)
        output = self.fc_3(output)
        # output = self.sig(output)

        return output, gbdt

if __name__ == '__main__':
    pass
