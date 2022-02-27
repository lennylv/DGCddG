from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from utils import *
from modelGBDT_4169 import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import scipy

from loaddata import *
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=170, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=1024,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--dataset', type=str, default='testset1', help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
GRAD_CLIP = 5.
NODES = 500
SetName = 's4169'
print('load '+SetName+'...')
mut_features, mut_adjs, mut_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
mut_features= mut_features[:,:65]

wild_features, wild_adjs, wild_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
wild_features = wild_features[:, :65]

train_residue_features = get_residue_feature(SetName)
train_labels = get_labels(SetName)
train_mutaion_site = getMutIndex(SetName)

length = len(mut_nodes_number) // 5 * 4
front_nodes = sum(mut_nodes_number[:length])

SetName = args.dataset
print('load '+SetName+'...')
test_mut_features, test_mut_adjs, test_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
test_mut_features = test_mut_features[:, :65]

test_wild_features, test_wild_adjs, test_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
test_wild_features = test_wild_features[:, :65]

test_residue_features = get_residue_feature(SetName)
test_labels = get_labels(SetName)
test_mutation_site = getMutIndex(SetName)

print('load done...')


#train features split N * NODES * features
indexes = [i for i in range(0, len(wild_nodes_number))]
wild_features_splits = get_features(wild_features, wild_nodes_number, indexes)
mut_features_splits = get_features(mut_features, mut_nodes_number, indexes)

indexes = [i for i in range(0, len(test_nodes_number))]
test_wild_features_splits = get_features(test_wild_features, test_nodes_number, indexes)
test_mut_features_splits = get_features(test_mut_features, test_nodes_number, indexes)

wild_features, mut_features, test_wild_features, test_mut_features = standarize(wild_features, mut_features, wild_features_splits, mut_features_splits, test_wild_features_splits, test_mut_features_splits, NODES)

train_residue_features, test_residue_features = standarize_residue(train_residue_features, test_residue_features)

loss_fcn = torch.nn.MSELoss()
model = GCNIIppi(nfeat=mut_features[0].shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant,
                    NODES = NODES).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
print(mut_features.shape)
print(test_mut_features.shape)
def train():
    model.train()
    loss_tra = 0
    # print(labels.shape)
    for batch in range(0 ,len(mut_features)):

        m_adj = mut_adjs[batch]
        m_adj = m_adj.to(device)
        m_feature = mut_features[batch].to(device)
        w_feature = wild_features[batch]
        w_feature = w_feature.to(device)

        label = train_labels[batch].to(device)
        mutation_site = torch.LongTensor(train_mutaion_site[batch])
        mutation_site = mutation_site.to(device)

        residue = train_residue_features[batch]
        nodes = mut_nodes_number[batch]
        aux = residue.to(device)
        optimizer.zero_grad()
        output = model(m_feature, m_adj, m_adj, w_feature, nodes, mutation_site, aux)
        loss_train = loss_fcn(output, label)

        loss_train.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, GRAD_CLIP)
        optimizer.step()
        
        loss_tra += loss_train.item()
    loss_counter = mut_features.shape[0]
    loss_tra/=loss_counter
    return loss_tra

def test():
    model.eval()
    loss_test = 0.
    y_pred = []
    with torch.no_grad():
        for batch in range(0 ,test_mut_features.shape[0]):
            m_adj = test_mut_adjs[batch]
            m_adj = m_adj.to(device)
            m_feature = test_mut_features[batch].to(device)
            w_feature = test_wild_features[batch]
            w_feature = w_feature.to(device)

            label = test_labels[batch].to(device)
            residue = test_residue_features[batch]
            aux = residue.to(device)

            nodes = test_nodes_number[batch]

            mutation_site = torch.LongTensor(test_mutation_site[batch])
            mutation_site = mutation_site.to(device)
            
            output = model(m_feature, m_adj, m_adj, w_feature, nodes, mutation_site, aux)
            y_pred.append(output.item())
            loss = loss_fcn(output, label)
            loss_test += loss.item()
    loss_test /= test_mut_features.shape[0]
    return y_pred, loss_test

from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau as kendall
t1 = time.time()
prime_pcc = 0.
prime_rmsd = 10.
p = 0
for epoch in range(0, args.epochs):

    loss_train, y_pred = train()
    y = train_labels.view(1, -1).squeeze(0)
    t2 = time.time()
    tt = t2 - t1
    t1 = t2

    pearson = scipy.stats.pearsonr(y, y_pred)[0]
    print('epoch:',epoch, 'train loss:', loss_train,' time cost:',tt,' pearson:', pearson)

y_, _ = test()
rmse = np.sqrt(mean_squared_error(y_, test_labels.view(1, -1).squeeze(0)))
pearson = scipy.stats.pearsonr(y_, test_labels.view(1, -1).squeeze(0))[0]
print('rmse: ',rmse, ' pearson:', pearson[0])
torch.save(model,'./model_4169.pkl')