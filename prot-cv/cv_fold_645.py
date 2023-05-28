from __future__ import division
from __future__ import print_function
from tabnanny import check
import time
import random
import argparse
import numpy as np
from sklearn.utils.validation import check_array
from lr import PolynomialDecayLR
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import dataset
from modelDGC import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import scipy
import sys
from loaddata_for_cross import *
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')

parser.add_argument('--warmup_updates', type=int, default=50)
parser.add_argument('--tot_updates', type=int, default=300)
parser.add_argument('--peak_lr', type=float, default=4e-4)
parser.add_argument('--end_lr', type=float, default=1e-5)

parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=1024,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--NODES', type=int, default=500, help='Patience')
parser.add_argument('--dataset', default='ab645_500', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--fold', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
args = parser.parse_args() 
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
GRAD_CLIP = 5.
SetName = args.dataset
print('load '+SetName+'...')
NODES = args.NODES
mut_features, mut_adjs, mut_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
wild_features, wild_adjs, wild_nodes_number = processWildFeaturesAndAdj(SetName, NODES)

fea = np.load('./ab645/features.npy')
mut_features = mut_features[:, fea]
wild_features = wild_features[:, fea]
print(mut_features.shape)
print('load done...')
wild_array, mut_array = getWildAndMutIndex(SetName)
residue_feature = get_residue_feature(SetName)

ske, Y_gr = get_ske_and_labels(SetName)

loss_fcn = torch.nn.MSELoss()

def train(model, optimizer, lr_sch, mut_f, mut_adj, wild_f, wild_adj, labels, wild_array, mut_array, residue_features):
    model.train()
    loss_tra = 0
    y_pred_train = []
    for batch in range(0, len(mut_f)):
        
        m_adj = mut_adj[batch]
        m_adj = m_adj.to(device)
        m_feature = mut_f[batch].to(device)
        w_feature = wild_f[batch]
        w_adj = wild_adj[batch]
        w_adj = w_adj.to(device)
        w_feature = w_feature.to(device)

        label = labels[batch].to(device)
        wild_ = torch.LongTensor(wild_array[batch])
        mut_ = torch.LongTensor(mut_array[batch])
        wild_ = wild_.to(device)
        mut_ = mut_.to(device)

        residue = residue_features[batch]
        residue = residue.to(device)
        
        optimizer.zero_grad()
        output, _ = model(m_feature, m_adj, w_adj, w_feature, mut_, wild_, residue)
        y_pred_train.append(output.item())

        loss_train = loss_fcn(output, label)

        loss_train.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, GRAD_CLIP)
        optimizer.step()
        
        loss_tra += loss_train.item()
    lr_sch.step()

    loss_counter = len(mut_f)
    loss_tra/=loss_counter

    return loss_tra, y_pred_train

def validation(model, mut_f, mut_adj, wild_f, wild_adj, labels, wild_array, mut_array, residue_features):
    pred = []
    Y = []
    model.eval()
    loss_val = 0

    with torch.no_grad():
        for batch in range(0, len(mut_f)):

            adj = mut_adj[batch]
            m_adj = adj.to(device)
            m_feature = mut_f[batch].to(device)

            w_feature = wild_f[batch]
            w_adj = wild_adj[batch]

            w_adj = w_adj.to(device)
            w_feature = w_feature.to(device)

            wild_ = torch.LongTensor(wild_array[batch])
            mut_ = torch.LongTensor(mut_array[batch])
            wild_ = wild_.to(device)
            mut_ = mut_.to(device)

            residue = residue_features[batch]
            residue = residue.to(device)

            output, _ = model(m_feature, m_adj, w_adj, w_feature, mut_, wild_, residue)


            label = labels[batch].to(device)
            loss_test = loss_fcn(output, label)

            loss_val += loss_test.item()
            pred.append(output.item())
            Y.append(label.item())
    try:
        pcc = scipy.stats.pearsonr(Y, pred)
    except:
        pcc = [0.0, 0.1]
        np.save('x.npy',pred)
    loss_val /= len(mut_f)
    return loss_val, pcc[0], pred



from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle = True)
Loss = []

if SetName == 'ab645':
    import pickle as pkl
    f = open('./ab645/divided-folds.pkl', 'rb')
    divid = pkl.load(f)
    f.close()
    split_folds = []
    for key in divid.keys():
        split_folds.append((divid[key][0], divid[key][1]))
else:
    f = open('./' + SetName + '/divided-folds.txt')
    divid = f.read()
    divid = eval(divid)
    f.close()
    split_folds = divid

v = -1
##Y_pred = np.empty(645,)
for train_index, test_index in split_folds:
    v += 1
    if v != args.fold: continue
    print('validation:', v)
    mut_adjs_train = get_adjs(mut_adjs, train_index)
    mut_adjs_test = get_adjs(mut_adjs, test_index)
    mut_nodes_train, mut_nodes_test =  mut_nodes_number[train_index], mut_nodes_number[test_index]
    mut_features_train, mut_index_train = get_features(mut_features, mut_nodes_number, train_index)
    mut_features_test, wild_inddx_train = get_features(mut_features, mut_nodes_number, test_index)

    wild_adjs_train = get_adjs(wild_adjs, train_index)
    wild_adjs_test = get_adjs(wild_adjs, test_index)
    wild_nodes_train, wild_nodes_test = wild_nodes_number[train_index], wild_nodes_number[test_index]
    wild_features_train, wild_index_train = get_features(wild_features, wild_nodes_number, train_index)
    wild_features_test, wild_index_test = get_features(wild_features, wild_nodes_number, test_index)

    train_labels = torch.FloatTensor(Y_gr[train_index])
    train_labels = -train_labels.view(-1, 1)
    test_labels = torch.FloatTensor(Y_gr[test_index])
    test_labels = -test_labels.view(-1, 1)

    train_residue ,test_residue = residue_feature[train_index], residue_feature[test_index]

    train_w_array = get_mutation_indexes(wild_array, train_index)
    test_w_array = get_mutation_indexes(wild_array, test_index)
    train_m_array = get_mutation_indexes(mut_array, train_index)
    test_m_array = get_mutation_indexes(mut_array, test_index)

    #standarize 
    wild_features_train, mut_features_train ,wild_features_test, mut_features_test = standarize(mut_features, wild_features, mut_index_train, wild_index_train, wild_features_train, mut_features_train, wild_features_test, mut_features_test, NODES)
    train_residue, test_residue = standarize_residue(train_residue, test_residue)

    model = GCNIIppi(nfeat=mut_features_train[0].shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant,
                    NODES = NODES).to(device)

    optimizer = optim.Adam( model.parameters(), lr = args.peak_lr, weight_decay = args.wd)

    lr_scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=args.warmup_updates,
                tot_updates=args.tot_updates,
                lr=args.peak_lr,
                end_lr=args.end_lr,
                power=1.0,
    )

    t = time.time()
    loss_train_tag = 10.
    for epoch in range(args.epochs):
        loss_train, y_pred_train = train(model, optimizer, lr_scheduler, mut_features_train, mut_adjs_train, wild_features_train, wild_adjs_train, train_labels, train_w_array, train_m_array, train_residue)
        
        pearson_train = scipy.stats.pearsonr(train_labels.reshape(-1), y_pred_train)[0]
        t1 = time.time()
        tt = t1 - t
        t = t1
        print('Epoch:{:04d}'.format(epoch+1), 'train_loss:',loss_train,' time:',round(tt,2),' train_pcc:',pearson_train)

    loss_val, pcc, y_pred = validation(model, mut_features_test, mut_adjs_test, wild_features_test, wild_adjs_test, test_labels, test_w_array, test_m_array, test_residue)
    print('pcc:', pcc, ' val_loss:', loss_val)
    np.save('prediction/ab645/fold'+str(v)+'_index.npy', test_index)
    np.save('prediction/ab645/fold'+str(v)+'_pred.npy', -y_pred)