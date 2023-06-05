from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
from sklearn.utils.validation import check_array
from scipy.stats import pearsonr as pear
from lr import PolynomialDecayLR
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import dataset
from utils import *
from modelDGC import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import scipy
import sys
from loaddata_for_cross import processMutantFeaturesAndAdj
from loaddata_for_cross import getWildAndMutIndex
from loaddata_for_cross import processWildFeaturesAndAdj
from loaddata_for_cross import get_residue_feature
from loaddata_for_cross import get_ske_and_labels
from loaddata_for_cross import get_mutation_indexes
from loaddata_for_cross import get_adjs
from loaddata_for_cross import standarize
from loaddata_for_cross import standarize_residue
from loaddata_for_cross import get_features
from loaddata_for_cross import get_labels
from loaddata_for_cross import standarize_train
from loaddata_for_cross import getMutIndex
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')

parser.add_argument('--warmup_updates', type=int, default=50)
parser.add_argument('--tot_updates', type=int, default=30)
parser.add_argument('--peak_lr', type=float, default=2e-4)
parser.add_argument('--end_lr', type=float, default=1e-5)
parser.add_argument('--validation', type=int, default=5)

parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=2, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

parser.add_argument('--patient', type=int, default=50, help='Patience')
parser.add_argument('--dataset', default='s4169', help='dateset')
parser.add_argument('--dev', type=int, default=2, help='device id')
parser.add_argument('--Fold', type=int, default=1, help='device id')

parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--NODES', type=int, default=1000, help='Patience')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
# GRAD_CLIP = 5.
GRAD_CLIP = 1.

NODES = args.NODES #s1131
# NODES = 653

SetName = args.dataset
print('load '+SetName+'...')
mut_features, mut_adjs, mut_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
wild_features, wild_adjs, wild_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
# NODES = max(mut_nodes_number)

# q = [0,1,2,3,4,5]
# q = [i for i in range(45, 65)]
# k = [i for i in range(0, 15)]
# k.extend(q)
mut_features = mut_features[:, :65]
wild_features = wild_features[:, :65]

print('load done...')
wild_array, mut_array = getWildAndMutIndex(SetName)
residue_feature = get_residue_feature(SetName)
# residue_feature[:, [2,3]]=0.
# residue_feature[:, [0,1,2,3]]=0.
ske, Y_gr = get_ske_and_labels(SetName)
# Y_gr = -Y_gr

loss_fcn = torch.nn.MSELoss()

def train(lr_sch, model, optimizer, mut_f, mut_adj, wild_f, wild_adj, labels, wild_array, mut_array, residue_features):
    model.train()
    loss_tra = 0
    # print(labels.shape)
    y_pred_train = []
    y_label_train = []
    for batch in range(0, len(mut_f)):
        
        m_adj = mut_adj[batch]
        m_adj = m_adj.to(device)
        m_feature = mut_f[batch].to(device)
        w_feature = wild_f[batch]
        w_adj = wild_adj[batch]
        w_adj = w_adj.to(device)
        w_feature = w_feature.to(device)

        label = labels[batch].to(device)
        # print(label)
        wild_ = torch.LongTensor(wild_array[batch])
        mut_ = torch.LongTensor(mut_array[batch])
        wild_ = wild_.to(device)
        mut_ = mut_.to(device)

        residue = residue_features[batch]
        residue = residue.to(device)
        
        optimizer.zero_grad()
        output, check = model(m_feature, m_adj, w_adj, w_feature, mut_, wild_, residue)
        # if batch == 0:
        #     print(label)
        #     print('output:',output.shape,'labels:',label.shape)
        # if True in torch.isnan(check):
        

        loss_train = loss_fcn(output, label)
        loss_train.backward()

        y_pred_train.append(output.item())
        y_label_train.append(label.item())
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, GRAD_CLIP)
        optimizer.step()
        
        loss_tra += loss_train.item()
    lr_sch.step()

    loss_counter = len(mut_f)
    loss_tra/=loss_counter
    # x_for_gbdt_train = x_for_gbdt_train.view(-1, args.hidden + 42)
    # x_for_gbdt_train = []
    return loss_tra, y_pred_train, y_label_train

def validation(model, mut_f, mut_adj, wild_f, wild_adj, labels, wild_array, mut_array, residue_features, mean, std):
    X = []
    Y = []
    x_for_gbdt_test = []
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

            output, x_for_gbdt_test_item = model(m_feature, m_adj, w_adj, w_feature, mut_, wild_, residue)
            x_for_gbdt_test.append(x_for_gbdt_test_item.cpu().numpy())
            label = labels[batch].to(device)

            # output = output * std + mean
            loss_test = loss_fcn(output, label)

            loss_val += loss_test.item()
            X.append(output.item())
            Y.append(label.item())
    try:
        pcc = scipy.stats.pearsonr(Y, X)
    except:
        pcc = [0.0, 0.1]
        np.save('x.npy',X)
    loss_val /= len(mut_f)
    return loss_val, pcc[0], X, x_for_gbdt_test



from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle = True)
Y_pred = np.empty(len(Y_gr),)
validation_time = 0
if SetName == 'ab645':
    import pickle as pkl
    f = open('./ab645/divided-folds.pkl', 'rb')
    divid = pkl.load(f)
    f.close()
    split_folds = []
    for key in divid.keys():
        split_folds.append((divid[key][0], divid[key][1]))

elif SetName == 's1131':
    f = open('./s1131/divided_folds.txt')
    divid = f.read()
    f.close()
    divid = eval(divid)
    split_folds = divid
else:
    f = open('./' + SetName + '/divided-folds-hgroup.txt')
    print('divided-folds-hgroup.txt')
    divid = f.read()
    f.close()
    divid = eval(divid)
    split_folds = divid
    # split_folds = kf.split(ske)

step = [5, 2, 1, 4, 3]
validation_times=[args.dev*2+1,args.dev*2+2]
#validation_times=[2,]
for train_index, test_index in split_folds:
# for s in step:
    validation_time += 1
    # if validation_time != 5: continue
    # validation_time = s
    print('validation cross number:', validation_time)
    #test_index = train_index[-730:]
    #train_index = train_index[:-730]

    print(len(train_index), len(test_index))
    if validation_time != args.Fold:continue
    # train_index, test_index = split_folds[s-1]
    test_index_copy = test_index
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

    train_labels, test_labels = Y_gr[train_index], Y_gr[test_index]
    mean, std = np.mean(train_labels), np.std(train_labels)
    # train_labels = (train_labels - mean) / (std + 1e-5)
    train_labels = torch.FloatTensor(train_labels)
    train_labels = train_labels.view(-1, 1)

    test_labels = torch.FloatTensor(Y_gr[test_index])
    test_labels = test_labels.view(-1, 1)

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
                    NODES=NODES).to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)

    lr_scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=args.warmup_updates,
                tot_updates=args.tot_updates,
                lr=args.peak_lr,
                end_lr=args.end_lr,
                power=1.0,
    )
    # train_loss = []
    bad_count = 0
    best_epoch = 0
    best_val = 100
    best_pcc = 0
    t = time.time()
    patient = 0
    loss_train_eval = 100.
    
    for epoch in range(args.epochs):
        loss_train, y_pred_train, y_label_train = train(lr_scheduler, model, optimizer, mut_features_train, mut_adjs_train, wild_features_train, wild_adjs_train, train_labels, train_w_array, train_m_array, train_residue)
        # train_loss.append(loss_tra)
        
        # if loss_train<loss_train_eval:
        t1 = time.time()
        tt = t1 - t
        t = t1
        print('Epoch:{:04d}'.format(epoch+1), 'train_pcc:', pear(y_pred_train, y_label_train)[0], round(loss_train,3),' time:',round(tt,2),'s',)
    # if True:
    loss_val, pcc, y_pred, gbdt = validation(model, mut_features_test, mut_adjs_test, wild_features_test, wild_adjs_test, test_labels, test_w_array, test_m_array, test_residue, mean, std)
    print('testloss:', loss_val,' pcc:',pcc)

    np.save('prediction/4169_eval/4169_fold'+str(validation_time)+'.npy', y_pred)
    # np.save('prediction/4169_eval/4169_feature'+str(validation_time)+'.npy', gbdt)
    np.save('prediction/4169_eval/4169_testindex_'+str(validation_time)+'.npy',test_index)
