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
#from utils import *
from modelDGC_4169 import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import scipy

from loaddata import processMutantFeaturesAndAdj
from loaddata import getWildAndMutIndex
from loaddata import processWildFeaturesAndAdj
from loaddata import get_residue_feature
from loaddata import get_ske_and_labels
from loaddata import get_mutation_indexes
from loaddata import get_adjs
from loaddata import standarize
from loaddata import standarize_residue
from loaddata import get_features
from loaddata import get_labels
from loaddata import standarize_train
from loaddata import getMutIndex 
from loaddata import standarize_residue
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--description', type=str, default='no description', help='descri.')
parser.add_argument('--dataset', type=str, default='ace2', help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
GRAD_CLIP = 5.
# NODES = 453  #s1131
# NODES = 500 #test1
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
# val_mut_features = mut_features[front_nodes:, :]
# val_mut_adjs = mut_adjs[length:]
# val_mut_nodes_number = mut_nodes_number[length:]

# val_wild_features = wild_features[front_nodes:, :]
# # val_wild_adjs = wild_adjs[nodes_number:]
# val_wild_adjs = val_mut_adjs
# val_wild_nodes_number = val_mut_nodes_number
# val_label = train_labels[length:]

# mut_features = mut_features[:front_nodes, :]
# mut_adjs = mut_adjs[:front_nodes]
# mut_nodes_number = mut_nodes_number[:length]
# wild_features = wild_features[:front_nodes, :]
# wild_adjs = wild_adjs[:length]
# wild_nodes_number = mut_nodes_number
# train_labels = train_labels[:length]

SetName = args.dataset
print('load '+SetName+'...')
test_mut_features, test_mut_adjs, test_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
test_mut_features = test_mut_features[:, :65]

test_wild_features, test_wild_adjs, test_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
test_wild_features = test_wild_features[:, :65]

test_residue_features = get_residue_feature(SetName)
test_labels = get_labels(SetName)
# test_labels = val_label
test_mutation_site = getMutIndex(SetName)

print('load done...')
# wild_array, mut_array = getMutIndex(SetName)
# residue_features = get_residue_feature(SetName)


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
# loss_fcn = torch.nn.BCELoss()
model = GCNIIppi(nfeat=mut_features[0].shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant,
                    NODES = NODES).to(device)
# model = torch.load('./new_model/model_150.pkl')
# model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
gbdt_save = [0 for i in range(len(mut_features))]
gbdt_save_test = [0 for i in range(test_mut_features.shape[0])]
print(args.description)
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
        # w_adj = wild_adjs[batch]
        # w_adj = w_adj.to(device)

        w_feature = w_feature.to(device)

        label = train_labels[batch].to(device)

        # wild_ = torch.LongTensor(wild_array[batch])
        # mut_ = torch.LongTensor(mut_array[batch])

        # wild_ = wild_.to(device)
        # mut_ = mut_.to(device)
        mutation_site = torch.LongTensor(train_mutaion_site[batch])
        mutation_site = mutation_site.to(device)

        residue = train_residue_features[batch]
        nodes = mut_nodes_number[batch]
        aux = residue.to(device)
        # aux = residue
        
        optimizer.zero_grad()
        output, gbdt = model(m_feature, m_adj, m_adj, w_feature, nodes, mutation_site, aux)
        loss_train = loss_fcn(output, label)

        loss_train.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, GRAD_CLIP)
        optimizer.step()
        
        loss_tra += loss_train.item()
    loss_counter = mut_features.shape[0]
    loss_tra/=loss_counter
    return loss_tra

def validation():
    model.eval()
    loss_test = 0.
    y_pred = []
    with torch.no_grad():
        for batch in range(0 ,test_mut_features.shape[0]):
            m_adj = test_mut_adjs[batch]
            m_adj = m_adj.to(device)
            m_feature = test_mut_features[batch].to(device)
            w_feature = test_wild_features[batch]
            # w_adj = test_wild_adjs[batch]
            # w_adj = w_adj.to(device)
            w_feature = w_feature.to(device)

            label = test_labels[batch].to(device)
            residue = test_residue_features[batch]
            aux = residue.to(device)
            # aux = residue
            nodes = test_nodes_number[batch]

            mutation_site = torch.LongTensor(test_mutation_site[batch])
            mutation_site = mutation_site.to(device)
            
            output, gbdt = model(m_feature, m_adj, m_adj, w_feature, nodes, mutation_site, aux)
            y_pred.append(output.item())
            # print(output)
            # print(gbdt)
            gbdt_save_test[batch] = gbdt.cpu()
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

    loss_train = train()
    # print(epoch)
    y_pred, loss_test = validation()
    y = test_labels.view(1, -1).squeeze(0)
    t2 = time.time()
    tt = t2 - t1
    t1 = t2
    if (epoch+1) == args.epochs // 10 * 9:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
        
    if SetName[0] == 'c' or SetName[0] == 'a':
        ken = kendall(y, y_pred)[0]
        print('epoch:',epoch,' time cost:',tt,' kandell:',ken)
        torch.save(model,'./model_170_seed_'+ str(args.seed) +'.pkl')
    # if rmsd < 1.93 and pearson>0.64:
    #     torch.save(model,'./new_model/model_'+str(epoch + 1)+'_'+str(round(rmsd,2))+'_'+str(round(pearson,3))+'.pkl')
    else:
        if SetName == 'delta_29':
            y_pred = y_pred[:24]
            y = y[:24]
            test_labels1 = test_labels[:24]
            print('alpha:',y_pred[2],'alpha*:',y_pred[-2],'delta:',y_pred[-1])
            print('beta:',y_pred[-3],'gamma:',y_pred[-4],'delta*:',y_pred[-5])
        else:
            test_labels1 = test_labels
        pearson = scipy.stats.pearsonr(y, y_pred)[0]
        rmsd = np.sqrt(mean_squared_error(test_labels1, y_pred))
        print('epoch:',epoch, 'train loss:', loss_train, 'val loss:',rmsd,' time cost:',tt,' pearson:', pearson)
        # if epoch == 169:
        
        # if pearson > prime_pcc and rmsd < prime_rmsd:
        #     p = 0
        #     torch.save(model,'./val_model/model.pkl')
        # else:
        #     p += 1
        #     if p == 30:
        #         break

        # if pearson > 0.64: torch.save(model,'./new_model_8/model_'+str(epoch + 1)+'_'+str(round(rmsd,2))+'_'+str(round(pearson,3))+'.pkl')
    # print('epoch:',epoch,' alpha:',y_pred[2],'alpha*:',y_pred[-1],'beta:',y_pred[-2],'gamma:',y_pred[-3],'delta:',y_pred[-4])

# for i in range(0, len(gbdt_save)):
#     gbdt_save[i] = gbdt_save[i].detach().numpy()
# for i in range(0, len(gbdt_save_test)):
#     gbdt_save_test[i] = gbdt_save_test[i].detach().numpy()
# np.save('prediction/' + SetName +'_pred.npy',y_pred)
# np.save('X_gbdt_train.npy', gbdt_save)
# np.save('X_gbdt_'+SetName+'.npy', gbdt_save_test)
# # np.save('delta_pred.npy', y_pred)
# torch.save(model,'./model/model_4169_427.pkl')