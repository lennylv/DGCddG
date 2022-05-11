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
# from utils import *
from modelGBDT import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import scipy
from loaddata import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--model_type', type=str, default='1', help='evaluation on test set.')
parser.add_argument('--dataset', type=str, default='testset1', help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev) if torch.cuda.is_available() else 'cpu'
device = torch.device(cudaid)
GRAD_CLIP = 5.
NODES = 604
SetName = 'trainset'
# if SetName == 's4169':
#     NODES = 500
print('load '+SetName+'...')
mut_features, mut_adjs, mut_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
mut_features= mut_features[:, :65]

wild_features, wild_adjs, wild_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
wild_features = wild_features[:, :65]

train_residue_features = get_residue_feature(SetName, stand=False)
train_labels = get_labels(SetName)
train_mutaion_site = getMutIndex(SetName)



SetName = args.dataset
print('load '+SetName+'...')
test_mut_features, test_mut_adjs, test_nodes_number = processMutantFeaturesAndAdj(SetName, NODES)
test_mut_features = test_mut_features[:, :65]

test_wild_features, test_wild_adjs, test_nodes_number = processWildFeaturesAndAdj(SetName, NODES)
test_wild_features = test_wild_features[:, :65]

test_residue_features = get_residue_feature(SetName, stand=False)
test_labels = get_labels(SetName)
test_mutation_site = getMutIndex(SetName)


print('load done...')

indexes = [i for i in range(0, len(wild_nodes_number))]
wild_features_splits = get_features(wild_features, wild_nodes_number, indexes)
mut_features_splits = get_features(mut_features, mut_nodes_number, indexes)

indexes = [i for i in range(0, len(test_nodes_number))]
test_wild_features_splits = get_features(test_wild_features, test_nodes_number, indexes)
test_mut_features_splits = get_features(test_mut_features, test_nodes_number, indexes)

wild_features, mut_features, test_wild_features, test_mut_features = standarize(wild_features, mut_features, wild_features_splits, mut_features_splits, test_wild_features_splits, test_mut_features_splits, NODES)


loss_fcn = torch.nn.MSELoss()
gbdt_save_test = [0 for i in range(test_mut_features.shape[0])]

print(mut_features.shape)
print(test_mut_features.shape)

model_path = args.model_type
if model_path == '1':
    model_path = './model_stand.pkl'
    train_residue_features, test_residue_features = standarize_residue(train_residue_features[:,-4:], test_residue_features[:, -4:])
else:
    model_path = './model_1470.pkl'

model = torch.load(model_path, map_location=device)


model = model.to(device)
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
            w_feature = w_feature.to(device)

            label = test_labels[batch].to(device)
            residue = test_residue_features[batch]
            aux = residue.to(device)
            # aux = residue
            nodes = test_nodes_number[batch]

            mutation_site = torch.LongTensor(test_mutation_site[batch])
            mutation_site = mutation_site.to(device)
            
            output = model(m_feature, m_adj, m_adj, w_feature, nodes, mutation_site, aux)
            y_pred.append(output.item())
            loss = loss_fcn(output, label)
            loss_test += loss.item()
    loss_test /= test_mut_features.shape[0]
    return y_pred, loss_test


y_pred, loss_test = validation()
print('predicted ddg:', y_pred)