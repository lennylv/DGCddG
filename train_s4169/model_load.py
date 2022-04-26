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
from modelGBDT import *
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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dense',type=str, default='16', help='dateset')
parser.add_argument('--dev', type=int, default=2, help='device id')
parser.add_argument('--fold', type=int, default=1, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--modelPath', type=str, default='..', help='evaluation on test set.')
parser.add_argument('--dataset', type=str, default='ace2', help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev) if torch.cuda.is_available() else 'cpu'
device = torch.device(cudaid)
GRAD_CLIP = 5.
NODES = 604
SetName = 's4169'
if SetName == 's4169':
    NODES = 500
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


train_residue_features, test_residue_features = standarize_residue(train_residue_features[:,-4:], test_residue_features[:, -4:])
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

model = torch.load('./model_4169_save.pkl', map_location=device)
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

y_pred, loss_test = validation()

# Here, because the label values of S4169 are mutant - wild
# take 'protein 1A22 ; chain A ; wild C ; id 171 ; mutant A' as an example, See table s1131 row 2; table s4169 row 24
# except for S4169, all datasets are wild - mutant
# so here, we need a negative
y_pred = -y_pred
y = test_labels.view(1, -1).squeeze(0)

if SetName == 'delta_29':
    y_pred = y_pred[:24]
    y = y[:24]
    test_labels = test_labels[:24]
    print('alpha:',y_pred[2],'alpha*:',y_pred[-2],'delta:',y_pred[-1])
    print('beta:',y_pred[-3],'gamma:',y_pred[-4],'delta*:',y_pred[-5])

i = [j for j in range(len(y_pred)) if np.isnan(y_pred[j])]
y_pred = np.array(y_pred)
y_pred[i] = 0.
# np.save('prediction/'+ SetName +'.npy',y_pred)
pearson = scipy.stats.pearsonr(y, y_pred)[0]
ken = kendall(y, y_pred)[0]
rmsd = np.sqrt(mean_squared_error(test_labels, y_pred))
print('loss:',rmsd,' pearson:', pearson, ' kandell:',ken)



