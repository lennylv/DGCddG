# from crossvalidation import train
from tokenize import single_quoted
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import KFold
import pickle
import scipy
import sys
import random
import xgboost as xgb
from scipy.stats import kendalltau as kendall

train_list = np.load('trainset/Y.npy')
train_multiidx = np.load('trainset/multiidx.npy')
train_singleidx = [i for i in range(len(train_list)) if i not in train_multiidx]
train_y = np.load('trainset/Y.npy')[:, -1].astype(float)

train_x1 = np.load('trainset/34_feature.npy')
train_x2 = np.load('trainset/environment.npy')
train_x3 = np.load('trainset/normal_feature.npy')
importance = np.load('importance_onehot.npy')[:]

train_x5 = np.load('250epoch_model_graph_features/X_gbdt_train.npy')[:, -128:][:, importance]


select_idx = [1,2,3,4,]
normal_idx = [4,15, 8,19, 9,20, 16, 2,13, 24,26,27,30,31,18]
one_idx = [i for i in range(0, 34) if i not in [1,3,4,6,7,8,14,16,18,19,20,25,27,30,31,32,33]]
environ_idx =[i for i in range(0,25) if i not in [0,3,12,16,18,21]]

train_x = np.concatenate((train_x1[:, one_idx], train_x2[:, environ_idx], train_x3[:,normal_idx]), 1)
for index in train_multiidx :
    item = train_list[index]
    numbers = len(item[1].split('_')[:-1])
    train_x[index] /= numbers

train_x = np.concatenate((train_x, train_x5), 1)

set_name = 'testset4'

test_list = np.load(set_name+'/Y.npy')
test_multiidx = np.load(set_name+'/multiidx.npy')
test_singleidx = [i for i in range(len(test_list)) if i not in test_multiidx]

test_y = np.load(set_name+'/Y.npy')[:, -1].astype(float)
# test_y = test_y[test_singleidx]
test_x1 = np.load(set_name+'/34_feature.npy')
test_x2 = np.load(set_name+'/environment.npy')
test_x3 = np.load(set_name+'/normal_feature.npy')
test_x5 = np.load('250epoch_model_graph_features/X_gbdt_'+set_name+'.npy')[:,-128:][:, importance]

test_x = np.concatenate((test_x1[:, one_idx], test_x2[:, environ_idx],  test_x3[:, normal_idx]), 1)
for index in test_multiidx:
    item = test_list[index]
    numbers = len(item[1].split('_')[:-1])
    test_x[index] /= numbers

test_x = np.concatenate((test_x, test_x5), 1)

print(test_x.shape)

trainL = train_x.shape[0] // 10 * 9
train_x_t = train_x[:trainL, :]
val_x = train_x[trainL:, :]
val_y = train_y[trainL:]
train_y = train_y[:trainL]
train_x = train_x_t

model = xgb.XGBRegressor(max_depth = 15, 
                        learning_rate=0.001, 
                        n_estimators= 3500, #6000 for s734 , 3000 for s888
                        objective='reg:linear', 
                        nthread=-1,  
                        gamma=0,
                        min_child_weight = 2, 
                        max_delta_step=0, 
                        subsample=0.6, 
                        colsample_bytree=0.6, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=1)
model.fit(train_x, train_y, eval_metric='rmse', verbose = True, eval_set = [(val_x, val_y)],early_stopping_rounds=14000)

preds = model.predict(test_x)
RMSD = np.sqrt(mean_squared_error(test_y,preds))
pearsonr = scipy.stats.pearsonr(test_y,preds)
kan = kendall(preds,test_y )

##print('pearson:', pearsonr[0])
print('loss:', RMSD)
print('kandell:', kan[0])
