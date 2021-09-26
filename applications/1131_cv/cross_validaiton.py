from os import environ
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
import scipy
import sys
import xgboost as xgb

X = np.load('environment.npy')

X1 = np.load('34_features.npy')

X2 = np.load('normal_residue.npy')
X2 = np.load('residue_feature.npy')

X3 = np.load('dssp.npy')

gpu_index = np.load('graph_arg_importance_onehot.npy')[-108:]

# X5 = np.load('4169_graphfeature.npy')[:,-128:][:, gpu_index]
X5 = np.load('1131_graph_features_onehot.npy')[:, -256:][:, gpu_index]
# X4 = np.load('X_gbdt.npy')[:, -1024:]
Y = np.load('skempi_1131.npy')[:, -1].astype(float)

print(Y.shape)
n_num = Y.shape[0]

normal_idx = [-18, -29, -20, -31, -16, -27, -14, -25, -13, -24, -12, -23, -17, -28]
# one_idx_34 = [i for i in range(0, 30) if i not in [14,19,25,1,27,7,3,30,31,33,18,20,16]]
one_idx_34 = [i for i in range(0, 34) if i not in [0,1,4,31,32,33]]
# environ_idx = [i for i in range(0, 25) if i not in [18,]]

##X = np.concatenate((X3[:, :], X1[:, one_idx_34], X, X2[:, normal_idx]), 1)

X = np.concatenate(( X1[:, one_idx_34], X, X2[:, :], X3), 1)

X = np.concatenate((X, X5), 1)
# X = X1

print(X.shape)

n_split = 10

RMSE_whole = []
pearsonr_whole = []

iter_num = 1
importance = []

kf = KFold(n_splits=10, shuffle = True)
for j in range(0,iter_num):

    result = np.zeros(n_num)
    kf = KFold(n_splits=10,shuffle=True)
    
    pear_sum = 0
    fold = 0
    loss = 0
    for train_index, test_index in kf.split(X):
        # print(test_index)
        fold += 1
        print(fold)
        
        X_train,X_test = X[train_index], X[test_index]
        # X_train = np.concatenate((X_train, train_gbdt[:, -128:][:, gpu_index]), 1)
        # X_test = np.concatenate((X_test, test_gbdt[:, -128:][:, gpu_index]), 1)

        # X_train = train_gbdt[:, -128:]
        # X_test = test_gbdt[:, -128:]
        # print(X_test)``
        Y_train,Y_test = Y[train_index], Y[test_index]
        print('start training')
        model = GradientBoostingRegressor(n_estimators=10000, max_features = "sqrt", learning_rate=0.001, max_depth=16, min_samples_split= 4, subsample=0.4).fit(X_train, Y_train)
        
        # model = RandomForestRegressor(n_estimators=15000, max_features = "sqrt", max_depth=13, min_samples_split= 5).fit(X_train, Y_train)
        # model = xgb.XGBRegressor(max_depth=7, 
        #                 learning_rate=0.01, 
        #                 n_estimators=5000, 
        #                 objective='reg:linear', 
        #                 nthread=-1,  
        #                 gamma=0,
        #                 min_child_weight=3, 
        #                 max_delta_step=0, 
        #                 subsample=0.85, 
        #                 colsample_bytree=0.7, 
        #                 colsample_bylevel=1, 
        #                 reg_alpha=0, 
        #                 reg_lambda=1, 
        #                 scale_pos_weight=1, 
        #                 seed=1440, 
        #                 missing=1)
        # model.fit(X_train, Y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)],early_stopping_rounds=15000)

        result[test_index] = model.predict(X_test)
        # preds = model.predict(X_test)
        # # print(preds)
        RMSD = np.sqrt(mean_squared_error(Y_test,model.predict(X_test)))
        pearsonr = scipy.stats.pearsonr(Y_test,model.predict(X_test))

        pear_sum = pear_sum + pearsonr[0]
        loss += RMSD ** 2
        print(RMSD)
        print(pearsonr[0])
        imp = model.feature_importances_
        importance.append(imp)
    print('pearson sum:',pear_sum)
    # print('loss sum:',loss ** 0.5)
    RMSD = np.sqrt(mean_squared_error(Y,result))
    pearsonr = scipy.stats.pearsonr(Y,result)

    
    pearsonr_whole.append(pearsonr[0])
    RMSE_whole.append(RMSD)

# np.save("pearsonr_whole",pearsonr_whole)
importance = np.array(importance)
importance = np.sum(importance, 0)
# np.save('graph_arg_importance_onehot.npy',importance)
print('pearson:', pearsonr_whole)
print('loss:', RMSD)
# np.save('result/preds.npy', result)


##file_out = open("crossvalidation_skempi.txt","wb")
##file_out.write("Pearsonr_average is:")
##file_out.write(str(np.mean(pearsonr_whole)))
##file_out.write("\n")
##file_out.write("RMSE_average is:")
##file_out.write(str(np.mean(RMSE_whole)))
##file_out.close()
