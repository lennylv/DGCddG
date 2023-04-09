import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

pred = np.empty(4169,)
y = np.load('./itemlist.npy')[:,-1].astype(float)
pcc = []
rmse= []
for i in range(1,9):
        index = np.load('4169_testindex_'+str(i)+'.npy')
        p = np.load('4169_fold'+str(i)+'.npy')
        pred[index] = p
        yy = y[index]
        pcc.append(pearsonr(p,yy)[0])
        rmse.append(np.sqrt(mean_squared_error(p,yy)))
print(pcc)

print('overall pcc:',pearsonr(y, pred)[0])
print('mean pcc:',np.mean(pcc))

print()
print('overall rmse:', np.sqrt(mean_squared_error(y,pred)))
print('mean rmse:', np.mean(rmse))
