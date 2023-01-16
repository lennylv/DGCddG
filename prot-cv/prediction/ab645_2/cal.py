import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

pred = np.load('./ab645_y_pred.npy')
y = np.load('./itemlist.npy')[:,-1].astype(float)
pcc = []
rmse= []
for i in range(5):
        index = np.load('fold'+str(i)+'_index.npy')
##         = np.load('fold'+str(i)+'_pred.npy')
        p = pred[index]
        yy = y[index]
        pcc.append(pearsonr(p,yy)[0])
        rmse.append(np.sqrt(mean_squared_error(p,yy)))
print(pcc)

print('mean pcc:',np.mean(pcc))
print('o pcc:',pearsonr(y, pred)[0])
print('avg pcc:', (np.mean(pcc)+pearsonr(y, pred)[0])/2)

print()
print(np.sqrt(mean_squared_error(y,pred)))
print(np.mean(rmse))
print('avg rmse:', (np.sqrt(mean_squared_error(y,pred)) + np.mean(rmse))/2)
