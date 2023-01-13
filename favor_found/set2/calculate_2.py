import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
def affinity_2_binary(s, cutoff):
    for i in range(len(s)):
        if s[i]<cutoff:s[i] = 1
        else:
            s[i] = 0
    return s.astype(int)

def readfromtxt(path):
    f = open(path)
    lines = f.readlines()
    values = [float(l.strip()) for l in lines[1:]]
    return np.array(values)

items = np.load('./set2_y.npy')
items2 = np.load('../set1/set1_y.npy')
items = np.concatenate([items, items2], 0)
y = np.load('./set2_y.npy')[:,-1].astype(float)
y3 = y
y1 = np.load('../set1/set1_y.npy')[:,-1].astype(float)
y = np.concatenate([y, y1])

favor = []
strong_favor = []
neural = []
unfavor = []
strong_unfavor = []
for i in range(len(y)):
    # if len(items[i][1].split('_'))>2:continue
    if y[i]<=0: favor.append(i)
    if y[i]<=-1: strong_favor.append(i)
    if y[i]>=0: unfavor.append(i)
    if y[i]>=1: strong_unfavor.append(i)
    if y[i]<1 and y[i]>-1: neural.append(i)
favor = np.array(favor).astype(int)


# bindx = readfromtxt('bindx.txt')
# evoef = readfromtxt('evoef.txt')
# flex = readfromtxt('flexddg.txt')
# foldx = readfromtxt('foldx.txt')
ssipe = readfromtxt('./set2.txt')
ssipe2 = np.load('../set1/ssipe_testset1.npy')
ssipe = np.concatenate([ssipe, ssipe2])

dgc = np.load('../prediction_2/testset2.npy')
dgc3 = dgc
dgc2 = np.load('../prediction_2/testset1.npy')
dgc = np.concatenate([dgc, dgc2])

print(pearsonr(y, dgc)[0], np.sqrt(mean_squared_error(y, dgc)))
print(pearsonr(y3, dgc3)[0], np.sqrt(mean_squared_error(y3, dgc3)))

if len(favor)>1:
    ssipe_favor = pearsonr(y[favor], ssipe[favor])[0]
    dgc_favor = pearsonr(y[favor], dgc[favor])[0]
    print('favor:', len(favor))
    print('ssipe_favor', ssipe_favor)
    print('dgc_favor', dgc_favor)
    

if len(strong_favor)>1:
    ssipe_pre = pearsonr(y[strong_favor], ssipe[strong_favor])[0]
    dgc_pre = pearsonr(y[strong_favor], dgc[strong_favor])[0]
    print('strong_favor:', len(strong_favor))
    print('ssipe_favor', ssipe_pre)
    print('dgc_favor', dgc_pre)

ssipe_pre = pearsonr(y[unfavor], ssipe[unfavor])[0]
dgc_pre = pearsonr(y[unfavor], dgc[unfavor])[0]
# bindx_pre = pearsonr(y[unfavor], bindx[unfavor])[0]
# evoef_pre = pearsonr(y[unfavor], evoef[unfavor])[0]
# flex_pre = pearsonr(y[unfavor], flex[unfavor])[0]
# foldx_pre = pearsonr(y[unfavor], foldx[unfavor])[0]
print('unfavor:', len(unfavor))
print('ssipe_unfavor', ssipe_pre)
print('dgc_unfavor', dgc_pre)
# print('bindx_unfavor', bindx_pre)
# print('evoef_unfavor', evoef_pre)
# print('flex_unfavor', flex_pre)
# print('foldx_unfavor', foldx_pre)

ssipe_pre = pearsonr(y[neural], ssipe[neural])[0]
dgc_pre = pearsonr(y[neural], dgc[neural])[0]
# bindx_pre = pearsonr(y[neural], bindx[neural])[0]
# evoef_pre = pearsonr(y[neural], evoef[neural])[0]
# flex_pre = pearsonr(y[neural], flex[neural])[0]
# foldx_pre = pearsonr(y[neural], foldx[neural])[0]
print('neural:', len(neural))
print('ssipe_neural', ssipe_pre)
print('dgc_neural', dgc_pre)
# print('bindx_neural', bindx_pre)
# print('evoef_neural', evoef_pre)
# print('flex_neural', flex_pre)
# print('foldx_neural', foldx_pre)

ssipe_pre = pearsonr(y[strong_unfavor], ssipe[strong_unfavor])[0]
dgc_pre = pearsonr(y[strong_unfavor], dgc[strong_unfavor])[0]
# bindx_pre = pearsonr(y[strong_unfavor], bindx[strong_unfavor])[0]
# evoef_pre = pearsonr(y[strong_unfavor], evoef[strong_unfavor])[0]
# flex_pre = pearsonr(y[strong_unfavor], flex[strong_unfavor])[0]
# foldx_pre = pearsonr(y[strong_unfavor], foldx[strong_unfavor])[0]
print('strong_unfavor:', len(strong_unfavor))
print('ssipe_neural', ssipe_pre)
print('dgc_neural', dgc_pre)
# print('bindx_neural', bindx_pre)
# print('evoef_neural', evoef_pre)
# print('flex_neural', flex_pre)
# print('foldx_neural', foldx_pre)
