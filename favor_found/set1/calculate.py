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

items = np.load('./set1_y.npy')
# items2 = np.load('../set2/set2_y.npy')
# items = np.concatenate([items, items2], 0)
y = np.load('./set1_y.npy')[:,-1].astype(float)
# y1 = np.load('../set2/set2_y.npy')[:,-1].astype(float)
# y = np.concatenate([y, y1])
favor = []
strong_favor = []
neural = []
unfavor = []
strong_unfavor = []
mul = []
single = []
for i in range(len(y)):
    if len(items[i][1].split('_'))>2:
        mul.append(i)
    else:
        single.append(i)
    if y[i]<=0: favor.append(i)
    if y[i]<=-1: strong_favor.append(i)
    if y[i]>=0: unfavor.append(i)
    if y[i]>=1: strong_unfavor.append(i)
    if y[i]<1 and y[i]>-1: neural.append(i)
favor = np.array(favor).astype(int)


bindx = readfromtxt('bindx.txt')
evoef = readfromtxt('evoef.txt')
flex = readfromtxt('flexddg.txt')
foldx = readfromtxt('foldx.txt')
ssipe = np.load('./ssipe_testset1.npy')
dgc = np.load('../prediction_2/testset1.npy')
# dgc2 = np.load('../prediction_ss_asa/testset2_multiple_o.npy')
# dgc = np.concatenate([dgc, dgc2])

print(pearsonr(y, dgc)[0], np.sqrt(mean_squared_error(y, dgc)))
print('single:', pearsonr(y[single], dgc[single])[0], 'rmse:', np.sqrt(mean_squared_error(y[single], dgc[single])))
print('mul:', pearsonr(y[mul], dgc[mul])[0], 'rmse:', np.sqrt(mean_squared_error(y[mul], dgc[mul])))

if len(favor)>1:
    ssipe_favor = pearsonr(y[favor], ssipe[favor])[0]
    dgc_favor = pearsonr(y[favor], dgc[favor])[0]
    bindx_favor = pearsonr(y[favor], bindx[favor])[0]
    evoef_favor = pearsonr(y[favor], evoef[favor])[0]
    flex_favor = pearsonr(y[favor], flex[favor])[0]
    foldx_favor = pearsonr(y[favor], foldx[favor])[0]
    print('favor:', len(favor))
    print('ssipe_favor', ssipe_favor)
    print('dgc_favor', dgc_favor)
    print('bindx_favor', bindx_favor)
    print('evoef_favor', evoef_favor)
    print('flex_favor', flex_favor)
    print('foldx_favor', foldx_favor)

if len(strong_favor)>1:
    ssipe_pre = pearsonr(y[strong_favor], ssipe[strong_favor])[0]
    dgc_pre = pearsonr(y[strong_favor], dgc[strong_favor])[0]
    bindx_pre = pearsonr(y[strong_favor], bindx[strong_favor])[0]
    evoef_pre = pearsonr(y[strong_favor], evoef[strong_favor])[0]
    flex_pre = pearsonr(y[strong_favor], flex[strong_favor])[0]
    foldx_pre = pearsonr(y[strong_favor], foldx[strong_favor])[0]
    print('strong_favor:', len(strong_favor))
    print('ssipe_favor', ssipe_pre)
    print('dgc_favor', dgc_pre)
    print('bindx_favor', bindx_pre)
    print('evoef_favor', evoef_pre)
    print('flex_favor', flex_pre)
    print('foldx_favor', foldx_pre)

ssipe_pre = pearsonr(y[unfavor], ssipe[unfavor])[0]
dgc_pre = pearsonr(y[unfavor], dgc[unfavor])[0]
bindx_pre = pearsonr(y[unfavor], bindx[unfavor])[0]
evoef_pre = pearsonr(y[unfavor], evoef[unfavor])[0]
flex_pre = pearsonr(y[unfavor], flex[unfavor])[0]
foldx_pre = pearsonr(y[unfavor], foldx[unfavor])[0]
print('unfavor:', len(unfavor))
print('ssipe_unfavor', ssipe_pre)
print('dgc_unfavor', dgc_pre)
print('bindx_unfavor', bindx_pre)
print('evoef_unfavor', evoef_pre)
print('flex_unfavor', flex_pre)
print('foldx_unfavor', foldx_pre)

ssipe_pre = pearsonr(y[neural], ssipe[neural])[0]
dgc_pre = pearsonr(y[neural], dgc[neural])[0]
bindx_pre = pearsonr(y[neural], bindx[neural])[0]
evoef_pre = pearsonr(y[neural], evoef[neural])[0]
flex_pre = pearsonr(y[neural], flex[neural])[0]
foldx_pre = pearsonr(y[neural], foldx[neural])[0]
print('neural:', len(neural))
print('ssipe_neural', ssipe_pre)
print('dgc_neural', dgc_pre)
print('bindx_neural', bindx_pre)
print('evoef_neural', evoef_pre)
print('flex_neural', flex_pre)
print('foldx_neural', foldx_pre)

ssipe_pre = pearsonr(y[strong_unfavor], ssipe[strong_unfavor])[0]
dgc_pre = pearsonr(y[strong_unfavor], dgc[strong_unfavor])[0]
bindx_pre = pearsonr(y[strong_unfavor], bindx[strong_unfavor])[0]
evoef_pre = pearsonr(y[strong_unfavor], evoef[strong_unfavor])[0]
flex_pre = pearsonr(y[strong_unfavor], flex[strong_unfavor])[0]
foldx_pre = pearsonr(y[strong_unfavor], foldx[strong_unfavor])[0]
print('strong_unfavor:', len(strong_unfavor))
print('ssipe_neural', ssipe_pre)
print('dgc_neural', dgc_pre)
print('bindx_neural', bindx_pre)
print('evoef_neural', evoef_pre)
print('flex_neural', flex_pre)
print('foldx_neural', foldx_pre)

