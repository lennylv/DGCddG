import numpy as np
graph_features = np.empty((1131, 768))
ii = []
for i in range(1, 11):
    feature = np.load('pre_train/' + str(i) +'/test_gdbt.npy')
    index = np.load('pre_train/' + str(i) +'/test_index.npy')
    graph_features[index, :] = feature
    if len(ii) == 0:
        ii = index
    else:
        ii = np.concatenate((ii, index),0)
print(ii.shape)
np.save('1131_graphfeature.npy', graph_features)