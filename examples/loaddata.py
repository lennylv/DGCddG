from typing import Set
import scipy.sparse as sp
import os
import numpy as np
import torch
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def processWildFeaturesAndAdj(set_name ,nodes_per_graph):
    print('load wild features...')
    if set_name != 'ctc':
        ske = np.load(set_name+'/itemlist.npy')
        ske = ske.astype(str)
    else:
        f = open(set_name + '/list.txt')
        ske = f.read()
        f.close()
        ske = eval(ske)
    
    # ske = cross_validation_split(ske, 10, fold)
    wild_feature_path = set_name+'/wild_features.npy'
    wild_nodes_number_path = set_name+'/nodes_number.npy'
    wild_features = np.load(wild_feature_path)
    wild_nodes_number = np.load(wild_nodes_number_path)
    # nodes_per_graph = 453

    adj_sub = np.empty((len(wild_nodes_number), nodes_per_graph, nodes_per_graph))

    print('processing wild...')
    # before = 0
    adj_list = []
    for i in range(len(wild_nodes_number)):
        #adj
        subgraph=np.identity(nodes_per_graph)
        subgraph = sp.csr_matrix(subgraph).tolil()
        if set_name[0]=='a':
            file = '7kmb' + str(i)
        elif set_name=='delta_29':
            file = '7kmb'
        elif set_name=='ctc':
            file = 'ctc'
        elif set_name == 'deep_sars2':
            file = str(i)
        else:
            file = ske[i][0]
        file = file + '.npy'
        path = set_name+'/adjs/' + file
        adj = np.load(path, allow_pickle=True).tolist()
        subgraph[:adj.shape[0], :adj.shape[1]] = adj
        adj_sub[i, :, :] = subgraph.todense()

    for i in range(adj_sub.shape[0]):
        adj = adj_sub[i]
        adj = sp.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        temp = sparse_mx_to_torch_sparse_tensor(tmp)
        adj_list.append(temp)

    adj_sub = adj_list
    return wild_features, adj_sub, wild_nodes_number


def processMutantFeaturesAndAdj(set_name, nodes_per_graph):
    print('load mutant features...')
    if set_name != 'ctc':
        ske = np.load(set_name+'/itemlist.npy')
        ske = ske.astype(str)
    else:
        f = open(set_name + '/list.txt')
        ske = f.read()
        f.close()
        ske = eval(ske)
    
    # ske = cross_validation_split(ske, 10, fold)
    mut_feature_path = set_name+'/mut_features.npy'
    mut_nodes_number_path = set_name+'/nodes_number.npy'
    mut_features = np.load(mut_feature_path)
    mut_nodes_number = np.load(mut_nodes_number_path)
    
    #here use 453
    # nodes_per_graph = 453
    # features_sub = np.empty((len(wild_nodes_number), nodes_per_graph, 30))
    adj_sub = np.empty((len(mut_nodes_number), nodes_per_graph, nodes_per_graph))

    print('processing mutant...')
    adj_list = []
    for i in range(0, len(mut_nodes_number)):

        #adj
        subgraph=np.identity(nodes_per_graph)
        subgraph = sp.csr_matrix(subgraph).tolil()
        if set_name[0]=='a':
            file = '7kmb' + str(i)
        elif set_name=='delta_29':
            file = '7kmb'
        elif set_name=='ctc':
            file = 'ctc'
        elif set_name == 'deep_sars2':
            file = str(i)
        else:
            file = ske[i][0]
        file = file + '.npy'
        path = set_name+'/adjs/' + file
        adj = np.load(path, allow_pickle=True).tolist()
        subgraph[:adj.shape[0], :adj.shape[1]] = adj
        adj_sub[i, :, :] = subgraph.todense()

    for i in range(len(adj_sub)):
        adj = adj_sub[i]
        adj = sp.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        temp = sparse_mx_to_torch_sparse_tensor(tmp)
        adj_list.append(temp)

    adj_sub = adj_list
    return mut_features, adj_sub, mut_nodes_number

def helpGetWildAndMutIndex_40(wild, mut, mut_id):
    w = []
    mutant = []
    for i in range(len(wild)):
        if int(wild[i][-3]) == 1 :
            w.append(i)
    for i  in range(len(mut)):
        if int(mut[i][-3]) == 1 :
            mutant.append(i)
            
    return w, mutant

def helpGetWildAndMutIndex_30(wild, mut, mut_id):
    w = []
    mutant = []
    for i in range(len(wild)):
        if int(wild[i][24]) == 1 and int(wild[i][25])==mut_id:
            w.append(i)
    for i  in range(len(mut)):
        if int(mut[i][24]) == 1 and int(mut[i][25])==mut_id:
            mutant.append(i)
            
    return w, mutant

def getWildAndMutIndex(set_name):
    w = []
    m = []
    before_mut = 0
    before_wild = 0
    ske = np.load(set_name+'/itemlist.npy').astype(str)
    mut_features = np.load(set_name+'/mut_features.npy')
    mut_nodes = np.load(set_name+'/mut_nodes_number.npy')
    wild_features = np.load(set_name+'/wild_features.npy')
    wild_nodes = np.load(set_name+'/wild_nodes_number.npy')

    for index in range(len(ske)):
        this_wild_nodes = wild_nodes[index]
        current_wild = wild_features[before_wild:before_wild+this_wild_nodes]
        this_mut_nodes = mut_nodes[index]
        current_mut = mut_features[before_mut:before_mut+this_mut_nodes]

        wild, mutant = helpGetWildAndMutIndex_40(current_wild, current_mut, ske[index][3])
        w.append(wild)
        m.append(mutant)

        before_mut += this_mut_nodes
        before_wild += this_wild_nodes
    
    return w, m

def get_residue_feature(set_name='s1131', stand = False):
    # wild_residue = set_name + '/wild_residue.npy'
    # mutant_residue = set_name + '/mutant_residue.npy'
    
    # wild_residue = np.load(wild_residue)
    # mutant_residue = np.load(mutant_residue)
    # wuhu = mutant_residue - wild_residue
    # wild_and_mutant = np.concatenate((wild_residue, mutant_residue, wuhu), 1)
    if not stand:
        path = set_name + '/aux_features.npy'
    else:
        path = set_name + '/aux_features_stand.npy'
    residue_feature = np.load(path)
    wild_and_mutant = torch.from_numpy(residue_feature)

    return wild_and_mutant

def get_ske_and_labels(set_name = 's1131'):
    ske = set_name + '/itemlist.npy'
    ske = np.load(ske).astype(str)
    labels = ske[:, -1].astype(float)
    return ske, labels

def standarize(w_features, m_features, wild, mutant, wild_test, mutant_test, nodes_per_graph):
    print('standarizing ...')
    #concat all wild,mutant
    # stand = []
    # for w in wild:
    #     if len(stand) == 0:
    #         stand = w
    #     else:
    #         stand = np.concatenate((stand, w), 0)
    # for m in mutant:
    #     stand = np.concatenate((stand, m), 0)
    
    #standarize
    feature = np.concatenate((w_features, m_features), 0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(feature)

    wild_numpy = np.empty((len(wild), nodes_per_graph, wild[0].shape[1]))
    mutant_numpy = np.empty((len(wild), nodes_per_graph, wild[0].shape[1]))
    wild_test_numpy = np.empty((len(wild_test), nodes_per_graph, wild[0].shape[1]))
    mutant_test_numpy = np.empty((len(wild_test), nodes_per_graph, wild[0].shape[1]))
    print('reshape features...')
    for i in range(0, len(wild)):
        wild[i] = scaler.transform(wild[i])
        n = len(wild[i])
        wild_numpy[i, :n, :] = wild[i]
        
        mutant[i] = scaler.transform(mutant[i])
        n = len(mutant[i])
        mutant_numpy[i, :n, :] = mutant[i] 
    for i in range(0, len(wild_test)):
        wild_test[i] = scaler.transform(wild_test[i])
        n = len(wild_test[i])
        wild_test_numpy[i, :n, :] = wild_test[i]

        mutant_test[i] = scaler.transform(mutant_test[i])
        n = len(mutant_test[i])
        mutant_test_numpy[i, :n, :] = mutant_test[i]

    wild = torch.FloatTensor(wild_numpy)
    mutant = torch.FloatTensor(mutant_numpy)
    wild_test = torch.FloatTensor(wild_test_numpy)
    mutant_test = torch.FloatTensor(mutant_test_numpy)

    return wild, mutant, wild_test, mutant_test

def standarize_train(wild_features, mut_features):
    feature = np.concatenate((wild_features, mut_features), 0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(feature)
    wlid_features =scaler.transform(wild_features)
    mut_features = scaler.transform(mut_features)

    return wild_features, mut_features

def standarize_residue(residue, residue_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(residue)
    residue = scaler.transform(residue)
    residue_test = scaler.transform(residue_test)
    
    residue = torch.from_numpy(residue)
    residue_test = torch.from_numpy(residue_test)
    return residue, residue_test

def get_mutation_indexes(array, indexes):
    result = []
    for index in indexes:
        result.append(array[index])
    return result

def get_adjs(adjs, indexes):
    result = []
    for index in indexes:
        result.append(adjs[index])
    return result

def get_features(features, nodes_number, indexes):
    result = []
    for index in indexes:
        start = sum(nodes_number[:index])
        end = start + nodes_number[index]
        fea = features[start: end]
        result.append(fea)
    return result

def get_labels(SetName):
    # labels = np.load(SetName + '/ab645_new.npy')[:, -1].astype(float)
    # labels = np.load(SetName + '/y_gcn.npy')
    if SetName == 'ctc':
        f = open('./ctc/list.txt')
        l = f.read()
        l = eval(l)
        f.close()
        l = np.array(l)[:, -1].astype(float)
        l = torch.FloatTensor(l)
        l = l.view(-1, 1)
        return l

    labels = np.load(SetName+'/itemlist.npy')[:, -1].astype(float)
    labels = torch.FloatTensor(labels)
    labels = labels.view(-1, 1)
    return labels
def getMutIndex(Setname):
    if Setname == 'ctc':
        path = Setname + '/mutation_site.txt'
        f = open(path)
        mutation_site = f.read()
        mutation_site = eval(mutation_site)
        f.close()
        return mutation_site
    path = Setname + '/mutation_site.npy'
    mutaion_site = np.load(path,allow_pickle=True).tolist()
    return mutaion_site