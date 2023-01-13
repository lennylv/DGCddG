import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def txt_to_list(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    lines = [line.strip() for line in lines[1:]]
    pdb = []
    chains = []
    mutations = []
    ddg = []
    pred = []

    for line in lines:
        l = line.split()
        pdb.append(l[0])
        chains.append(l[1])
        mutations.append(l[2])
        ddg.append(float(l[3]))
        pred.append(float(l[-1]))

    return pdb, chains, mutations, ddg, pred

def compare_id(path1, path2):
    set1_pdb, set1_chains, set1_mutaion, set1_ddg, _ = txt_to_list(path1)
    set2_pdb, set2_chains, set2_mutaion, set2_ddg, pred = txt_to_list(path2)
    import numpy as np
    dgc = np.load('../prediction_ss_asa/testset1.npy')
    sort_idx = []
    for i in range(len(set2_pdb)):
        pdb = set2_pdb[i]
        chains = set2_chains[i]
        mutation = set2_mutaion[i]
        ddg = set2_ddg[i]

        for j in range(len(set1_pdb)):
            s_pdb = set1_pdb[j]
            s_chains = set1_chains[j]
            s_mutation = set1_mutaion[j]
            s_ddg = set1_ddg[j]
            if pdb == s_pdb and chains == s_chains and mutation == s_mutation:
                sort_idx.append(j)
                break
    
    if len(sort_idx) == len(pred):
        print('yes', len(sort_idx))
        print('dgc rmse:'+str(np.sqrt(mean_squared_error(dgc[sort_idx], np.array(set2_ddg)))), ' pcc:', pearsonr(dgc[sort_idx], np.array(set2_ddg))[0])
        print(path2 + 'rmse:'+str(np.sqrt(mean_squared_error(pred, np.array(set2_ddg)))), ' pcc:', pearsonr(pred, np.array(set2_ddg))[0])
        print()
    else:
        print('no')

apps = ['beatmusic', 'elaspic', 'mCSM', 'MutaBind', 'SAAMBE']
for app in apps:
    compare_id('./sort/ssipe.txt', './sort/' + app + '.txt')