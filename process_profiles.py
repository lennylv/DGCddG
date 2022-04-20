import numpy as np
import os

from scipy import special

def process_hhm(path):
    with open(path,'r') as fin:
        fin_data = fin.readlines()
        hhm_begin_line = 0
        hhm_end_line = 0
        for i in range(len(fin_data)):
            if '#' in fin_data[i]:
                hhm_begin_line = i+5
            elif '//' in fin_data[i]:
                hhm_end_line = i
        feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
        axis_x = 0
        for i in range(hhm_begin_line,hhm_end_line,3):
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i+1].split()
            axis_y = 0
            for j in line1:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            for j in line2:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            axis_x+=1
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        # hmm_dict[file.split('.')[0]] = feature
        # print(feature.shape)
        return feature

def process_pssm(path):
    import math
    with open(path,'r') as fin:
        fin_data = fin.readlines()
        pssm_begin_line = 3
        pssm_end_line = 0
        for i in range(1,len(fin_data)):
            if fin_data[i] == '\n':
                pssm_end_line = i
                break
        feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
        axis_x = 0
        for i in range(pssm_begin_line,pssm_end_line):
            raw_pssm = fin_data[i].split()[2:22]
            axis_y = 0
            for j in raw_pssm:
                feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                axis_y+=1
            axis_x+=1
        # nor_pssm_dict[file.split('.')[0]] = feature
        # print(feature.shape)
    # with open(feature_dir+'/{}_PSSM.pkl'.format(ligand),'wb') as f:
    #     pickle.dump(nor_pssm_dict,f)
    return feature
# process_pssm('./1A22_A.pssm')

