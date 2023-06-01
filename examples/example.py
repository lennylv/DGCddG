from itertools import chain
import numpy as np

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

def residue_feature(AA):
    def AAcharge(AA):
        if AA in ['D','E']:
            return -1.
        elif AA in ['R','H','K']:
            return 1.
        else:
            return 0.
    residueFeature = []
    Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    PolarAll = ['S','T','N','Q','R','H','K','D','E']
    PolarUncharged = ['S','T','N','Q']
    PolarPosCharged = ['R','H','K']
    PolarNegCharged = ['D','E']
    SpecialCase = ['C','U','G','P']
    AAvolume = {'A':88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, 'E':138.4, 'Q':143.8, 'G':60.1, 'H':153.2, 'I':166.7, 'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, 'S':89., 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140. }
    AAhydropathy = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D': -3.5, 'C': 2.5, 'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5, 'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
    AAarea = {'A':115.,'R':225.,'D':150.,'N':160.,'C':135.,'E':190.,'Q':180.,'G':75.,'H':195.,'I':175.,'L':170.,'K':200.,'M':185.,'F':210.,'P':145.,'S':115.,'T':140.,'W':255.,'Y':230.,'V':155.}
    AAweight = {'A':89.094,'R':174.203,'N':132.119,'D':133.104,'C':121.154,'E':147.131,'Q':146.146,'G':75.067,'H':155.156,'I':131.175,'L':131.175,'K':146.189,'M':149.208,'F':165.192,'P':115.132,'S':105.093,'T':119.12,'W':204.228,'Y':181.191,'V':117.148}
    Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]
    # AA = residue
    for Group in Groups:
        if AA in Group:
            residueFeature.append(1.0)
        else:
            residueFeature.append(0.0)
    residueFeature.append(AAvolume[AA])
    residueFeature.append(AAhydropathy[AA])
    residueFeature.append(AAarea[AA])
    residueFeature.append(AAweight[AA])
    residueFeature.append(AAcharge(AA))

    flex = {'A':'1','R':'81','N':'36','D':'18','C':'3','E':'54','Q':'108','G':'1','H':'36','I':'9','L':'9','K':'81','M':'27','F':'18','P':'2','S':'3','T':'3','W':'36','Y':'18','V':'3'}
    flexbility = flex[AA]
    residueFeature.append(flexbility)

    if AA in ('A','G','I','L','P','V'):chemical = 0
    elif AA in ('R','H','K'):chemical = 1
    elif AA in ('D','E'):chemical = 2
    elif AA in ('N','Q'):chemical = 3
    elif AA in ('C','M'):chemical = 4
    elif AA in ('S','T'):chemical = 5
    elif AA in ('F','W','Y'):chemical = 6
    residueFeature.append(chemical)

    if AA in ('G','A','S'):size = 0
    elif AA in ('C','D','P','N','T'):size = 1
    elif AA in ('E','V','Q','H'):size = 2
    elif AA in ('M','I','L','K','R'):size = 3
    elif AA in ('F','Y','W'):size = 4
    residueFeature.append(size)

    if AA in ('R','W','K'):hbonds = 0
    if AA in ('A','C','G','I','L','M','F','P','V'):hbonds = 1 
    if AA in ('N','Q','S','T','H','Y'):hbonds = 3  
    if AA in ('D','E'):hbonds = 4
    residueFeature.append(hbonds)
    # one-hot version
    residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
                       'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    
    residue_hot = np.eye(20)
    z = zip(residues, residue_hot)
    z = list(z)
    residue_type={}
    for residue in z:
        residue_type[residue[0]] = residue[1]

    def res_to_onehot( res ):
        return residue_type[res]
    
    AA_to_three = AA
    for a in res_dict.keys():
        if res_dict[a] == AA:
            AA_to_three = a
            break
    
    # 35 dimension
    res = np.concatenate((residueFeature, res_to_onehot(AA_to_three)), 0)
    
    return residueFeature

def help_analyze_pdb(pdb, chain):
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    f = open(pdb)
    lines = f.readlines()
    seq = []
    pos = []
    ResId = []

    wuhu = []
    for line in lines:
        if line[:4] !='ATOM':continue
        current_chain = line[21:22].strip()
        if chain != current_chain: continue

        resid = line[22:27].strip()
        atom_type = line[12:16].strip()
        if atom_type != 'CA':continue
        if len(ResId) != 0 and ResId[-1] == resid:continue
        restype = line[17:20]
        try:
            restype = res_dict[restype]
            # seq.append(restype)
            # pos.append(([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            # ResId.append(resid)
            wuhu.append([restype, resid, line[30:38], line[38:46], line[46:54]])
        except:
            continue
    # import numpy as np
    # pos = np.asarray(pos)
    return wuhu

def generate_adj( select_pos):
    from scipy.spatial import cKDTree
    import scipy.sparse as sparse

    matrix = [[0 for i in range(select_pos.shape[0])] for j in range(select_pos.shape[0])]

    t = cKDTree(select_pos)
    nb_3 = cKDTree.query_pairs(t, 10)
    # print(len(nb_3))
    for i,j in nb_3:
        matrix[i][j] = 1
        matrix[j][i] = 1
    adj = np.array(matrix)
    adj = sparse.coo_matrix(adj)
    # np.save(direc+name+'.npy', adj)
    return adj

def dssp(pdb, chain, resid):
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.DSSP import DSSP
    # from Bio.PDB.Polypeptide import *
    def ss2index( st ):
        if st == 'H':
            return 1
        elif st == 'E':
            return 2
        elif st == 'G':
            return 3
        elif st == 'S':
            return 4
        elif st == 'B':
            return 5
        elif st == 'T':
            return 6
        elif st == 'I':
            return 7
        elif st == '-':
            return 0
    
    chainid = chain
    parser = PDBParser()
    pdb_path = './pdb_sets/' + pdb + '.pdb'
    # pdb_path = './A_mut.pqr'
    structure = parser.get_structure('prot', pdb_path)
    model = structure[0]
    dssp = DSSP(model, pdb_path, './dddd')

    temp = dssp[(chainid, (' ', int(resid), ' '))]
    temp = list(temp)
    print(temp)
    # temp[1] = ss2index(temp[1])
    temp[2] = ss2index(temp[2])
    temp = temp[:1] + temp[2:]
    # print()
    # ssindex = ss2index(dssp[(chainid, (' ', 20, ' '))][2])
    ##self.FeatureSeq.append(ssindex)
    # print(ssindex)
    # print(temp)
    # FeatureSeq.append(temp)
    #     # break
    # FeatureSeq = np.array(FeatureSeq)
    return np.array(temp[1:5])

def get_chains(pdb_path):
    f = open(pdb_path)
    lines = f.readlines()
    f.close()

    lines = [line for line in lines if line[:4]=='ATOM']
    chains = [line[21] for line in lines]
    chains = list(set(chains))
    return chains

def pdb_to_res(pdb, pdb_path, mut_chain, residue_id, wild, mutant):

    residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
                       'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}

    # from Bio.PDB import PDBParser as parser
    # p = parser()
    # structure = p.get_structure(pdb, pdb_path)
    chains = get_chains(pdb_path)

    # chains = [c.get_id() for c in chains]
    chain_residue = {}
    chain_residue_feature = {}

    mut_chain_residue = []
    mut_chain_residue_feature = []
    for chain in chains:
        c = chain
        chain_residue[c] = []
        chain_residue[c].extend(  help_analyze_pdb(pdb_path, c) )

        reslist = [a[0] for a in chain_residue[c]]

        regular_feature = np.array([residue_feature(aa) for aa in reslist])
        hhm = process_hhm( 'pdb_set/' + pdb + '_' + c + '.hhm' )
        pssm = process_pssm( 'pdb_set/' + pdb + '_' + c + '.pssm' )

        chain_feature = np.concatenate((regular_feature, hhm, pssm), 1)
        chain_residue_feature[c] = chain_feature

        if c == mut_chain:
            resid_list = [a[1] for a in chain_residue[mut_chain]]
            index = resid_list.index(residue_id.upper())
            reslist[index] = mutant

            mutant_regular_feature = np.array([residue_feature(aa) for aa in reslist])

            mutant_hhm_path = 'pdb_set/mutant.hhm'
            mutant_pssm_path = 'pdb_set/mutant.pssm'

            mutant_hhm = process_hhm( mutant_hhm_path )
            mutant_pssm = process_pssm ( mutant_pssm_path )
            mutant_chain_feature = np.concatenate((mutant_regular_feature, mutant_hhm, mutant_pssm), 1)

    resid_list = [a[1] for a in chain_residue[mut_chain]]

    position_list = [a[2:] for a in chain_residue[mut_chain]]

    mut_res_position = position_list[resid_list.index(residue_id.upper())]
    mut_res_position = [float(a) for a in mut_res_position]
    mut_res_position = np.array(mut_res_position)

    all_res_position = []
    all_res_feature = []

    mutant_res_feature_all = []

    for chain in chains:
        c = chain
        position_tmp = np.array([a[2:] for a in chain_residue[c]])
        # print('what')
        if len(all_res_position) != 0:
            all_res_position = np.concatenate((all_res_position, position_tmp), 0)
        else:
            all_res_position = position_tmp

        if len(all_res_feature) != 0:
            all_res_feature = np.concatenate((all_res_feature, chain_residue_feature[c]), 0)
        else:
            all_res_feature = chain_residue_feature[c]


        if c != mut_chain:
            if len(mutant_res_feature_all) != 0:
                mutant_res_feature_all = np.concatenate((mutant_res_feature_all, chain_residue_feature[c]), 0)
            else:
                mutant_res_feature_all = chain_residue_feature[c]
        else:
            if len(mutant_res_feature_all) != 0:
                mutant_res_feature_all = np.concatenate((mutant_res_feature_all, mutant_chain_feature), 0)
            else:
                mutant_res_feature_all = mutant_chain_feature

    N = all_res_position.shape[0]
    distance = [( j, np.linalg.norm( all_res_position[j].astype(float) - mut_res_position )) for j in range(N)]

    distance.sort(key = lambda x: x[1])
    distance_temp = distance[:500]

    m_in = distance_temp[0][0]
    s = [i[0] for i in distance_temp]
    s.sort()
    m_in = s.index(m_in)
    select_pos = all_res_position[s]

    # print(chain_residue['A'])
    adj = generate_adj( select_pos )
    wild_features = all_res_feature[s]
    mutant_features = mutant_res_feature_all[s]
    nodes_number = np.array([wild_features.shape[0]])
    mutation_site = np.array([[m_in]], dtype=np.int32)
    y = np.array([[pdb, mut_chain, wild, residue_id, mutant, '0']])

    try:
        wild_ss = dssp(pdb, chain, residue_id)
    except:
        wild_ss = np.zeros((1,4))
    
    adj_path = 'exampleset/adjs/' + pdb + '.npy'
    wild_features_path = 'exampleset/wild_features.npy'
    mutant_features_path = 'exampleset/mutant_features.npy'
    nodes_number_path = 'exampleset/nodes_number.npy'
    mutation_site_path ='exampleset/mutation_site.npy'
    y_path = 'exampleset/itemlist.npy'
    aux_path = 'exampleset/aux_features.npy'

    np.save(adj_path, adj)
    np.save(wild_features_path, wild_features)
    np.save(mutant_features_path, mutant_features)
    np.save(nodes_number_path, nodes_number)
    np.save(mutation_site_path, mutation_site)
    np.save(y_path, y)
    np.save(aux_path, wild_ss)


import argparse
import sys

def parse_args(args):
    parser = argparse.ArgumentParser(description='ddG')

    parser.add_argument('--pdb', default='T56', type=str)
    parser.add_argument('--mutation_chain', default='G', type=str)
    parser.add_argument('--residue_id', default='20', type=str)
    parser.add_argument('--wild', default='E', type=str)
    parser.add_argument('--mutant', default='A', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args(sys.argv[1:])
    pdb = args.pdb
    pdb_path = 'pdb_set/' + pdb + '.pdb'
    mutation_chain = args.mutation_chain
    residue_id = args.residue_id
    wild = args.wild
    mutant = args.mutant
    pdb_to_res(pdb, pdb_path, mutation_chain, residue_id=residue_id, wild=wild, mutant=mutant)
    os.system('python model_load.py --dataset exampleset')

main()
