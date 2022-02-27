import numpy as np
from scipy.spatial import cKDTree
def processpqr(chain, pdb):
    path = pdb
    pqrFile = open(path)
    lines = pqrFile.read().splitlines()
    result = []
    AtomPos = []
    for line in lines:
        if line[:4] == 'ATOM':
            chain_id = line[20:22].strip()
            if chain_id != chain:continue
            temp = []
            pos = np.array([ float(line[26:38]),float(line[38:46]),float(line[46:54]) ])
            AtomPos.append(np.array([ float(line[26:38]),float(line[38:46]),float(line[46:54]) ]))
            # AtomVerboseType.append( line[11:17] )
            # print(line[54:62])
            resid = line[22:27].strip()
            temp = [chain, resid]
            result.append(temp)

    pqrFile.close()
    return result, AtomPos

def analyze_interface(ske):
    #这个方法得到所有inferface处的残基
    import os
    # exist = os.listdir('interface_dic/')
    exist = []
    nodes = []
    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        pdb = item[0].split('_')[0]
        if item[0] + '.npy' in exist:
            continue
        else:
            exist.append(item[0]+'.npy')
        # if pdb != '1KBH':continue
        chains = item[0].split('_')[1]
        cr = []
        Pos = []
        dic = {}
        for chain in chains:
            # pdb_path = 'trainset/pdbs_set1/' + pdb + '.pdb'
            # pdb_path = 'testset_2/TestSet2/' + pdb +'.pdb'
            pdb_path = './' + pdb + '.pdb'
            #处理单链
            chain_resid, atomPos = processpqr(chain, pdb_path)
            cr = cr + chain_resid
            Pos = Pos + atomPos
            dic[chain] = []
        # print(len(cr))
        Pos = np.asarray(Pos, float)
        t = cKDTree(Pos)
        #这里设置 cut-off值
        nbLong = cKDTree.query_pairs(t, 5)

        for i,j in nbLong:
            chain_i = cr[i][0]
            chain_j = cr[j][0]

            i_res = cr[i][1]
            j_res = cr[j][1]
            if chain_i == chain_j:
                continue
            if i_res not in dic[chain_i]:
                dic[chain_i].append(i_res)
            if j_res not in dic[chain_j]:
                dic[chain_j].append(j_res)
        
        #下面用于检查是否将突变处囊括
        # cs = item[1].split('_')[:-1]
        # ids = item[3].split('_')[:-1]
        # for k in range(0, len(cs)):
        #     if ids[k] not in dic[cs[k]]:
        #         print(item)
        #         dic[cs[k]].append(ids[k])

        nodes.append(len(dic[chains[0]] + dic[chains[1]]))
        # print(len(dic['A']))
        # print(len(dic['B']))
    # np.save('nodes_number.npy', nodes)
        np.save('./interface/'+pdb+'_'+chains+'.npy', dic)



def help_analyze_pdb(pdb, chain):
    #这个方法用于分析pdb文件中的蛋白质残基序列 以及得到残基的坐标
    #残基坐标用CA的坐标代表
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    f = open(pdb)
    lines = f.readlines()
    seq = []
    pos = []
    ResId = []
    for line in lines:
        current_chain = line[21:22].strip()
        if line[:4] !='ATOM':continue
        if chain != current_chain: continue

        resid = line[22:27].strip()
        atom_type = line[12:16].strip()
        if atom_type != 'CA':continue
        if len(ResId) != 0 and ResId[-1] == resid:continue
        restype = line[17:20]
        try:
            restype = res_dict[restype]
            seq.append(restype)
            pos.append(([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            ResId.append(resid)
        except:
            continue
    import numpy as np
    pos = np.asarray(pos)
    return seq, pos, ResId

def analyze_information():
    #这个方法用于得到pdb 文件 的 seq, resid, coordinate
    # ske = np.load('trainset/ssipe_train.npy')
    # ske = np.load('testset_2/ssipe_test_2.npy')
    ske = np.load('skempi_4169.npy')
    exist = []
    chains_dic = eval(open('4169_chains.txt').read())
    for i in range( 0, len(ske)):
        print(i)
        item = ske[i]
        # pdb = item[0].split('_')[0]
        pdb = item[0]
        # chains = item[0].split('_')[1]
        chains = chains_dic[pdb]
        # if pdb in exist:continue
        # else:
        #     exist.append(pdb)
        for chain in chains:
            # pdb_path = 'trainset/pdbs_set1/' + pdb +'.pdb'
            if pdb +'_' +chain in exist:
                continue
            else:
                exist.append(pdb +'_' +chain)
            
            pdb_path = '../../dataset/4169pdbs/' + pdb + '.pdb'

            seq, pos, ResId = help_analyze_pdb(pdb_path, chain)
            seq, pos, ResId = np.array(seq), np.array(pos), np.array(ResId)
            seq = seq.reshape(-1, 1)
            ResId = ResId.reshape(-1, 1)
            try:
                chain_information = np.concatenate((seq, ResId, pos), 1)
                np.save('seq_information/'+pdb+'_'+chain+'.npy', chain_information)
            except:
                continue

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

    # dic = np.load('residue_dic.npy',allow_pickle=True).tolist()
    # residueFeature.extend( dic[AA] )

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
    return res

def seq_information_toseq():
    #这个方法将wild序列依次转换为突变序列
    import os
    import sys

    ske = np.load('skempi_4169.npy')
    chains_file = open('4169_chains.txt')
    chains_dic = chains_file.read()
    chains_dic = eval(chains_dic)

    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        pdb = item[0].split('_')[0]
        chains = chains_dic[pdb]

        mut_chain = item[1]

        # wilds = item[2].split('_')[:-1]
        wilds = item[2]
        ids = item[3]
        mutants = item[4]

        seq_path = []
        seq_information = []
        Pos = []
        mut_seq = []
        for chain in chains:
            try:
                seq_path_temp = 'seq_information/' + pdb + '_' + chain + '.npy'
                seq_information_temp = np.load(seq_path_temp)
            except:
                print('without chain')
                sys.exit()
                continue
            pos_temp = seq_information_temp[:, 2:5]
            mut_seq_temp = list(seq_information_temp[:, 0])

            seq_information.append(seq_information_temp)
            Pos.append(pos_temp)
            mut_seq.append(mut_seq_temp)
        
        #这个循环用于处理多点突变双链
        # for k in range(0, len(mut_chains)):
        #     if mut_chains[k] == chains[0]:
        #         resids = seq_information1[:, 1]
        #         ind = list(resids).index(ids[k])
        #         if seq_information1[ind][0] != wilds[k]:
        #             print('error')
        #             raise Exception
        #         mut_seq1[ind] = mutants[k]
        #     else:
        #         resids = seq_information2[:, 1]
        #         ind = list(resids).index(ids[k])
        #         if seq_information2[ind][0] != wilds[k]:
        #             print('error')
        #             raise Exception
        #         mut_seq2[ind] = mutants[k]

        #下面是处理多链下的单点突变
        index = chains.index(mut_chain)
        seq_information1 = seq_information[index]
        # mut_seq1 = mut_seq[index]

        resids = seq_information1[:, 1]
        ind = list(resids).index(ids.upper())
        if seq_information1[ind][0] != wilds:
            print('error')
            sys.exit()
            raise Exception
        mut_seq[index][ind] = mutants

        pos_save = []
        for p in Pos:
            if len(pos_save)==0: pos_save = p
            else:
                pos_save = np.concatenate((pos_save, p), 0)
        
        # pos = np.concatenate((pos1, pos2), 0)
        # os.makedirs('features/' + str(i))
        np.save('features/'+str(i)+'/pos.npy', pos_save)
        mut_seq_save = []
        for seq in mut_seq:
            if len(mut_seq_save)==0:mut_seq_save = seq
            else:
                mut_seq_save = mut_seq_save + seq

        np.save('features/'+str(i)+'/mut_seq.npy', mut_seq)

        wild_save = []
        for information in seq_information:
            if len(wild_save) == 0 : wild_save = information
            else:
                wild_save = np.concatenate((wild_save, information), 0)

        np.save('features/'+str(i)+'/wild_information.npy',  )
        # mut_seq = mut_seq1 + mut_seq2
        # np.save('features/'+str(i)+'/mut_seq.npy', mut_seq)

# seq_information_toseq()

def seq_to_features():
    #这个方法将所有序列信息转为feature
    # ske = np.load('trainset/ssipe_train.npy')
    # ske = np.load('testset_2/ssipe_test_2.npy')
    ske = np.load('skempi_4169.npy')
    for i in range( 0, len(ske)):
        # break
        item = ske[i]
        if item[0] != '2KSO':continue
        print(i)
        # # pdb = item[0].split('_')[0]
        # pdb = item[0]
        # chains = item[0].split('_')[1]
        # seq_information1 = pdb +'_' +chains[0]
        # seq_information2 = pdb +'_' + chains[1]
        # #wild process
        # seq1 = np.load('testset_2/seq_information/' + seq_information1 +'.npy')[:, 0]
        # seq2 = np.load('testset_2/seq_information/'+seq_information2+'.npy')[:, 0]
        # seq_wild = np.concatenate((seq1, seq2), 0)

        wild_information = np.load('features/'+str(i)+'/wild_information.npy')
        seq_wild = wild_information[:, 0]
        wild_features = []
        for ch in seq_wild:
            feature = residue_feature(ch)
            wild_features.append(feature)
        wild_features = np.array(wild_features)
        np.save('./features/'+str(i)+'/wild.npy', wild_features)
        # break


    for i in range(0, len(ske)):
        
        item = ske[i]
        # pdb = item[0].split('_')[0]
        # chains = item[0].split('_')[1]
        #process mutant
        if item[0] != '2KSO':continue
        print(i)
        seq_path = './features/'+str(i)+'/mut_seq.npy'
        seq = np.load(seq_path, allow_pickle=True)
        s = []
        for q in seq:
            s += q
        seq = s
        mut_features = []
        for ch in seq:
            feature = residue_feature(ch)
            mut_features.append(feature)
        mut_features = np.array(mut_features)
        np.save('./features/'+str(i)+'/mutant.npy', mut_features)
        # break
    # j = -1
    # for i in range(0, len(seq)):
    #     if seq_wild[i] != seq[i]:j = i
    # print(j)
    print('done')

# seq_to_features()

def help_select_atoms():
    #这个方法将 链残基id 放入
    # ske = np.load('trainset/ssipe_train.npy')
    ske = np.load('testset_2/ssipe_test_2.npy')
    for i in range(468, len(ske)):
        print(i)
        item = ske[i]
        pdb = item[0].split('_')[0]
        chains = item[0].split('_')[1]
        seq_information1 = pdb +'_' +chains[0]
        seq_information2 = pdb +'_' + chains[1]
        #wild process resid and chain
        seq1 = np.load('testset_2/seq_information/' + seq_information1 +'.npy')[:, 1]
        seq2 = np.load('testset_2/seq_information/'+seq_information2+'.npy')[:, 1]
        for j in range(len(seq1)):
            seq1[j] = chains[0] + seq1[j]
        for j in range(len(seq2)):
            seq2[j] = chains[1] + seq2[j]
        seq = np.concatenate((seq1, seq2), 0).reshape(-1, 1)

        wild = np.load('testset_2/features/' + str(i) +'/wild.npy')
        wild = np.concatenate((wild, seq), 1)
        np.save('testset_2/features/' + str(i) +'/wild.npy', wild)

        mut = np.load('testset_2/features/' + str(i) +'/mutant.npy')
        mut = np.concatenate((mut, seq), 1)
        np.save('testset_2/features/' + str(i) +'/mutant.npy', mut)
        # break

# help_select_atoms()

def select_atoms():
    wild_features = []
    mut_features = []
    nodes_number = []
    Pos = []
    # ske = np.load('trainset/ssipe_train.npy')
    # ske = np.load('testset_2/ssipe_test_2.npy')
    ske = np.load('./s4169.npy')
    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        pdb = item[0].split('_')[0]
        interface_path = 'interface_dic/' + item[0] +'.npy'
        interface = np.load(interface_path,allow_pickle=True).tolist()
        interface_index = []
        wild = np.load('features/'+str(i)+'/wild.npy')
        mutant = np.load('features/'+str(i)+'/mutant.npy')
        pos = np.load('features/'+str(i)+'/pos.npy')
        for j in range(0, wild.shape[0]):
            res = wild[j]
            chain = res[-1][0]
            resid = res[-1][1:]
            if resid in interface[chain]:interface_index.append(j)
        wild_feature = wild[interface_index, :-1]
        mut_feature = mutant[interface_index, :-1]
        p = pos[interface_index, :]

        if len(wild_features) == 0:
            wild_features = wild_feature
        else:
            wild_features = np.concatenate((wild_features, wild_feature), 0)
        if len(mut_features) == 0:
            mut_features = mut_feature
        else:
            mut_features = np.concatenate((mut_features, mut_feature), 0)
        
        if len(Pos) == 0:Pos = p
        else:
            Pos = np.concatenate((Pos, p),0)

        nodes_number.append(len(wild_feature))
    
    np.save('wild_features.npy', wild_features)
    np.save('nodes_number.npy', nodes_number)
    np.save('mut_features.npy', mut_features)
    np.save('Pos.npy', Pos)

# select_atoms()
def select_atoms_for_single(flag):
    ske = np.load('./s4169.npy')
    target = []
    nodes_number = []
    wild_features = []
    mut_features = []
    Pos = []
    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        pdb = item[0].split('#')[0]
        seq_information = 'seq_information/' + pdb + '_' + item[1] + '.npy'
        seq_information = np.load(seq_information)
        ii = 0
        
        for coor in seq_information:
            # print(coor) 
            if coor[1] == item[3].upper():
                ii += 1
                coordinate = tuple(coor[2:5].astype(float))
                target.append(coordinate)
        # if ii != 1:
        #     print(item)
        #     print(ii)
        #     raise Exception
        select_index = []
        pos_path = './features/' + str(i) + '/pos.npy'
        pos = np.load(pos_path)
        for j in range(len(pos)):
            p = pos[j]
            p = p.astype(float)
            distance = np.linalg.norm( p - coordinate )
            # if distance < 100:
            select_index.append((j, distance))
        
        select_index.sort(key = lambda x: x[1])
        select_index = select_index[:500]
        select_index.sort(key = lambda x: x[0])
        select_index = [i[0] for i in select_index]
        select_index.sort()
        select_pos = pos[select_index]

        if len(Pos) == 0:
            Pos = select_pos
        else:
            Pos = np.concatenate((Pos, select_pos), 0)

        if flag:
            wild_path = './features/' + str(i) + '/wild2.npy'
            wild = np.load(wild_path)[:, :-1].astype(float)
            wild_select = wild[select_index]
            if len(wild_features) == 0:
                wild_features = wild_select
            else:
                pass
                wild_features = np.concatenate((wild_features, wild_select), 0)
        else:
            mut_path = './features/' + str(i) + '/mutant2.npy'
            mut = np.load(mut_path)[:, :-1].astype(float)
            mut_select = mut[select_index]
            if len(mut_features) == 0:
                mut_features = mut_select
            else:
                pass
                mut_features = np.concatenate((mut_features, mut_select), 0)

        nodes_number.append( len(select_index) )
    if flag:
        np.save('./gcn_features/wild_features.npy', wild_features)
    else:
        np.save('./gcn_features/mut_features.npy', mut_features)

    np.save('./gcn_features/Pos.npy', Pos)
    np.save('./gcn_features/nodes_number.npy', nodes_number)

# select_atoms_for_single(True)
# select_atoms_for_single(False)

def adjs():
    from scipy.spatial import cKDTree
    import scipy.sparse as sparse

    # ske = np.load('trainset/ssipe_train.npy')
    # ske = np.load('testset_2/ssipe_test_2.npy')
    ske = np.load('./s4169.npy')
    nodes_number = np.load('./gcn_features/nodes_number.npy')
    Pos = np.load('./gcn_features/Pos.npy')
    direc = './gcn_features/adjs/'
    before = 0
    e = []
    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        name = item[0]
        number = nodes_number[i]
        pos = Pos[before:before+number, :]
        before += number
        if name in e:continue
        else:
            e.append(name)

        matrix = [[0 for i in range(number)] for j in range(number)]

        t = cKDTree(pos)
        nb_3 = cKDTree.query_pairs(t, 8)
        # print(len(nb_3))
        for i,j in nb_3:
            matrix[i][j] = 1
            matrix[j][i] = 1
        adj = np.array(matrix)
        adj = sparse.coo_matrix(adj)
        np.save(direc+name+'.npy', adj)
# adjs()
def mutation_site():
    # ske = np.load('trainset/ssipe_train.npy')
    # ske = np.load('testset_2/ssipe_test_2.npy')
    ske = np.load('./s4169.npy')
    nodes_number = np.load('./gcn_features/nodes_number.npy')
    wild_features = np.load('./gcn_features/wild_features.npy')
    mut_features = np.load('./gcn_features/mut_features.npy')
    mutation_site = []
    before = 0
    for i in range(0, len(ske)):
        print(i)
        item = ske[i]
        # wild_path = 'testset_1/features/'+str(i)+'./wild.npy'
        # mut_path = 'testset_1/features/'+str(i)+'./mutant.npy'
        # wild = np.load(wild_path)
        # mut = np.load(mut_path)
        n = nodes_number[i]
        wild = wild_features[before:before+n, :]
        mut = mut_features[before:before+n, :]
        before += n
        temp_site = []

        # chains = item[1].split('_')[:-1]
        chains = item[1]
        wilds = item[2].split('_')[:-1]
        mutants = item[4].split('_')[:-1]
        ids = item[3].split('_')[:-1]
        # for j in range(0, len(chains)):
        #     ch = chains[j]
        #     w = wilds[j]
        #     m = mutants[j]
        #     d = ids[j]
        #     for index in range(0, wild.shape[0]):
        #         this_res = wild[index][-1]
        #         if this_res == ch + d:temp_site.append(index)
        for j in range(0, wild.shape[0]):
            if False not in (wild[j][-20:] == mut[j][-20:]):
                continue
            temp_site.append(j)
        if len(temp_site) != len(chains):
            print(item)
            print(temp_site)
            break
        mutation_site.append(temp_site)
    np.save('./gcn_features/mutation_site.npy', mutation_site)
# mutation_site()