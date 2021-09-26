import numpy as np
def total_AreaVolumeEnergy(file_path):
    feature = []
    areaVolumeFile = open(file_path + '.areavolume')
    Area = float(areaVolumeFile.readline())
    Volume = float(areaVolumeFile.readline())
    areaVolumeFile.close()

    engFile = open(file_path+'.eng')
    engFile.readline()
    SolvEng = engFile.readline()
    SolvEng = float( SolvEng )

    return [Area, Volume, SolvEng]

def construct_residue_level_feature(residue):
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
    AA = residue
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
    

    return residueFeature

def generate_residue_features():
    wild = []
    mutant = []
    ske = np.load('skempi_4169.npy').astype(str)
    for i in range(0, len(ske)):
        item = ske[i]
        pdbname = item[0]
        chain = item[1]
        # chain_mut = chain+'_mut'
        residue_wild = []
        residue_mut = []
        # wild_areaVolumeEnergy = total_AreaVolumeEnergy('wild_files/'+pdbname+'/'+chain)
        # mut_areaVolumeEnergy = total_AreaVolumeEnergy('s1131_files/'+str(i)+'/'+chain_mut)

        # residue_wild.extend(wild_areaVolumeEnergy)
        # residue_mut.extend(mut_areaVolumeEnergy)

        residue_wild.extend( construct_residue_level_feature(item[2]) )
        residue_mut.extend( construct_residue_level_feature(item[4]) )

        wild.append( residue_wild )
        mutant.append( residue_mut )
    wild = np.asarray(wild, float)
    mutant = np.asarray(mutant, float)
    print(wild.shape)
    print(mutant.shape)
    # np.save('wild_residue.npy', wild)
    # np.save('mutant_residue.npy', mutant)
    w_m = np.concatenate((wild, mutant, mutant - wild), 1)
    np.save('s4169/normal_residue.npy', w_m)
# generate_residue_features()


def get_near_resid(coordinate, resid, pdb, mut_chain):
    pdb_id = 's645/temp_pdb/'+pdb+'.pdb'
    f = open(pdb_id)
    lines = f.readlines()
    near_resid = {}
    restype = []
    for line in lines:
        if line[:4]!='ATOM':continue
        this_resid = line[22:27].strip()
        this_chain = line[20:22].strip()
        if this_chain == mut_chain and this_resid == resid:
            continue
        if this_chain in near_resid.keys():
            if  this_resid in near_resid[this_chain]:continue
        # atom_type =
        this_coordinate = np.array([ float(line[27:38]),float(line[38:46]),float(line[46:54]) ])
        distance = np.linalg.norm(this_coordinate - coordinate)
        if distance < 10:
            # near_resid.append(this_resid)
            restype.append(line[17:20])
            if this_chain not in near_resid.keys():
                near_resid[this_chain] = [this_resid]
            else:
                near_resid[this_chain].append(this_resid)
    f.close()
    return near_resid, restype

def construct_environment_feature(coordinate, mut_resid, pdb, mut_chain):
    from Bio.PDB.Polypeptide import three_to_one
    near_resid, near_restype = get_near_resid(coordinate, mut_resid, pdb, mut_chain)
    #feature
    def AAcharge(AA):
        if AA in ['D','E']:
            return -1.
        elif AA in ['R','H','K']:
            return 1.
        else:
            return 0.
    FeatureEnvironment = []
    NearSeq = []
    helpNearSeq = []
    #
    for i in near_restype:
        try:
            NearSeq.append(three_to_one(i))
        except:
            print('quit ',i)
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
    for Group in Groups:
        cnt = 0.
        for AA in NearSeq:
            if AA in Group:
                cnt += 1.
        FeatureEnvironment.append(cnt)
        FeatureEnvironment.append(cnt/max(1., float(len(NearSeq))))
    Vol = []; Hyd = []; Area = []; Wgt = []; Chg = []
    for AA in NearSeq:
        Vol.append(AAvolume[AA])
        Hyd.append(AAhydropathy[AA])
        Area.append(AAarea[AA])
        Wgt.append(AAweight[AA])
        Chg.append(AAcharge(AA))
    Vol = np.asarray(Vol)
    Hyd = np.asarray(Hyd)
    Area = np.asarray(Area)
    Wgt = np.asarray(Wgt)
    if len(NearSeq) == 0:
        FeatureEnvironment.extend([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    else:
        FeatureEnvironment.extend([np.sum(Vol), np.sum(Vol)/float(len(NearSeq)), np.var(Vol)])
        FeatureEnvironment.extend([np.sum(Hyd), np.sum(Hyd)/float(len(NearSeq)), np.var(Hyd)])
        FeatureEnvironment.extend([np.sum(Area), np.sum(Area)/float(len(NearSeq)), np.var(Area)])
        FeatureEnvironment.extend([np.sum(Wgt), np.sum(Wgt)/float(len(NearSeq)), np.var(Wgt)])
    FeatureEnvironment.append(sum(Chg))
    return FeatureEnvironment
def get_CA_coordinate(pdb, mut_resid, mut_chain):
    pdb_id = 's645/temp_pdb/'+pdb+'.pdb'
    f = open(pdb_id)
    lines = f.readlines()
    for line in lines:
        if line[:4]!='ATOM':continue
        this_chain = line[20:22].strip()
        if this_chain != mut_chain:continue
        this_resid = line[22:27].strip()
        if this_resid != mut_resid:
            # print(this_resid, mut_resid)
            continue
        atom_type = line[12:16].strip()
        if atom_type != 'CA':continue
        this_coordinate = np.array([ float(line[27:38]),float(line[38:46]),float(line[46:54]) ])
        break
    f.close()
    return this_coordinate
def environment():
    # ab = np.load('ab645.npy')
    ab = np.load('skempi_4169.npy')
    i = -1
    # environ = np.empty((645, 25), dtype=float)
    environ = np.empty((4169, 25), dtype=float)
    for item in ab:
        i+=1
        print(i)
        # if i<3699:continue
        # pdb = item[0].split('#')[0]
        pdb = item[0]
        mut_resid = item[3].upper()
        mut_chain = item[1]
        coordinate = get_CA_coordinate(pdb, mut_resid, mut_chain)
        feature_environment = construct_environment_feature(coordinate, mut_resid, pdb, mut_chain)
        feature_environment = np.array(feature_environment)
        # print(feature_environment.shape)
        environ[i] = feature_environment
    np.save('environment.npy',environ)

def dssp():
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
    # ske = np.load('skempi_4169.npy')
    ske = np.load('ssipe/')
    FeatureSeq = []
    # Structure based DSSP
    i = -1
    for item in ske:
        i += 1
        # if 'a' not in item[3]:continue
        # if i<777:continue
        # print(item)
        print(i)
        pdb = item[0]
        chainid = item[1]
        parser = PDBParser()
        pdb_path = './dssp/' + pdb + '_' + chainid + '/' +''.join(item[:5])+'.pdb'
        # pdb_path = './A_mut.pqr'
        structure = parser.get_structure('prot', pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, './dddd')
        dic_path = './dssp/' + pdb + '_' + chainid +'/'+ chainid + '_resdic.txt'
        f = open(dic_path)
        dic = eval(f.read())
        for key in dic.keys():
            if dic[key] == item[3].upper():
                resid = key
                break
        
        # print(resid)
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
        print(temp)
        FeatureSeq.append(temp)
        # break
    FeatureSeq = np.array(FeatureSeq)
    np.save('dssp.npy',FeatureSeq)


generate_residue_features()
environment()
dssp()
