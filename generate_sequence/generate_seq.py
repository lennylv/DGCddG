import numpy as np
def analyze_pdb(pdb, chain):
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
    return wuhu


def climain(pdb, pdb_path, mut_chain, residue_id, wild, mutant):

    from Bio.PDB import PDBParser as parser
    p = parser()
    structure = p.get_structure(pdb, pdb_path)
    chains = structure.get_chains()

    chains = [c.get_id() for c in chains]
    chain_residue = {}
    chain_residue_feature = {}

    for chain in chains:
        c = chain
        chain_residue[c] = []
        chain_residue[c].extend(  analyze_pdb(pdb_path, c) )
        # wild chain
        reslist = [a[0] for a in chain_residue[c]]
        f = open(pdb + '_' + c + '.seq', 'w')
        f.write('>' + pdb + '_' + c + '\n')
        f.write(''.join(reslist))
        f.close()

        if c == mut_chain:
            resid_list = [a[1] for a in chain_residue[mut_chain]]
            index = resid_list.index(residue_id.upper())
            reslist[index] = mutant

            f = open('mutant.seq', 'w')
            f.write('>mutant' + '\n')
            f.write(''.join(reslist))
            f.close()


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser(description='ddG')

    parser.add_argument('--pdb', default='T56', type=str)
    parser.add_argument('--mutation_chain', default='G', type=str)
    parser.add_argument('--residue_id', default='20', type=str)
    parser.add_argument('--wild', default='E', type=str)
    parser.add_argument('--mutant', default='A', type=str)

    args = parser.parse_args()
    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    pdb = args.pdb
    pdb_path = './' + pdb + '.pdb'
    mutation_chain = args.mutation_chain
    residue_id = args.residue_id
    wild = args.wild
    mutant = args.mutant
    climain(pdb, pdb_path, mutation_chain, residue_id=residue_id, wild=wild, mutant=mutant)
    # os.system('python model_load.py --dataset exampleset')

main()