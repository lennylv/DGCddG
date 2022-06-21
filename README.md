

DGCddG

############################################################################

Any more question, please contact to me : 20205227080@stu.suda.edu.cn

############################################################################

### install dependency:

conda create -n dgc python==3.7.10 -y

conda activate dgc

pip install  --no-cache-dir torch==1.7.0+cu110 -f  https://download.pytorch.org/whl/torch_stable.html

(pip3 install torch torchvision torchaudio)

pip install scipy

pip install sklearn


### Generate Examples:
- cd examples

Unzip the data set. 

I recommend using WinRAR on Windows to do this, as the zip command on Linux may cause unzipped files to be lost.

example: unzip train.zip

- python example.py --pdb T56 --mutation_chain G --residue_id 20 --wild E --mutant A

As an example, we prepare a 'T56.pdb' and two profiles for each chain of 'T56.pdb' and mutant chain, in this example, the mutant chain is 'G' and the residue 
at the position '20' is mutated to 'A' from 'E'.

### To test your own datasets, you need prepare profiles from PSI-Blast and HHblits for each chain and the mutation_chain.
In this example, the mutation is 'E' to 'A' in chain 'G', this 'T56.pdb' has 3 chains, i.e. 'ABG'. So we prepare the profiles as follow:

1: two profiles for chain A:
'T56_A.pssm',
'T56_A.hhm'

2: two profiles for chain B:
'T56_B.pssm',
'T56_B.hhm'

3: two profiles for chain G:
'T56_G.pssm',
'T56_G.hhm'

4: two profiles for chain mutated G (this sequence is generated from chain G with a substitute in residue 20, 'E' -> 'A'):

'mutant.hhm',
'mutant.pssm'

### If you have no idea to generate profiles from PSI-Blast or HHblits, you can download 'SPIDER3:Sequence-based prediction of structural features for proteins' from http://zhouyq-lab.szbl.ac.cn/download/, this software can generate two profiles for protein sequence to predict secondary structures.
you should move use this software with protein sequence file, if you hove no idea to generate protein sequences from protein pdb files, you can use generate_seq.py

For example, make sure that the pdb file and this script are in the same directory:

- cd generate_sequence
- python generate_seq.py --pdb T56 --mutation_chain G --residue_id 20 --wild E --mutant A

Than two profiles can be easily generated using the software SPIDER3 with these sequence files. (Please see its README file) 

Finally, move all '.pssm' and '.hhm' files to the directory 'examples/pdb_set/'

############################################################################

### Reproduce experimental results: 

before doing that, Unzip the data set and move it up to the next level. 

Again, I recommend using WinRAR on Windows to do this, as the zip command on Linux may cause unzipped files to be lost.

example: unzip train_1470/testset1/testset1.zip -> train_1470/

----------------------------------------------------------------------------
skempi2-m734, skempi2-2m888, skempi2-3m888, capri-t55, capri-t56:

- cd train_1470

(testset1: skempi2-m734, testset2: skempi2-2m888, testset2_multiple_o: skempi2-3m888, capri_t56: capri_t56, capri_t55_o: capri_t55)

take testset1 as an example:

- python model_load.py --dataset testset1

or

- python model_load.py --dataset testset1 --model_type 0

you can also re-train the model by

- python graphTrain.py --dataset testset1

----------------------------------------------------------------------------

spike-ace2-418:

- cd train_4169

(ace2: spike-ace2-418,)

- python model_load.py --dataset ace2

----------------------------------------------------------------------------

protein-level cv: ab645, s1131, s4169, s8838, s1131

- cd pro-cv

- python crossvalidation_1131.py




