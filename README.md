

DGCddG

############################################################################

 Any more questions, please do not hesitate to contact me: 20234027015@stu.suda.edu.cn

############################################################################


### install dependency:

- conda create -n dgc python==3.7.10 -y

- conda activate dgc

- conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

- pip install scipy

- pip install sklearn

- pip install Bio

### Generate Examples:
- cd examples

Unzip the data set. 

Due to the limitation of Github, I have to split the zip file which is >25MB into several files.

You can WinRAR on Windows to unzip the trainset.zip, as the zip command on Linux may cause unzipped files to be lost.

or

You can use cat | tar command on Linux

example: unzip train.zip

- cat ./trainset.tar.gz* | tar -xzv
- python example.py --pdb T56 --mutation_chain G --residue_id 20 --wild E --mutant A

As an example, we prepare a 'T56.pdb' and two profiles for each chain of 'T56.pdb' and mutant chain, in this example, the mutant chain is 'G' and the residue 
at the position '20' is mutated to 'A' from 'E'.

############################################################################

### Reproduce experimental results: 

- cd train_1470

before doing that, Unzip the data set. 

Again, you can use WinRAR on Windows to do this, as the zip command on Linux may cause unzipped files to be lost.

example: unzip train_1470/testset1/testset1.zip -> train_1470/

or

use: cat | tar

- cat ./trainset/trainset.tar.gz* | tar -xzv
- cat ./testset1/testset1.tar.gz* | tar -xzv
- cat ./testset2/testset2.tar.gz* | tar -xzv
- cat ./testset2_multiple/testset2_multiple_o.tar.gz* | tar -xzv
- cat ./capri_t55/capri_t55.tar.gz* | tar -xzv
- cat ./capri_t56/capri_t56.tar.gz* | tar -xzv

----------------------------------------------------------------------------
skempi2-m734, skempi2-2m888, skempi2-3m888, capri-t55, capri-t56:

(testset1: skempi2-m734, testset2: skempi2-2m888, testset2_multiple_o: skempi2-3m888, capri_t56: capri_t56, capri_t55: capri_t55)

take testset1 as an example:

- python model_load.py --dataset testset1

or

- python model_load.py --dataset testset1 --model_type 0

The prediction ddg is saved into the ./prediction/tesetse1.npy

- cd prediction
- import numpy as np
- prediction = np.load('testset1.npy')

you can also re-train the model by

- python graphTrain.py --dataset testset1

----------------------------------------------------------------------------


----------------------------------------------------------------------------

protein-level cv: ab645, s1131, s4169, s8838

- cd pro-cv
- cat ./s1131/s1131.tar.gz* | tar -xzv
- python crossvalidation_1131.py

mutation-level:
- python crossvalidation_1131.py --cv 1

----------------------------------------------------------------------------

Examples/* will be updated in the future

Besides, all Profiles are available:

https://pan.baidu.com/s/1kPErcJ_7-vkNnoxU-YKRcg

password: zjml

----------------------------------------------------------------------------
To test your own datasets, you need prepare profiles from PSI-Blast and HHblits for each chain and the mutation_chain.
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

### If you have no idea to generate profiles from PSI-Blast or HHblits. You can check the profile_generation.pdf and you can generate the two profiles step by step

If you have no idea to generate protein sequences from protein pdb files, you can use generate_seq.py

For example, make sure that the pdb file and this script are in the same directory:

- cd pdb_set
- python generate_seq.py --pdb T56 --mutation_chain G --residue_id 20 --wild E --mutant A

Then two profiles can be easily generated using the software SPIDER3 with these sequence files. (Please see its README file) 

- ./SPD3-numpy/run_list.sh ./*.seq

If you do not want to install PSI-Blast or HHblits, you can use their online webservers. Or you can send your data to me and I will do my best to generate  all profiles for you.




