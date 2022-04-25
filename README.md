

DGCddG

############################################################################

install:

conda create -n dgc python==3.7.10 -y

conda activate dgc

pip install  --no-cache-dir torch==1.7.0+cu110 -f  https://download.pytorch.org/whl/torch_stable.html

pip install scipy

pip install skearn

############################################################################

to Reproduce experimental results: 
before doing that, Unzip the data set and move it up to the next level. example: unzip train_1470/testset1/testset1.zip -> train_1470/

----------------------------------------------------------------------------
skempi2-m734, skempi2-2m888, skempi2-3m888, capri-t55, capri-t56:

cd train_1470

(testset1: skempi2-m734, testset2: skempi2-2m888, testset2_multiple: skempi2-3m888)

take testset1 as an example:

python model_load.py --dataset testset1

you can also re-train the model by

python graphTrain.py --dataset testset1

----------------------------------------------------------------------------

ace2-418, spike-540:

cd train_4169

(ace2: ace2-418, deep_sars2: spike-540)

python model_load.py --dataset ace2

you can also re-train the model by

python graphTrain_4169.py --dataset ace2

----------------------------------------------------------------------------

protein-level cv: ab645, s1131, s4169, s8838, s1131

cd pro-cv

python crossvalidation_1131.py
