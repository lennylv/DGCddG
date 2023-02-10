1.These predictors used different labels during training. 
The labels of SSIPe and our method are dMutant - dWild, then GeoPPI and mCSM-PPI2 are dWild - dMutant
Please check their papers.

2.Besides, if the label is dWild-dMutant, ddG>0 represents the encreased affinity (please check mCSM-PPI2 paper).
Therefore, the output of SSIPe and DGCddG need a reverse before calculating the Kendall coefficient.

3.The directory of train_s4169 needs to be deleted, the ace2 directory needs to be moved to ../train_1470
