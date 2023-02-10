These predictors used different labels during training. 
The labels of SSIPe and our method are dMutant - dWild, then GeoPPI and mCSM-PPI2 are dWild - dMutant
Please check their papers.

Besides, if the label is dWild-dMutant, ddG>0 represents the encreased affinity (please check mCSM-PPI2 paper).
Therefore, the output of SSIPe and DGCddG need a reverse before calculating the Kendall coefficient.
