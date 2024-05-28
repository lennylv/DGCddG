### Noticed by recently published work, DDMut-PPI, https://doi.org/10.1093/nar/gkae412 , Due to the mcsm-ppi2 server randomly misorders the original input entries when executing tasks, in the SPIKE-ACE2 data set, we used the wrong order as the calculation indicator. 
### This needs to be corrected and the actual value of kendall for mcsm-PPI2 is 0.246

1.These predictors used different labels during training. 
The labels of SSIPe and our method are dMutant - dWild, then GeoPPI and mCSM-PPI2 are dWild - dMutant
Please check their papers.

2.Besides, if the label is dWild-dMutant, ddG>0 represents the encreased affinity (please check mCSM-PPI2 paper).
Therefore, the output of SSIPe and DGCddG need a reverse before calculating the Kendall coefficient.