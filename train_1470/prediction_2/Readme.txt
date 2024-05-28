The labels of SSIPe and our method are dMutant - dWild, then GeoPPI and mCSM-PPI2 are dWild - dMutant,
Therefore, the output of SSIPe and DGCddG need a reverse before calculating the Kendall coefficient.

The capri_t55.npy and capri_t56.npy were calculated based on the graphs constructed from dimers.
Therefore, the results here are -0.056 and 0.193 for capri_t55 and capri_t56, respectively.

The paper records the result under graphs constructed from multi-chain complex, i.e. -0.0216 and -0.06 for capri_t55 and capri_t56.
We will update them soon.

For T56.pdb we failed to generate the secondary structure and failed to predicted ss, then we use the most homology protein sequence (>90%) in PDB via blast
to get the secondary structure as the similar values.

We also quit all the aux information to re-train the model and get results as: T55:0.02, T56:0.027
