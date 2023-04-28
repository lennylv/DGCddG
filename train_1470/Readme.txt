1. For multi-mutation, aux_features.npy saved the mean value of each single mutation:
For example: an item inclues two single-mutations [pdb, chain1_chain2, wild1_wild2, resid1_resid2, mutant1_mutant2, ddG]

For [chain1 wild1 resid1 mutant1]: dssp module outputs [ss1, asa1, phi1, psi1]
For [chain2 wild2 resid2 mutant2]: dssp module outputs [ss2, asa2, phi2, psi2]

Then for this item, aux_features.npy saves [(ss1 + ss2)/2, (asa1 + asa2)/2, (phi1+phi2)/2, (psi1+psi2)/2]

2. The capri_t55.npy and capri_t56.npy were calculated based on the graphs constructed from dimers. Therefore, the results here are -0.056 and 0.193 for capri_t55 and capri_t56, respectively. The paper records the result under graphs constructed from multi-chain complex, e.g. -0.0216 and -0.06 for capri_t55 and capri_t56.

3. Complement Ablation experiments for four aux features (pearson & kendall):

        2M734   2M888   3M888    Capri-t55    Capri-t56
SS+ASA  0.648   0.274   0.281     -0.0216       -0.06
Four    0.659   0.251   0.255     -0.0534       -0.1
