1. For multi-mutation, aux_features.npy saved the mean value of each single mutation:
For example: an item inclues two single-mutations [pdb, chain1_chain2, wild1_wild2, resid1_resid2, mutant1_mutant2, ddG]

For [chain1 wild1 resid1 mutant1]: dssp module outputs [ss1, asa1, phi1, psi1]
For [chain2 wild2 resid2 mutant2]: dssp module outputs [ss2, asa2, phi2, psi2]

Then for this item, aux_features.npy saves [(ss1 + ss2)/2, (asa1 + asa2)/2, (phi1+phi2)/2, (psi1+psi2)/2]
