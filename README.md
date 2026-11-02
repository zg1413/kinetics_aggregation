# kinetics_aggregation
Repository for codes to compute how coordination numbers, association probabilities and cluster distributions change in time in concentrated electrolytes in non-equilibrium bulk simulations. The scripts are those used in an example salt-in-ionic liquid electrolyte [1]. Scripts are provided to convert LAMMPS dump files into the required format, compute g(r) to find appropriate real space cut-offs, compute the local associations in the electrolyte and the adjacency matrics for cluster distributions, and scripts to compare the theory predictions against the simulated results. The simulated data can be found here - [2] 

Here is the workflow which allows the results [1] to be reproduced from the data [2]:

1. Import.py - takes LAMMPS dump file of trajectory and saves the important coordinates for Li and TFSI O's as numpy binary matrices

2. pair_corr.py - computes pair correlation function between Li and O's in TFSI to determine suitable real-space cut-off for associations.

3. probability*.py - computes coordination numbers and adjacency matrix from the MD trajector
	3.1 - probability.py - computes the coordination numbers of TFSI O's around Na, and Na's around TFSI
	3.2 - probability_adj.py - computes the adjacency matrix for the Na-O's associations
	3.3 - probability_common.py - common functions used by 3.1 and 3.2

4. GRAPHS.py - computes cluster distribution from the corresponding adjecency matrix

5. PLOT_*.py - plotting functions 
	5.1 -  
	5.2 - 
	5.3 - 
	5.4 - 

Any questions can be directed to - zac.goodwin@materials.ox.ac.uk

References:

1. ZAH Goodwin Reversible Ionic Aggregetion Kinetics in Concentrated Electrolytes, Chemical Science, 17 (26), 13171-13185 (2026)
