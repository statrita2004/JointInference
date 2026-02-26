This is the source code for the paper :
Joint likelihood-free inference of the number of selected single nucleotide polymorphisms and the selection coefficient in an evolving population
https://www.biorxiv.org/content/10.1101/2022.09.20.508756v1

## Joint-Inference-of-selection-and-number-of-selected-target

- **abcpy**: A local repository containing necessary abcpy codebase.
- **input**: Contains the input haplotype dataset and mimiCREE2 Java code.
- **model_mimiCREE.py**: Contains simulation model by using mimiCREE2 [2].
- **Statistics_new.py**: Computes the summary statistics used for this study.
- **DataGenerator.py**: Code to simulate all the simulated data for the simulation study.
- **Data**: Contains all the simulated data created by DataGenerator.py for the simulation study.
- **RunSimulationExperiment.py**: Runs the inference for different simulation setups. 
- **Results**: Contains all the posteriors inferred by **RunSimulationExperiment.py**
- **AnalyseSimulationPosterior.py**: Analyses the posterior inferred by **RunSimulationExperiment.py**
- **YeastData**: Contains Yeast dataset, inferred posterior and **yeast.py** needed to create dataset corresponding to different windows. Also some figures.  
- **RunYeastExperiment.py**: Runs the inference for Yeast data. 

ABCpy pacakage details available at [here](https://github.com/eth-cscs/abcpy)

mimiCREE2 user manual available at [here](https://sourceforge.net/p/mimicree2/wiki/Home/)

[1] Carlo Albert, Hans R Künsch, and Andreas Scheidegger. A simulated annealing approach to approximate bayes computations. Statistics and computing, 25(6):1217–1232, 2015.

[2] Christos Vlachos and Robert Kofler. Mimicree2: Genome-wide forward simulations of evolve and resequencing studies. PLoS computational biology, 14(8):e1006413, 2018.

[3] Hui Zou. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476):1418–1429, 2006. 15

[4] Thomas Taus, Andreas Futschik, and Christian Schlötterer. Quantifying selection with pool-seq time series data. Molecular biology and evolution, 34(11):3023–3034, 2017.

[5] Ritabrata Dutta, Marcel Schoengens, Lorenzo Pacchiardi, Avinash Ummadisingu, Nicole Widmer, Pierre Künzli,
Jukka-Pekka Onnela, and Antonietta Mira. Abcpy: A high-performance computing perspective to approximate
bayesian computation. Journal of Statistical Software, 100(7):1–38, 2021.
