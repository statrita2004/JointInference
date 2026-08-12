This is the repository for the paper :
[Joint likelihood-free inference of the number of selected single nucleotide polymorphisms and the selection coefficient in an evolving population](https://doi.org/10.1016/j.jtbi.2026.112544)

## Joint-Inference-of-selection-and-number-of-selected-target

- **abcpy**: A local repository containing necessary abcpy codebase.
- **input**: Contains the input haplotype dataset and mimiCREE2 Java code.
- **model_mimiCREE.py**: Contains simulation model by using mimiCREE2 [1].
- **Statistics_new.py**: Computes the summary statistics used for this study.
- **DataGenerator.py**: Code to simulate all the simulated data for the simulation study.
- **Data**: Contains all the simulated data created by DataGenerator.py for the simulation study.
- **RunSimulationExperiment.py**: Runs the inference for different simulation setups. 
- **Results**: Contains all the posteriors inferred by **RunSimulationExperiment.py**
- **AnalyseSimulationPosterior.py**: Analyses the posterior inferred by **RunSimulationExperiment.py**
- **YeastData**: Contains Yeast dataset, inferred posterior and **yeast.py** needed to create dataset corresponding to different windows. Also some figures.  
- **RunYeastExperiment.py**: Runs the inference for Yeast data. 

ABCpy pacakage [2] details available at [here](https://github.com/eth-cscs/abcpy)

mimiCREE2 user manual available at [here](https://sourceforge.net/p/mimicree2/wiki/Home/)

[1] Christos Vlachos and Robert Kofler. Mimicree2: Genome-wide forward simulations of evolve and resequencing studies. PLoS computational biology, 14(8):e1006413, 2018.

[2] Ritabrata Dutta, Marcel Schoengens, Lorenzo Pacchiardi, Avinash Ummadisingu, Nicole Widmer, Pierre Künzli,
Jukka-Pekka Onnela, and Antonietta Mira. Abcpy: A high-performance computing perspective to approximate
bayesian computation. Journal of Statistical Software, 100(7):1–38, 2021.
