import numpy as np
import time
import sys
# Load simulation model
from model_mimiCREE import Mimicree
# Load probability model for prior
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform
# Load summary statistics
from Statistics_new import LogitTransformStat
# Load distance measure
from abcpy.distances import EnergyDistance
# Load distance kernel
from abcpy.perturbationkernel import DefaultKernel
# Load inference scheme
from abcpy.inferences import PMCABC
# Load backend
from abcpy.backends import BackendDummy, BackendMPI

### This file runs inference for one configuration chosen from configuration matrix ####
### This chosen configuration is given as a command line input e.g. 2 ####
########################################################################################
chosen_configuration = int(sys.argv[1])
start_data_index = int(sys.argv[2])
###########################################
### First we define the simulator model ###
# Defining the prior for selection coefficient
s1 = Uniform([[0], [0.2]], name='s1')
s2 = Uniform([[0], [0.2]], name='s2')
# Defining the prior for number of selected target
nns = DiscreteUniform([[0], [2]], name='nns')
# Recombination rate
recombrate = 2.5
# 1 for haploid, 2 for diploid
ploidy = 1
# Defining graphical model for simulations
MMC = Mimicree([s1, s2, nns, recombrate, ploidy], name='MMC')
# Specify the summary statistics function
statcalc = LogitTransformStat(degree=1,cross=False, num_snp=289)
# Specify the distance function
distance_calculator = EnergyDistance(statcalc)
# Specify the kernel function
kernel = DefaultKernel([s1, s2,nns])
# define backend (Choose BackendDummy if running sequentially, for parallelized run use BackendMPI)
#backend = BackendDummy()
backend = BackendMPI()
# Specify sampler
sampler = PMCABC([MMC], [distance_calculator], backend, kernel, seed=1)

### Define the experimental configurations to simulate data #####
### We define this as a matrix, whose columns are correspondingly s1, s2, nns, recombination rate,
### ploidy (1 = haploid_1000, 2 = diploid_1000, 3=yeast, 4=haploid_500, 5=diploid_500), no_replicates. Each
### row defines an experimental configuration #####
Config_matrix = np.array([[0, 0, 0, 0, 1, 20],
                            [0.02, 0, 1, 0, 1, 20],
                            [0.05, 0, 1, 0, 1, 20],
                            [0.07, 0, 1, 0, 1, 20],
                            [0.05, 0.02, 2, 0, 1, 20],
                            [0.05, 0.07, 2, 0, 1, 20],
                            [0.07, 0.09, 2, 0, 1, 20],
                            [0.07, 0.09, 2, 0, 1, 5],
                            [0.07, 0.09, 2, 0, 1, 10],
                            [0, 0, 0, 0, 2, 20],
                            [0.07, 0, 1, 0, 2, 20],
                            [0.1, 0, 1, 0, 2, 20],
                            [0.14, 0, 1, 0, 2, 20],
                            [0.18, 0, 1, 0, 2, 20],
                            [0.05, 0.05, 2, 2.5, 2, 20],
                            [0.07, 0.09, 2, 2.5, 2, 20],
                            [0.09, 0.12, 2, 2.5, 2, 20],
                            [0.14, 0.18, 2, 2.5, 2, 20],
                            [0.14, 0.18, 2, 0, 2, 20],
                            [0.14, 0.18, 2, 15, 2, 20],
                            [0.14, 0.18, 2, 30, 2, 20],
                            [0.14, 0.18, 2, 49, 2, 20],
                            [0.05, 0.02, 2, 0, 4, 20],
                            [0.05, 0.07, 2, 0, 4, 20],
                            [0.07, 0.09, 2, 0, 4, 20],
                            [0.05, 0.05, 2, 2.5, 5, 20],
                            [0.07, 0.09, 2, 2.5, 5, 20],
                            [0.09, 0.12, 2, 2.5, 5, 20],
                            [0.14, 0.18, 2, 2.5, 5, 20]])

## All total 30 setups to feed 0:29

for j in range(start_data_index, 20):
    input_vals = [Config_matrix[chosen_configuration,0],
                  Config_matrix[chosen_configuration,1],
                  int(Config_matrix[chosen_configuration,2]),
                  Config_matrix[chosen_configuration,3],
                  Config_matrix[chosen_configuration,4],
                  Config_matrix[chosen_configuration,5]]
    print('Inference now running for '+str(j)+'-st/nd/th replicate fakedata of configuration:'
          +str(input_vals))
    file_name = "/"
    for input_val in input_vals: file_name = file_name + str(input_val) + "_"
    data_file_name = "Data"+file_name + str("fakedata")+"_"+str(j)+".npz"
    result_file_name = "Results"+file_name+str("posterior")+"_"+str(j)
    fakedata = [np.load(data_file_name)['fakeobs'][ind,:] for ind in range(int(input_vals[5]))]
    t0 = time.time()
    journal_pmcabc = sampler.sample([fakedata], steps=6, epsilon_init=[10],
                                    n_samples=100, n_samples_per_param=100,
                                    epsilon_percentile=50, covFactor=2, full_output=1,
                                    path_to_save_journal=result_file_name)
    print('Inference completed in '+str(time.time()-t0)+' seconds')
    #journal_pmcabc.plot_posterior_distr(path_to_save=result_file_name+".png")