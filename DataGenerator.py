import numpy as np
import sys
# Load simulation model
from model_mimiCREE import Mimicree
# Load probability model for prior
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform

### This file generates data for one configuration chosen from configuration matrix ####
### This chosen configuration is given as a command line input e.g. 2 ####
########################################################################################
chosen_configuration = int(sys.argv[1])
### This file generates all the data from MIMICREE used for simulation study ####
#################################################################################
### First we define the simulator model ###
# Defining the prior for selection coefficient
s1 = Uniform([[0], [0.2]], name='s1')
s2 = Uniform([[0], [0.2]], name='s2')
# Defining the prior for number of selected target
nns = DiscreteUniform([[0], [2]], name='nns')
# Recombination rate
recombrate = 2.5
# (1 = haploid_1000, 2 = diploid_1000, 3=yeast, 4=haploid_500, 5=diploid_500)
ploidy = 1
# Defining graphical model for simulations
MMC = Mimicree([s1, s2, nns, recombrate, ploidy], name='MMC')

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

## All total 29 setups to feed 0:28

for i in [chosen_configuration]:#range(Config_matrix.shape[0]):
    for j in range(20):
        input_vals = [Config_matrix[i,0], Config_matrix[i,1], int(Config_matrix[i,2]),
                      Config_matrix[i,3],Config_matrix[i,4], Config_matrix[i,5]]
        file_name = "Data/"
        for input_val in input_vals: file_name = file_name + str(input_val) + "_"
        file_name = file_name + str("fakedata")
        fakeobs = MMC.forward_simulate(input_vals, int(Config_matrix[i,5]))
        np.savez(file_name+"_"+str(j)+".npz", input_vals=input_vals, fakeobs=fakeobs)
