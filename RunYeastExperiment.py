import numpy as np
import sys
import pylab as plt
import time

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

### This file runs inference for one window of Yeast data ####
### This chosen window is given as a command line input integer values between 6 and 12 ####
########################################################################################
chosen_window_yeast = int(sys.argv[1])
### This file generates all the data from MIMICREE used for simulation study ####
###########################################
### First we define the simulator model ###
# Defining the prior for selection coefficient
s1 = Uniform([[0], [0.2]], name='s1')
s2 = Uniform([[0], [0.2]], name='s2')
# Defining the prior for number of selected target
nns = DiscreteUniform([[0], [2]], name='nns')
# Recombination rate
recombrate = 0
# ploidy (1 = haploid_1000, 2 = diploid_1000, 3=yeast, 4=haploid_500, 5=diploid_500)
# ploidy (6 = yeast_140, 7 = yeast_141_280 etc.)
ploidy = chosen_window_yeast
# Defining graphical model for simulations
MMC = Mimicree([s1, s2, nns, recombrate, ploidy], name='MMC')
# Specify the summary statistics function
if chosen_window_yeast == 12:
    statcalc = LogitTransformStat(degree=1,cross=False, num_snp=160)
else:
    statcalc = LogitTransformStat(degree=1, cross=False, num_snp=140)
# Specify the distance function
distance_calculator = EnergyDistance(statcalc)
# Specify the kernel function
kernel = DefaultKernel([s1, s2,nns])
# define backend (Choose BackendDummy if running sequentially, for parallelized run use BackendMPI)
backend = BackendDummy()
#backend = BackendMPI()
# Specify sampler
sampler = PMCABC([MMC], [distance_calculator], backend, kernel, seed=1)

if chosen_window_yeast == 6:
    ploidy_type = 'yeast_140'
elif chosen_window_yeast == 7:
    ploidy_type = 'yeast_141_280'
elif chosen_window_yeast == 8:
    ploidy_type = 'yeast_281_420'
elif chosen_window_yeast == 9:
    ploidy_type = 'yeast_421_560'
elif chosen_window_yeast == 10:
    ploidy_type = 'yeast_561_700'
elif chosen_window_yeast == 11:
    ploidy_type = 'yeast_701_840'
elif chosen_window_yeast == 12:
    ploidy_type = 'yeast_841_1000'

data_file_name = "YeastData/"+ploidy_type+"_2_3.npz"
result_file_name = "YeastData/"+ploidy_type+str("_posterior_2")
realdata = [np.load(data_file_name)['realobs'][ind,:] for ind in range(2)]

#data_file_name = "YeastData/"+ploidy_type+"_12_3.npz"
#realdata = [np.load(data_file_name)['realobs'][ind,:] for ind in range(12)]
# realstats = statcalc.statistics(realdata)
# plt.figure()
# for ind in range(12):
#     plt.plot(realstats[ind,:], label=str(ind))
# plt.xlabel('SNP positions')
# plt.ylabel('Summary statistics values')
# plt.savefig('YeastData/summarystats_'+str(ploidy_type)+'.png')

# t0 = time.time()
# print('Inference now running for '+str(chosen_window_yeast)+'-th window of yeast:')
# journal_pmcabc = sampler.sample([realdata], steps=5, epsilon_init=[10],
#                                 n_samples=50, n_samples_per_param=50,
#                                 epsilon_percentile=50, covFactor=2, full_output=1,
#                                 path_to_save_journal=result_file_name)
# print('Inference completed in '+str(time.time()-t0)+' seconds')
# # # #journal_pmcabc.plot_posterior_distr(path_to_save=result_file_name+".png")