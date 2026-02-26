import numpy as np
import sys
import pylab as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

from abcpy.output import Journal

### This file analyses inferred posterior for one configuration chosen from configuration matrix ####
### This chosen configuration is given as a command line input e.g. 2 ####
########################################################################################
chosen_configuration = int(sys.argv[1])
ind_class = int(sys.argv[2])
###########################################
#######################################################################
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

inferred_nns = np.zeros(shape=(20,))
inferred_cond_s1s2_mode = np.zeros(shape=(20,2))
#inferred_joint_s1s2_mode = np.zeros(shape=(10,2))
for j in range(20):
    input_vals = [Config_matrix[chosen_configuration,0],
                  Config_matrix[chosen_configuration,1],
                  int(Config_matrix[chosen_configuration,2]),
                  Config_matrix[chosen_configuration,3],
                  Config_matrix[chosen_configuration,4],
                  Config_matrix[chosen_configuration,5]]
    print('Analysing now inferred posterior for '+str(j+1)+'-st/nd/th replicate fakedata of configuration:'
          +str(input_vals))
    file_name = "/"
    for input_val in input_vals: file_name = file_name + str(input_val) + "_"
    result_file_name = "Results"+file_name+str("posterior")+"_"+str(j)
    journal = Journal.fromFile(result_file_name+".jnl")
    samples_as_nparray = np.concatenate([np.array(journal.get_parameters(-1)['s1']).squeeze().reshape(-1, 1),
                                        np.array(journal.get_parameters(-1)['s2']).squeeze().reshape(-1, 1),
                                        np.array(journal.get_parameters(-1)['nns']).squeeze().reshape(-1, 1)],axis=1)
    weights = journal.get_weights()
    ## Compute marginal posterior probability of p0=P(nns=0|x), p0=P(nns=1|x), p0=P(nns=2|x) ##
    ## marginalizing s1, s2 out ##
    p0 = sum(weights[samples_as_nparray[:,2]==0])
    p1 = sum(weights[samples_as_nparray[:,2]==1])
    p2 = sum(weights[samples_as_nparray[:,2]==2])
    # # Infer the number of selected location by choosing the mode
    inferred_nns[j] = np.array([p0, p1, p2]).argmax()
    print('marginal posterior probability: (p0, p1, p2) = ('+str(p0)+','+str(p1)+
          ','+str(p2)+'). Inferred nns: '+str(inferred_nns[j]))
    samples_cond_inferred_nns = samples_as_nparray[samples_as_nparray[:,2]==inferred_nns[j],:2]
    weights_cond_inferred_nns = weights[samples_as_nparray[:,2]==inferred_nns[j]]
    if inferred_nns[j] == 1:
        values = np.vstack([samples_cond_inferred_nns[:,0].T])
        kernel_marginal = gaussian_kde(values, weights=weights_cond_inferred_nns.squeeze(), bw_method='scott')
        def ap_pdf_marginal(x): return -np.log(kernel_marginal(x))
        x0 = 0.01
        res_marginal = minimize(ap_pdf_marginal, x0, method='nelder-mead',options={'xatol': 1e-8})
        inferred_cond_s1s2_mode[j, :] = [res_marginal.x[0], 0]
        print('Conditional posterior mode of (s1, s2): ' + str(inferred_cond_s1s2_mode[j, :]))
    if inferred_nns[j] == 2:
        values = np.vstack([samples_cond_inferred_nns[:,0].T, samples_cond_inferred_nns[:,1].T])
        kernel_marginal = gaussian_kde(values, weights=weights_cond_inferred_nns.squeeze(), bw_method='scott')
        def ap_pdf_marginal(x): return -np.log(kernel_marginal([x[0], x[1]]))
        x0 = [0.01, 0.01]
        res_marginal = minimize(ap_pdf_marginal, x0, method='nelder-mead',options={'xatol': 1e-8})
        inferred_cond_s1s2_mode[j, :] = res_marginal.x
        print('Conditional posterior mode of (s1, s2): '+str(res_marginal.x))

print('True s1: '+str(input_vals[0])+
      ', True s2: '+str(input_vals[1])+', True nns: '+str(input_vals[2])+
      ', True rec rate: '+str(input_vals[3])+', hap/dip-loid: '+str(input_vals[4])+
      ', number of replicates: '+str(input_vals[5]))
print('Inferred nns: '+str(inferred_nns))
print('Conditional posterior mode of (s1, s2): '+str(inferred_cond_s1s2_mode))
mm = np.mean(inferred_cond_s1s2_mode[inferred_nns == ind_class, :], axis=0)
vv = np.var(inferred_cond_s1s2_mode[inferred_nns == ind_class, :], axis=0)
print('('+str(round(mm[0], 3))+', '+str(round(mm[1],3))+') / ('+str(round(vv[0],5))+', '+str(round(vv[1],5))+')')




