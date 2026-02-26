import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

from abcpy.output import Journal

### This file runs inference for one window of Yeast data ####
### This chosen window is given as a command line input integer values between 6 and 12 ####
########################################################################################
chosen_window_yeast = int(sys.argv[1])
iter_posterior = int(sys.argv[2])
### This file generates all the data from MIMICREE used for simulation study ####
###########################################

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


result_file_name = "YeastData/"+ploidy_type+str("_posterior_2")

inferred_nns = np.zeros(shape=(1,))
inferred_cond_s1s2_mode = np.zeros(shape=(1,2))
#inferred_joint_s1s2_mode = np.zeros(shape=(10,2))
k = iter_posterior
journal = Journal.fromFile(result_file_name+".jnl")
samples_as_nparray = np.concatenate([np.array(journal.get_parameters(k)['s1']).squeeze().reshape(-1, 1),
                                    np.array(journal.get_parameters(k)['s2']).squeeze().reshape(-1, 1),
                                    np.array(journal.get_parameters(k)['nns']).squeeze().reshape(-1, 1)],axis=1)
weights = journal.get_weights(k)
## Compute marginal posterior probability of p0=P(nns=0|x), p0=P(nns=1|x), p0=P(nns=2|x) ##
## marginalizing s1, s2 out ##
p0 = sum(weights[samples_as_nparray[:,2]==0])
p1 = sum(weights[samples_as_nparray[:,2]==1])
p2 = sum(weights[samples_as_nparray[:,2]==2])
# # Infer the number of selected location by choosing the mode
inferred_nns = np.array([p0, p1, p2]).argmax()
print('marginal posterior probability: (p0, p1, p2) = ('+str(p0)+','+str(p1)+
      ','+str(p2)+'). Inferred nns: '+str(inferred_nns))
samples_cond_inferred_nns = samples_as_nparray[samples_as_nparray[:,2]==inferred_nns,:2]
weights_cond_inferred_nns = weights[samples_as_nparray[:,2]==inferred_nns]
if inferred_nns == 1:
    values = np.vstack([samples_cond_inferred_nns[:,0].T])
    kernel_marginal = gaussian_kde(values, weights=weights_cond_inferred_nns.squeeze(), bw_method='scott')
    def ap_pdf_marginal(x): return -np.log(kernel_marginal(x))
    x0 = 0.01
    res_marginal = minimize(ap_pdf_marginal, x0, method='nelder-mead',options={'xatol': 1e-8})
    inferred_cond_s1s2_mode = [res_marginal.x[0], 0]
    print('Conditional posterior mode of (s1, s2): ' + str(inferred_cond_s1s2_mode))
if inferred_nns == 2:
    values = np.vstack([samples_cond_inferred_nns[:,0].T, samples_cond_inferred_nns[:,1].T])
    kernel_marginal = gaussian_kde(values, weights=weights_cond_inferred_nns.squeeze(), bw_method='scott')
    def ap_pdf_marginal(x): return -np.log(kernel_marginal([x[0], x[1]]))
    x0 = [0.01, 0.01]
    res_marginal = minimize(ap_pdf_marginal, x0, method='nelder-mead',options={'xatol': 1e-8})
    inferred_cond_s1s2_mode = res_marginal.x
    print('Conditional posterior mode of (s1, s2): '+str(inferred_cond_s1s2_mode))
    print(ap_pdf_marginal(inferred_cond_s1s2_mode))

print('Following are the analysed results for '+str(chosen_window_yeast)+'-th window of yeast:')
print('Inferred nns: '+str(inferred_nns))
print('Conditional posterior mode of (s1, s2): '+str(inferred_cond_s1s2_mode))

if inferred_nns == 2:
    xmin, xmax = 0, 0.2
    ymin, ymax = 0, 0.2
    # then you need to perform the KDE and plot the contour of posterior
    names = [r'$s_2$', r'$s_1$']
    figsize_actual = 4 * len(names)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel_marginal(positions).T, X.shape)
    plt.figure()
    plt.plot([xmin, xmax], [inferred_cond_s1s2_mode[1], inferred_cond_s1s2_mode[1]], "red", markersize='20',
                    linestyle='solid')
    plt.plot([inferred_cond_s1s2_mode[0], inferred_cond_s1s2_mode[0]], [ymin, ymax], "red", markersize='20',
                    linestyle='solid')
    CS = plt.contour(X, Y, Z, 14, linestyles='solid')
    plt.clabel(CS, fontsize=figsize_actual / len(names) * 2.25, inline=1)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.savefig(result_file_name+".png")

if inferred_nns == 1:
    xmin, xmax = [0, .2]
    positions = np.linspace(xmin, xmax, 100)
    plt.figure()
    values = kernel_marginal(positions)
    plt.plot(positions, values, color='k', linestyle='solid', lw="1",
                          alpha=1, label="Density")
    plt.plot([inferred_cond_s1s2_mode[0], inferred_cond_s1s2_mode[0]], [0, 1.1 * np.max(values)], "red", alpha=1,
                          label="Posterior mode")
    plt.xlim([xmin, xmax])
    plt.ylim([0, 1.1 * np.max(values)])
    plt.xlabel(r'$s_1$')
    plt.savefig(result_file_name + ".png")

