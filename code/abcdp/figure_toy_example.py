# """ once everything runs, we look at the results """

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as stats
import auxiliary_files.kernel as kernel
import auxiliary_files.feature as feature
import auxiliary_files.util as util
import auxiliary_files.abcbase as ab
import sys

Results_PATH = "/".join([os.getenv("HOME"), "laplace_resampling/"])
Results_PATH_no = "/".join([os.getenv("HOME"), "laplace_no_resampling/"])
Plot_PATH = "/".join([os.getenv("HOME"), "Desktop/resultados_toy/"])
# Create the folder for containing the figures.
os.makedirs(Results_PATH, exist_ok=True)

""" (1) bringing results here """

howmanyruns = 10
seednummat = np.arange(0,howmanyruns) # seednum = int(sys.argv[1])


# percentile_mat =[5, 10, 20] # int(sys.argv[4])
# epsilon_abc_mat = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] # epsilon_abc = sys.argv[5]
epsilon_abc_mat = [0.05, 0.1, 0.2, 0.5] # epsilon_abc = sys.argv[5]

epsilon_iter_mat = [ 0.5, 1.0, 10.0, 1e6] # epsilon_total = sys.argv[6]
c_stop_mat = [10, 100, 1000]
n_samples = 10000
n = 5000

err_mat_non_priv = np.zeros([howmanyruns, len(c_stop_mat), len(epsilon_abc_mat)])
err_mat_priv = np.zeros([len(seednummat),len(c_stop_mat), len(epsilon_abc_mat),  len(epsilon_iter_mat) ])
err_mat_priv_2 = np.zeros([len(seednummat),len(c_stop_mat), len(epsilon_abc_mat),  len(epsilon_iter_mat) ]) #No resampling threshold case
s_count=0

for s in seednummat:
    c_count=0

    for c in c_stop_mat:
        e_abc_count=0

        for e_abc in epsilon_abc_mat:

            e_total_count = 0

            for e_tot in epsilon_iter_mat:

#                print(s,n, n_samples,c, e_abc, e_tot)
                priv_method = os.path.join(Results_PATH, 'err_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (s, n, n_samples,c, e_abc, e_tot))
                print(priv_method)
                priv_err = np.load(priv_method + '.npy')
                print(priv_err)
                err_mat_priv[s_count, c_count, e_abc_count, e_total_count] = priv_err
                
                #No resampling threshold case
                priv_method_no = os.path.join(Results_PATH_no, 'err_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (s, n, n_samples,c, e_abc, e_tot))
                print(priv_method_no)
                priv_err_no = np.load(priv_method_no + '.npy')
                print(priv_err_no)
                err_mat_priv_2[s_count, c_count, e_abc_count, e_total_count] = priv_err_no
                
                e_total_count +=1

            e_abc_count+=1

        c_count+=1

    s_count+=1




true_param = np.array([0.25, 0.04, 0.33, 0.04, 0.34])
howmanyruns = 60
seednummat = np.arange(0,howmanyruns) # seednum = int(sys.argv[1])

def mean_abs_error(a, b):
    return np.mean(np.abs(a-b))

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html

# prior
# prior
prior = ab.Prior.from_scipy(
    stats.dirichlet(alpha=np.ones(5))
)

list_total_err=[]

for j in seednummat:
    list_params=[]

    for i in range(n_samples):
        param_i = prior.sample(n=1, seed=j + 286 + i)
        list_params.append(param_i)
    params = np.concatenate(list_params, axis=0)
    print(params.shape)
    posterior_mean = np.mean(params, 0)
    print(posterior_mean)
    total_err = mean_abs_error(true_param, posterior_mean)
    list_total_err.append(total_err)

""" (2) plotting the results """

# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 12
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(17,5))

range_min = 0
range_max = 4

plt.subplot(131)
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,0,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv[:,0,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='--', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,0,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv[:,0,range_min:range_max,1]),0),  label='eps_total = 1.0', linestyle='--', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,0,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv[:,0,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='--', color='grey', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,0,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,0,range_min:range_max,3]),0),  label='eps_total = 10.0', linestyle='--', color='blue', marker='o')

#No resampling threshold
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='dotted', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,1]),0),  label='eps_total = 1.0', linestyle='dotted', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='dotted', color='grey', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv_2[:,0,range_min:range_max,3]),0),  label='eps_total = 10.0', linestyle='dotted', color='olive', marker='o')

plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,0,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,0,range_min:range_max,3]),0), label='nonpriv', linestyle='-', color='black', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.full(4, np.mean(np.array(list_total_err))), np.full(4, np.std(np.array(list_total_err))), label='prior', linestyle='-', color='green', marker='o')
plt.title("C_stop=10")
plt.xlabel('epsilon_abc')
plt.ylabel('MSE')
ylim_bottom = 0.00
ylim_top = 0.15
plt.ylim(ylim_bottom,ylim_top)
plt.legend(loc='lower right',prop={'size': 9})
#plt.show()

plt.subplot(132)
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='--', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,1]),0),  label='eps_total = 1.0.', linestyle='--', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='--', color='grey', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,3]),0),  label='eps_total = 10.0', linestyle='--', color='blue', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,4]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,4]),0),  label='eps_total = 100.0', linestyle='--', color='purple', marker='o')

#No resampling threshold
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='dotted', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,1]),0),  label='eps_total = 1.0', linestyle='dotted', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='dotted', color='grey', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv_2[:,1,range_min:range_max,3]),0),  label='eps_total = 10.0', linestyle='dotted', color='olive', marker='o')


plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,1,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,1,range_min:range_max,3]),0), label='nonpriv', linestyle='-', color='black', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.full(4, np.mean(np.array(list_total_err))), np.full(4, np.std(np.array(list_total_err))), label='prior', linestyle='-', color='green', marker='o')
plt.title("C_stop=100")
plt.xlabel('epsilon_abc')
plt.ylabel('MSE')
ylim_bottom = 0.00
ylim_top = 0.15
plt.ylim(ylim_bottom,ylim_top)
plt.legend(loc='lower right',prop={'size': 9})
#plt.show()


plt.subplot(133)
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='--', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,1]),0),  label='eps_total = 1.0', linestyle='--', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='--', color='grey', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,3]),0),  label='eps_total = 10.0', linestyle='--', color='blue', marker='o')
#plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,4]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,4]),0),  label='eps_total = 100.0', linestyle='--', color='purple', marker='o')

#No resampling threshold
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,0]),0),  label='eps_total = 0.5', linestyle='dotted', color='pink', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,1]),0),  label='eps_total = 1.0', linestyle='dotted', color='red', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv_2[:,2,range_min:range_max,2]),0),  label='eps_total = 10.0', linestyle='dotted', color='grey', marker='o')


plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,2,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,2,range_min:range_max,3]),0), label='nonpriv', linestyle='-', color='black', marker='o')
plt.errorbar(epsilon_abc_mat[range_min:range_max], np.full(4, np.mean(np.array(list_total_err))), np.full(4, np.std(np.array(list_total_err))), label='prior', linestyle='-', color='green', marker='o')
plt.title("C_stop=1000")
plt.xlabel('epsilon_abc')
plt.ylabel('MSE')
ylim_bottom = 0.00
ylim_top = 0.15
plt.ylim(ylim_bottom,ylim_top)
plt.legend(loc='lower right' ,prop={'size': 9})
plt.savefig(Plot_PATH + 'toy_results.pdf', dpi=200)

