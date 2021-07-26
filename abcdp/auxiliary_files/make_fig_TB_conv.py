# """ once everything runs, we look at the results """

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

Results_PATH = "/".join([os.getenv("HOME"), "TB_results/"])
#<<<<<<< Updated upstream
# Results_PATH = "/".join([os.getenv("HOME"), "TBstuf/"])

def mean_abs_error(a, b):
    return np.mean(np.abs(a-b))

""" (1) bringing results here """

# ss_clipping_norm_bound = 5
#=======

""" (1) bringing results here """

#>>>>>>> Stashed changes

howmanyruns = 10
seednummat = np.arange(0,howmanyruns)

#<<<<<<< Updated upstream
# epsilon_abc_mat =[20,40,80]

epsilon_abc_mat =[5,10,20,50,100, 200, 400, 800]
# epsilon_abc_mat =[5,10,20, 50,100,200,400,800]
#=======
epsilon_abc_mat =[20,40,80]
#>>>>>>> Stashed changes

epsilon_total_mat = [1, 2, 4, 8]
delta_total = 1e-4

#<<<<<<< Updated upstream
ss_clipping_norm_bound = 200
#=======
#>>>>>>> Stashed changes

err_mat_non_priv = np.zeros([howmanyruns,len(epsilon_abc_mat)])
err_mat_priv = np.zeros([howmanyruns,len(epsilon_abc_mat), len(epsilon_total_mat)])
# print(np.shape(err_mat_priv))

#<<<<<<< Updated upstream
true_param = [5.88, 6.74, 0.09, 192]

#=======
#>>>>>>> Stashed changes

for seednum in seednummat:

    eps_abc_count = 0

    for epsilon_abc in epsilon_abc_mat:

        non_priv_method = os.path.join(Results_PATH, 'TB_seed=%s_epsilonabc=%s' % (seednum, epsilon_abc))
#<<<<<<< Updated upstream

        posterior_mean_non_priv = np.load(non_priv_method + '.npy')
        non_priv_err = mean_abs_error(true_param, posterior_mean_non_priv)
        non_priv_err = np.load(non_priv_method + '.npy')
#=======
#>>>>>>> Stashed changes

        err_mat_non_priv[seednum,eps_abc_count] = non_priv_err


        eps_tot_count = 0

        for epsilon_total in epsilon_total_mat:

#<<<<<<< Updated upstream
            # priv_method = os.path.join(Results_PATH, 'TB_seed=%s_epsilonabc=%s_epsilontotal=%s_deltatotal=%s' % (seednum, epsilon_abc, epsilon_total, delta_total))
            priv_method = os.path.join(Results_PATH, 'TB_seed=%s_epsilonabc=%s_epsilontotal=%s_deltatotal=%s_clinrm=%s' % (seednum, epsilon_abc, epsilon_total, delta_total, ss_clipping_norm_bound))

            priv_method = os.path.join(Results_PATH, 'TB_seed=%s_epsilonabc=%s_epsilontotal=%s_deltatotal=%s' % (seednum, epsilon_abc, epsilon_total, delta_total))
#=======
#>>>>>>> Stashed changes
            priv_err = np.load(priv_method+'.npy')
            err_mat_priv[seednum, eps_abc_count, eps_tot_count] = priv_err
            eps_tot_count = eps_tot_count + 1

        eps_abc_count = eps_abc_count + 1


#<<<<<<< Updated upstream

# err_mat = np.zeros(len(seednummat))
# for seednum in seednummat:
#     priv_method = os.path.join(Results_PATH, 'TB_seed=%s_epsilonabc=%s_epsilontotal=%s_deltatotal=%s' % (
#     seednum, 80, 4, 1e-4))
#     priv_err = np.load(priv_method + '.npy')
#     err_mat[seednum] = priv_err
#
# np.mean(np.squeeze(err_mat_priv[:,:,2]),0)

""" (2) plotting the results """


# np.mean(err_mat_non_priv,0)
# array([2.79072292, 3.04003854, 3.65678942])


#=======
""" (2) plotting the results """

#>>>>>>> Stashed changes
# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 18
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
#<<<<<<< Updated upstream


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# # font options
# font = {
#     #'family' : 'normal',
#     #'weight' : 'bold',
#     'size'   : 18
# }
#
# plt.rc('font', **font)
# plt.rc('lines', linewidth=2)
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42


# plt.errorbar(epsilon_abc_mat, np.mean(err_mat_non_priv,0), np.std(err_mat_non_priv,0),  label='non_priv', linestyle='-', color='black')


# Convergence plot: MSE vs Epsilon_DP

# plt.figure(1)
#
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,0]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,0]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 5', linestyle='--', color='black')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,1]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,1]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 10', linestyle='--', color='red')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,2]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,2]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 20', linestyle='--', color='blue')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,3]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,3]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 50', linestyle='--', color='magenta')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,4]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,4]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 100', linestyle='--', color='grey')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,5]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,5]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 200', linestyle='--', color='purple')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,6]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,6]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 400', linestyle='--', color='green')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_priv[:,7]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,7]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 800', linestyle='-', color='black')
# # # # plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,1],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,1],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.005', linestyle='-', color='black')
# # # # plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.01', linestyle='-', color='red')
# # # # plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.05', linestyle='-', color='blue')
# # # # plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.1', linestyle='-', color='grey')
# #
# plt.yscale('log')
# plt.xlabel('epsilon_total')
# plt.ylabel('MSE')
# # plt.ylim(1e-3*5,0.15)
# plt.legend(loc='lower left')
# plt.show()
# #
# # plt.figure(1)
#
#=======
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# # Convergence plot: MSE vs Epsilon_DP
# plt.figure(1)
#
#>>>>>>> Stashed changes
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_non_priv[:,0]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,0]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 20', linestyle='--', color='black')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_non_priv[:,1]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,1]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 40', linestyle='--', color='red')
# plt.errorbar(epsilon_total_mat, np.mean(np.squeeze(err_mat_non_priv[:,2]),0)*np.ones(len(epsilon_total_mat)), np.std(np.squeeze(err_mat_priv[:,2]),0)*np.ones(len(epsilon_total_mat)),  label='priv eps_abc = 80', linestyle='--', color='blue')
#
#
# plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,1],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,1],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.005', linestyle='-', color='black')
# plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.01', linestyle='-', color='red')
# plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.05', linestyle='-', color='blue')
# plt.errorbar(epsilon_total_mat, np.mean(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.1', linestyle='-', color='grey')
#
# plt.yscale('log')
# plt.xlabel('epsilon_total')
# plt.ylabel('MSE')
# # plt.ylim(1e-3*5,0.15)
# plt.legend(loc='center left')
# plt.show()


#<<<<<<< Updated upstream
# Interplay between epsilon_dp and epsilon_abc: MSE vs Epsilon_ABC
plt.figure(2)

plt.errorbar(epsilon_abc_mat, np.mean(err_mat_non_priv,0), np.std(err_mat_non_priv,0),  label='non_priv', linestyle='-', color='black')
plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,0]),0), np.std(np.squeeze(err_mat_priv[:,:,0]),0),  label='eps_total = 1', linestyle='-', color='red')
plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,1]),0), np.std(np.squeeze(err_mat_priv[:,:,1]),0),  label='eps_total = 2', linestyle='-', color='blue')
plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,2]),0), np.std(np.squeeze(err_mat_priv[:,:,2]),0),  label='eps_total = 4', linestyle='-', color='grey')
# plt.plot([1,2,4,8], )
# plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,3]),0), np.std(np.squeeze(err_mat_priv[:,:,3]),0),  label='eps_total = 8', linestyle='--', color='magenta')

# plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,2]),0), np.std(np.squeeze(err_mat_priv[:,:,2]),0),  label='eps_total = 4', linestyle='--', color='grey')
plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,3]),0), np.std(np.squeeze(err_mat_priv[:,:,3]),0),  label='eps_total = 8', linestyle='-', color='purple')


plt.yscale('log')
plt.xscale('log')
plt.xlabel('epsilon_abc')
plt.ylabel('MSE')
# plt.yticks([epsilon_abc_mat[1], epsilon_abc_mat[2], epsilon_abc_mat[3], epsilon_abc_mat[4]])
# plt.ylim(1e-3*5,0.12)
plt.legend(loc='lower right')
plt.show()

#=======
# # Interplay between epsilon_dp and epsilon_abc: MSE vs Epsilon_ABC
# plt.figure(2)
#
# plt.errorbar(epsilon_abc_mat, np.mean(err_mat_non_priv,0), np.std(err_mat_non_priv,0),  label='non_priv', linestyle='-', color='black')
# plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,0]),0), np.std(np.squeeze(err_mat_priv[:,:,0]),0),  label='eps_total = 1', linestyle='--', color='red')
# plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,1]),0), np.std(np.squeeze(err_mat_priv[:,:,1]),0),  label='eps_total = 2', linestyle='--', color='blue')
# # plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,2]),0), np.std(np.squeeze(err_mat_priv[:,:,2]),0),  label='eps_total = 4', linestyle='--', color='grey')
# plt.errorbar(epsilon_abc_mat, np.mean(np.squeeze(err_mat_priv[:,:,3]),0), np.std(np.squeeze(err_mat_priv[:,:,3]),0),  label='eps_total = 8', linestyle='--', color='grey')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('epsilon_abc')
# plt.ylabel('MSE')
# # plt.yticks([epsilon_abc_mat[1], epsilon_abc_mat[2], epsilon_abc_mat[3], epsilon_abc_mat[4]])
# # plt.ylim(1e-3*5,0.12)
# plt.legend(loc='bottom right')
# plt.show()
#
#>>>>>>> Stashed changes

# plt.figure(3)
#
# range_min =0
# range_max = 5
#
# # plt.subplot(211)
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,range_min:range_max,0]),0), np.std(np.squeeze(err_mat_priv[:,range_min:range_max,0]),0),  label='eps_total = 1', linestyle='--', color='pink', marker='o')
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,range_min:range_max,1]),0), np.std(np.squeeze(err_mat_priv[:,range_min:range_max,1]),0),  label='eps_total = 2', linestyle='--', color='red', marker='o')
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,range_min:range_max,2]),0), np.std(np.squeeze(err_mat_priv[:,range_min:range_max,2]),0),  label='eps_total = 4', linestyle='--', color='grey', marker='o')
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,range_min:range_max,3]),0), np.std(np.squeeze(err_mat_priv[:,range_min:range_max,3]),0),  label='eps_total = 6', linestyle='--', color='blue', marker='o')
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(np.squeeze(err_mat_priv[:,range_min:range_max,4]),0), np.std(np.squeeze(err_mat_priv[:,range_min:range_max,4]),0),  label='eps_total = 8', linestyle='--', color='purple', marker='o')
# #
# # plt.yscale('log')
# # plt.xscale('log')
# # # plt.xlabel('epsilon_abc')
# # plt.ylabel('MSE')
# # # plt.yticks([epsilon_abc_mat[1], epsilon_abc_mat[2], epsilon_abc_mat[3], epsilon_abc_mat[4]])
# # plt.ylim(1e-3*5,0.12)
# # plt.legend(loc='bottom right')
#
# # plt.subplot(212)
# plt.errorbar(epsilon_abc_mat[range_min:range_max], np.mean(err_mat_non_priv[:,range_min:range_max],0), np.std(err_mat_non_priv[:,range_min:range_max],0), label='nonpriv', linestyle='-', color='black', marker='o')
# # plt.errorbar(epsilon_abc_mat[1:5], np.mean(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,2],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.01', linestyle='-', color='red')
# # plt.errorbar(epsilon_abc_mat[1:5], np.mean(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,3],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.05', linestyle='-', color='blue')
# # plt.errorbar(epsilon_abc_mat[1:5], np.mean(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), np.std(err_mat_non_priv[:,4],0)*np.ones(len(epsilon_total_mat)), label='nonpriv eps_abc = 0.1', linestyle='-', color='grey')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('epsilon_abc')
# plt.ylabel('MSE')
# # plt.yticks(np.arange(5), [0.005, 0.01, 0.05, 0.1])
# plt.ylim(1e-3*5,0.12)
# plt.legend(loc='lower right')
# plt.show()
