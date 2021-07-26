# this is a wrapper function for running Figure1.py with different variables


import sys
import subprocess
import numpy as np
import os


howmanyruns = 60
seednummat = np.arange(0,howmanyruns) # seednum = int(sys.argv[1])


epsilon_abc_mat = [0.05, 0.1, 0.2, 0.5] # epsilon_abc = sys.argv[5]  , 
epsilon_iter_mat = [0.5, 1.0, 10.0, 1e6] # epsilon_total = sys.argv[6] 0.1, 0.5, 1.0, 10.0, , 1e6
n_samples = 10000
n = 5000
c_stop_mat = [10, 100, 1000] # 100, 1000 



for seednum in seednummat:
    for epsilon_abc in epsilon_abc_mat:
        for e_total in epsilon_iter_mat:
            for c_stop in c_stop_mat:

                print('seednum, n, n_samples, c_stop, eps_abc, eps_total', [seednum, n, n_samples, c_stop, epsilon_abc, e_total])
                sys.argv = ['simulated_example_rej_svt.py', str(seednum), str(n), str(n_samples), str(c_stop), str(epsilon_abc), str(e_total)]
                # execfile("./simulated_example.py")!@#
                exec(open("simulated_example_rej_svt.py").read())






# """ once everything runs, we look at the results """

# Results_PATH = "/".join([os.getenv("HOME"), "Fig1_results/"])
#
# for seednum in seednummat:
#     for epsilon_abc in epsilon_abc_mat:
#         for epsilon_total in epsilon_total_mat:
#
#             print('seednum, epsilon_abc, epsilon_total is', [seednum, epsilon_abc, epsilon_total])
#
#             non_priv_method = os.path.join(Results_PATH, 'seed=%s_n=%s_epsilonabc=%s_gamma=%s' % (seednum, n, epsilon_abc, gamma))
#             non_priv_p = np.load(non_priv_method+'.npy')
#             print(non_priv_p)
#
#             method = os.path.join(Results_PATH, 'seed=%s_n=%s_epsilonabc=%s_epsilontotal=%s_deltatotal=%s_gamma=%s' % (seednum, n, epsilon_abc, epsilon_total, delta_total, gamma))
#             pp = np.load(method + '.npy')
#             print(pp)
