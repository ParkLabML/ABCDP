# this is a wrapper function for running Figure1.py with different variables


import sys
import subprocess
import numpy as np
import os

howmanyruns = 10
seednummat = np.arange(0,howmanyruns) # seednum = int(sys.argv[1])
print(seednummat)

# is_priv = [0,1] # sys.argv[2]
is_priv = 0
epsilon_abc_mat =[20,40,80] # sys.argv[3]
# epsilon_abc, ss_clipping_norm_bound = (20,10)
# epsilon_abc, ss_clipping_norm_bound = (40, 5)
# epsilon_abc, ss_clipping_norm_bound = (80,2)

epsilon_total = 0
delta_total = 0

for seednum in seednummat:
    for epsilon_abc in epsilon_abc_mat:
        # if epsilon_abc == 20:
        #     clip_norm = 20
        # elif epsilon_abc == 40:
        #     clip_norm = 10
        # else:
        #     clip_norm = 5
        # for epsilon_total in epsilon_total_mat:

        print('seednum, epsilon_abc is', [seednum, epsilon_abc])
        sys.argv = ['test_TB.py', str(seednum), str(is_priv), str(epsilon_abc),  str(epsilon_total), str(delta_total)]
        # execfile("./simulated_example.py")
        exec(open("../auxiliary_files/test_TB.py").read())






