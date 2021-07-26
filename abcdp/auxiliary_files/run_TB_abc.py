# this is a wrapper function for running Figure1.py with different variables


import sys
import subprocess
import numpy as np
import os

howmanyruns = 10
seednummat = np.arange(0,howmanyruns) # seednum = int(sys.argv[1])
print(seednummat)

# is_priv = [0,1] # sys.argv[2]
#<<<<<<< Updated upstream

is_priv = 1
# is_priv = 0

# epsilon_abc_mat =[20,40,80] # sys.argv[3]
epsilon_abc_mat =[800] # sys.argv[3]
# epsilon_abc_mat =[50, 100, 200, 400, 800] # sys.argv[3]
#=======
is_priv = 0
epsilon_abc_mat =[20,40,80] # sys.argv[3]
#>>>>>>> Stashed changes
# epsilon_abc, ss_clipping_norm_bound = (20,10)
# epsilon_abc, ss_clipping_norm_bound = (40, 5)
# epsilon_abc, ss_clipping_norm_bound = (80,2)

epsilon_total_mat = [1, 2, 4, 8]# epsilon_total = sys.argv[4]
#<<<<<<< Updated upstream
# epsilon_total_mat = [0.1, 16]
#=======
#>>>>>>> Stashed changes
delta_total = 1e-4 # delta_total = sys.argv[5]

for seednum in seednummat:
    for epsilon_abc in epsilon_abc_mat:
#<<<<<<< Updated upstream

        # epsilon_total = 0
        # delta_total = 0
        # print('seednum, epsilon_abc is', [seednum, epsilon_abc])
        # sys.argv = ['test_TB.py', str(seednum), str(is_priv), str(epsilon_abc), str(epsilon_total), str(delta_total)]
        # # execfile("./simulated_example.py")
        # exec(open("./test_TB.py").read())

        for epsilon_total in epsilon_total_mat:

            print('seednum, epsilon_abc is', [seednum, epsilon_abc])
            sys.argv = ['test_TB.py', str(seednum), str(is_priv), str(epsilon_abc),  str(epsilon_total), str(delta_total)]
            # execfile("./simulated_example.py")
            exec(open("../auxiliary_files/test_TB.py").read())
#=======
        # if epsilon_abc == 20:
        #     clip_norm = 20
        # elif epsilon_abc == 40:
        #     clip_norm = 10
        # else:
        #     clip_norm = 5
        # for epsilon_total in epsilon_total_mat:

        epsilon_total = 0
        delta_total = 0

        print('seednum, epsilon_abc is', [seednum, epsilon_abc])
        sys.argv = ['test_TB.py', str(seednum), str(is_priv), str(epsilon_abc),  str(epsilon_total), str(delta_total)]
        # execfile("./simulated_example.py")
        exec(open("../auxiliary_files/test_TB.py").read())

#>>>>>>> Stashed changes





