# to plot the probability of flipping between ABC output and ABCDP output

import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 18
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def prob_flip(rho, eps_abc, b):

    if rho-eps_abc>0:
        tmp = - (rho-eps_abc)/b
    else:
        tmp =  (rho-eps_abc)/b

    prob = 1/6*(4*np.exp(tmp/2) - np.exp(tmp))

    return prob

def b_func(c, sensitivity, eps_tot):
    return 2*c*sensitivity/eps_tot



#####################################################################

n = 10
sensitivity = 1/n


rho_samps = np.random.uniform(0,1,size=100)
eps_abc = 0.2

eps_tot_mat = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
c_mat = [10, 100, 1000]
prob = np.zeros((len(rho_samps), len(eps_tot_mat), len(c_mat)))

for i in np.arange(0,len(rho_samps)):
    rho = rho_samps[i]
    for j in np.arange(0, len(eps_tot_mat)):
        eps_tot = eps_tot_mat[j]
        for k in np.arange(0, len(c_mat)):
            c = c_mat[k]

            b = b_func(c,sensitivity,eps_tot)
            prob[i,j,k] = prob_flip(rho, eps_abc, b)


plt.figure(1)
plt.subplot(311)
plt.title('flipping probability (c=10)')
plt.boxplot([prob[:,0,0], prob[:,1,0], prob[:,2,0], prob[:,3,0], prob[:,4,0], prob[:,5,0], prob[:,6,0], prob[:,7,0]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(312)
plt.title('(c=100)')
plt.boxplot([prob[:,0,1], prob[:,1,1], prob[:,2,1], prob[:,3,1], prob[:,4,1], prob[:,5,1], prob[:,6,1], prob[:,7,1]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(313)
plt.title('(c=1000)')
plt.boxplot([prob[:,0,2], prob[:,1,2], prob[:,2,2], prob[:,3,2], prob[:,4,2], prob[:,5,2], prob[:,6,2], prob[:,7,2]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
plt.xlabel('total epsilons')
# plt.ylim(0, 0.6)


#####################################################################


n = 100
sensitivity = 1/n
prob = np.zeros((len(rho_samps), len(eps_tot_mat), len(c_mat)))

for i in np.arange(0,len(rho_samps)):
    rho = rho_samps[i]
    for j in np.arange(0, len(eps_tot_mat)):
        eps_tot = eps_tot_mat[j]
        for k in np.arange(0, len(c_mat)):
            c = c_mat[k]

            b = b_func(c,sensitivity,eps_tot)
            prob[i,j,k] = prob_flip(rho, eps_abc, b)


plt.figure(2)
plt.subplot(311)
plt.title('flipping probability (c=10)')
plt.boxplot([prob[:,0,0], prob[:,1,0], prob[:,2,0], prob[:,3,0], prob[:,4,0], prob[:,5,0], prob[:,6,0], prob[:,7,0]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(312)
plt.title('(c=100)')
plt.boxplot([prob[:,0,1], prob[:,1,1], prob[:,2,1], prob[:,3,1], prob[:,4,1], prob[:,5,1], prob[:,6,1], prob[:,7,1]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(313)
plt.title('(c=1000)')
plt.boxplot([prob[:,0,2], prob[:,1,2], prob[:,2,2], prob[:,3,2], prob[:,4,2], prob[:,5,2], prob[:,6,2], prob[:,7,2]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
plt.xlabel('total epsilons')
# plt.ylim(0, 0.6)


#####################################################################

n = 1000
sensitivity = 1/n

prob = np.zeros((len(rho_samps), len(eps_tot_mat), len(c_mat)))

for i in np.arange(0,len(rho_samps)):
    rho = rho_samps[i]
    for j in np.arange(0, len(eps_tot_mat)):
        eps_tot = eps_tot_mat[j]
        for k in np.arange(0, len(c_mat)):
            c = c_mat[k]

            b = b_func(c,sensitivity,eps_tot)
            prob[i,j,k] = prob_flip(rho, eps_abc, b)


plt.figure(3)
plt.subplot(311)
plt.title('flipping probability (c=10)')
plt.boxplot([prob[:,0,0], prob[:,1,0], prob[:,2,0], prob[:,3,0], prob[:,4,0], prob[:,5,0], prob[:,6,0], prob[:,7,0]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(312)
plt.title('(c=100)')
plt.boxplot([prob[:,0,1], prob[:,1,1], prob[:,2,1], prob[:,3,1], prob[:,4,1], prob[:,5,1], prob[:,6,1], prob[:,7,1]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
# plt.ylim(0, 0.6)

plt.subplot(313)
plt.title('(c=1000)')
plt.boxplot([prob[:,0,2], prob[:,1,2], prob[:,2,2], prob[:,3,2], prob[:,4,2], prob[:,5,2], prob[:,6,2], prob[:,7,2]])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], eps_tot_mat)
plt.xlabel('total epsilons')
# plt.ylim(0, 0.6)

plt.show()