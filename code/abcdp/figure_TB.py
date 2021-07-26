import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import auxiliary_files.operations as ops
import operator

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

Results_PATH = "/".join([os.getenv("HOME"), "TB_results/"])

def mean_abs_error(a, b):
    return np.mean(np.abs(a - b))

def prior(mean_obs_bounds, t1_bound, a1_bound, d2):

    burden = 200 + np.sqrt(30)*np.random.randn(1)
    other_prior_sam = ops.JointPrior.rvs(burden, mean_obs_bounds=mean_obs_bounds, t1_bound=t1_bound, a1_bound=a1_bound)

    R2 = other_prior_sam[0]
    R1 = other_prior_sam[1]
    t1 = other_prior_sam[2]

    a2 = operator.mul(R2, d2)
    a1 = ops.Rt_to_a(R1, t1)
    d1 = ops.Rt_to_d(R1, t1)

    param = [burden, [a2], [d2], [a1], [d1]]

    others = [R1, t1, R2, burden]

    return param, others


true_params = [5.88, 6.74, 0.09, 192] # [R1, t1, R2, burden]


""" plot the histogram of the prior samples """
# others = [R1, t1, R2, burden]
mean_obs_bounds = (0, 350)
t1_bound = 30
a1_bound = 40
d2 = 5.95

list_params = []
for i in range(50):
    param_i, others_i = prior(mean_obs_bounds, t1_bound, a1_bound, d2)
    others_i = np.expand_dims(np.array(others_i, dtype='float64'), axis=0)
    list_params.append(others_i)

chosen_posterior_samps = np.concatenate(list_params, axis=0)

plt.figure(1)
plt.subplot(3,4,2)
plt.hist(chosen_posterior_samps[:, 0], label='R1', bins=10)
plt.axvline(x=true_params[0], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,0]), color='b', linestyle='--')
plt.xlim(0,12)

plt.subplot(3,4,4)
plt.hist(chosen_posterior_samps[:, 1], label='t1', bins=10)
plt.axvline(x=true_params[1], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,1]), color='b', linestyle='--')
plt.xlim(0, 31)

plt.subplot(3,4,3)
plt.hist(chosen_posterior_samps[:, 2], label='R2', bins=10)
plt.axvline(x=true_params[2], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,2]), color='b', linestyle='--')
plt.xlim(0.0,0.6)

plt.subplot(3,4,1)
plt.hist(chosen_posterior_samps[:, 3], label='burden rate', bins=10)
plt.axvline(x=true_params[3], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,3]), color='b', linestyle='--')
plt.xlim(180,220)


###################################################

epsilon_total = 1. # for non-private, input epsilon_total = 1e6, then we set sigma_rej = 0.
n = 100

plot_name = os.path.join(Results_PATH, 'n=%s_epstot=%s' % (n, epsilon_total))
chosen_posterior_samps = np.load(plot_name + 'poster_samps.npy')
posterior_mean = np.mean(chosen_posterior_samps,0)

print(plot_name)
print('posterior mean: ', posterior_mean)
err = mean_abs_error(true_params, posterior_mean)
print('err:', err)

plt.subplot(3,4,6)
plt.hist(chosen_posterior_samps[:, 0], label='R1', bins=10)
plt.axvline(x=true_params[0], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,0]), color='b', linestyle='--')
plt.xlim(0,12)

plt.subplot(3,4,8)
plt.hist(chosen_posterior_samps[:, 1], label='t1', bins=10)
plt.axvline(x=true_params[1], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,1]), color='b', linestyle='--')
plt.xlim(0, 31)

plt.subplot(3,4,7)
plt.hist(chosen_posterior_samps[:, 2], label='R2', bins=10)
plt.axvline(x=true_params[2], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,2]), color='b', linestyle='--')
plt.xlim(0.0,0.6)

plt.subplot(3,4,5)
plt.hist(chosen_posterior_samps[:, 3], label='burden rate', bins=10)
plt.axvline(x=true_params[3], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,3]), color='b', linestyle='--')
plt.xlim(180,220)

################################################################################

epsilon_total = 1. # for non-private, input epsilon_total = 1e6, then we set sigma_rej = 0.
n = 1000

plot_name = os.path.join(Results_PATH, 'n=%s_epstot=%s' % (n, epsilon_total))
chosen_posterior_samps = np.load(plot_name + 'poster_samps.npy')
posterior_mean = np.mean(chosen_posterior_samps,0)

print(plot_name)
print('posterior mean: ', posterior_mean)
err = mean_abs_error(true_params, posterior_mean)
print('err:', err)

plt.subplot(3,4,10)
plt.hist(chosen_posterior_samps[:, 0], label='R1', bins=10)
plt.axvline(x=true_params[0], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,0]), color='b', linestyle='--')
plt.xlim(0,12)

plt.subplot(3,4,12)
plt.hist(chosen_posterior_samps[:, 1], label='t1', bins=10)
plt.axvline(x=true_params[1], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,1]), color='b', linestyle='--')
plt.xlim(0, 31)

plt.subplot(3,4,11)
plt.hist(chosen_posterior_samps[:, 2], label='R2', bins=10)
plt.axvline(x=true_params[2], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,2]), color='b', linestyle='--')
plt.xlim(0.0,0.6)

plt.subplot(3,4,9)
plt.hist(chosen_posterior_samps[:, 3], label='burden rate', bins=10)
plt.axvline(x=true_params[3], color='k', linestyle='-')
plt.axvline(x=np.mean(chosen_posterior_samps[:,3]), color='b', linestyle='--')
plt.xlim(180,220)

plt.show()















plt.show()


