# this is for making histogram from the posterior distribution

import auxiliary_files.disease_simulator as si
import auxiliary_files.operations as ops

import numpy as np
import matplotlib.pyplot as plt
import operator
from autodp import privacy_calibrator
import sys
import os
import random

# define a simulator and prior
def simulator_func(param, t_obs, cluster_size_bound, warmup_bounds, seed=3):
    # unpack param
    burden = param[0]
    a0 = param[1]
    d0 = param[2]
    a1 = param[3]
    d1 = param[4]

    sam = ops.simulator(burden, a0, d0, a1, d1, t_obs, cluster_size_bound, warmup_bounds=warmup_bounds, batch_size=1,
              random_state=None, dtype=np.int16)
    return sam

def prior(mean_obs_bounds, t1_bound, a1_bound):
    burden = 200 + np.sqrt(30)*np.random.randn(1)
    sam = ops.JointPrior.rvs(burden, mean_obs_bounds=mean_obs_bounds, t1_bound=t1_bound, a1_bound=a1_bound)
    # sam = np.array((R0, R1, t1, m_obs)).T
    return burden, sam

def mean_abs_error(a, b):
    return np.mean(np.abs(a-b))


if __name__ == '__main__':

    seednum = 0
    is_priv = 1
#<<<<<<< Updated upstream
    epsilon_abc = 50
    epsilon_total = 8
    delta_total = 1e-4

    ss_clipping_norm_bound = 200

    # if epsilon_abc == 20:
    #     ss_clipping_norm_bound = 20
    # elif epsilon_abc == 40:
    #     ss_clipping_norm_bound = 10
    # else:
    #     ss_clipping_norm_bound = 5
#=======
    epsilon_abc = 20
    epsilon_total = 4
    delta_total = 1e-4
    if epsilon_abc == 20:
        ss_clipping_norm_bound = 20
    elif epsilon_abc == 40:
        ss_clipping_norm_bound = 10
    else:
        ss_clipping_norm_bound = 5
#>>>>>>> Stashed changes


    random.seed(seednum)

    cluster_size_bound = 80
    observed = ops.get_SF_data(cluster_size_bound)

    true_param = [5.88, 6.74, 0.09, 192] # [R1, t1, R2, burden] from Table 1 in Lintusaari et al paper.
    # R1_true = true_param[0]
    # t1_true = true_param[1]
    # R2_true = true_param[2]
    # burden_true = true_param[3]
    #
    mean_obs_bounds = (0, 350)
    t1_bound = 30
    a1_bound = 40
    d2 = 5.95
    warmup_bounds = (15, 300)
    # Observation pediod in years
    t_obs = 2

    # a2_true = operator.mul(R2_true, d2)
    # a1_true = ops.Rt_to_a(R1_true, t1_true)
    # d1_true = ops.Rt_to_d(R1_true, t1_true)
    #
    # param_input = [burden_true, a2_true, d2, a1_true, d1_true]

    # observed = simulator_func(param_input, t_obs, cluster_size_bound, warmup_bounds)

    observed_ss_0 = ops.pick(observed, 'n_obs')
    observed_ss_1 = ops.pick(observed, 'n_clusters')
    observed_ss_2 = ops.pick(observed, 'largest')
    observed_ss_3 = ops.pick(observed, 'clusters')
    observed_ss_4 = ops.pick(observed, 'obs_times')

    observed_ss = [observed_ss_0, observed_ss_1, observed_ss_2, observed_ss_3, observed_ss_4]

    """ ABC """

    howmanysteps = 200
    dist_mat = np.zeros([howmanysteps])
    howmanyparams = 4
    param_mat = np.zeros([howmanysteps,howmanyparams])

    if is_priv == 1:
        # compute privacy parameter
        privacy_params = privacy_calibrator.gaussian_mech(epsilon_total, delta_total, prob=1, k=howmanysteps)
        print('eps,delta,gamma = ({eps},{delta},{gamma}) ==> privacy parameter =', privacy_params['sigma'])
        obtained_privacy_param = privacy_params['sigma']
        sigma_soft = obtained_privacy_param*ss_clipping_norm_bound

    for step_count in np.arange(0,howmanysteps):

        # print('step count is ', step_count, 'out of ', howmanysteps)

        """ data simulation """

        burden, other_prior_sam = prior(mean_obs_bounds, t1_bound, a1_bound)

        R2 = other_prior_sam[0]
        R1 = other_prior_sam[1]
        t1 = other_prior_sam[2]

        a2 = operator.mul(R2, d2)
        a1 = ops.Rt_to_a(R1, t1)
        d1 = ops.Rt_to_d(R1, t1)

        param = [burden, a2, d2, a1, d1]
        param_mat[step_count,:] = [R1, t1, R2, burden]

        pseudo_data = simulator_func(param, t_obs, cluster_size_bound, warmup_bounds)

        # Summaries extracted from the simulator output
        clusters = ops.pick(pseudo_data, 'clusters')
        n_obs = ops.pick(pseudo_data, 'n_obs')
        n_clusters = ops.pick(pseudo_data, 'n_clusters')
        largest = ops.pick(pseudo_data, 'largest')
        obs_times = ops.pick(pseudo_data, 'obs_times')


        # Distance
        dist = ops.distance(n_obs, n_clusters, largest, clusters, obs_times, observed=observed_ss)
        dist_mat[step_count] = dist

        if is_priv==1:
            dist_clipped = min(dist/epsilon_abc, ss_clipping_norm_bound)

            noisy_dis = dist_clipped + sigma_soft*np.random.randn(1)
            dist_mat[step_count] = max(noisy_dis,0)

    if is_priv == 0:
        unnorm_W = np.exp(-dist_mat / epsilon_abc)
    else: # private
        unnorm_W = np.exp(-dist_mat)

    normed_W = unnorm_W / np.sum(unnorm_W)

#<<<<<<< Updated upstream
    posterior_mean = np.sum(param_mat * normed_W[:, None], 0)
    err_soft_abc = mean_abs_error(true_param, posterior_mean)
    print('true params are ', true_param)
    print('posterior_mean is', posterior_mean)
    print('err is ', err_soft_abc)
#=======
#>>>>>>> Stashed changes
    # posterior_mean = np.sum(param_mat * normed_W[:, None], 0)
    # err_soft_abc = mean_abs_error(true_param, posterior_mean)
    # print('true params are ', true_param)
    # print('posterior_mean is', posterior_mean)
    # print('err is ', err_soft_abc)


    """ plotting starts here """

    # plt.figure(2)
    # plt.subplot(141)
    # plt.title('burden rate')
    # plt.hist(param_mat[:,3], label='burden rate')
    # plt.xlim(180,220)
    # plt.subplot(142)
    # plt.title('R1')
    # plt.hist(param_mat[:,0], label='R1')
    # plt.xlim(0, 12)
    # plt.subplot(143)
    # plt.title('R2')
    # plt.hist(param_mat[:,2], label='R2')
    # plt.xlim(0, 0.6)
    # plt.subplot(144)
    # plt.title('t1')
    # plt.hist(param_mat[:,1], label='t1')
    # plt.xlim(0, 31)
    # plt.show()
    #


    """ resample from the posterior """
    # resample from the empirical posterior
    n_resample = howmanysteps
    I = np.random.choice(len(normed_W), n_resample, replace=True, p=normed_W)


    plt.figure(1)
    plt.title('posterior sample (non_priv)')
    plt.subplot(141)
    plt.title('burden rate')
    plt.hist(param_mat[I,3], label='burden rate')
    plt.xlim(180, 220)
    plt.subplot(142)
    plt.title('R1')
    plt.hist(param_mat[I,0], label='R1')
    plt.xlim(0, 12)
    plt.subplot(143)
    plt.title('R2')
    plt.hist(param_mat[I,2], label='R2')
    plt.xlim(0, 0.6)
    plt.subplot(144)
    plt.title('t1')
    plt.hist(param_mat[I,1], label='t1')
    plt.xlim(0, 31)
    plt.show()



    # np.mean(dist_mat)
    # np.var(dist_mat)

