import operations as ops

import numpy as np
import matplotlib.pyplot as plt
import operator
import sys
import os
import random
import feature as feature
import util as util
import abcbase as ab

# define a simulator and prior
def simulator_func(param):

    # unpack param
    burden = param[0]
    a0 = param[1]
    d0 = param[2]
    a1 = param[3]
    d1 = param[4]

    cluster_size_bound = 80
    warmup_bounds = (15, 300)
    t_obs = 2

    sam = ops.simulator(burden, a0, d0, a1, d1, t_obs, cluster_size_bound, warmup_bounds=warmup_bounds, batch_size=1,
              random_state=None, dtype=np.int16)
    return sam

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

def feature_map(pseudo_data):

    # Summaries extracted from the simulator output
    clusters = ops.pick(pseudo_data, 'clusters')
    n_obs = ops.pick(pseudo_data, 'n_obs') # number of observations
    n_clusters = ops.pick(pseudo_data, 'n_clusters') # number of clusters
    largest = ops.pick(pseudo_data, 'largest') # size of the largest cluster
    obs_times = ops.pick(pseudo_data, 'obs_times') # The number of months in which at least one observation was made \
    # from the largest cluster.

    return n_obs, n_clusters, largest, clusters, obs_times

def mean_abs_error(a, b):
    return np.mean(np.abs(a - b))

Results_PATH = "/".join([os.getenv("HOME"), "TB_results/"])

def main():

    seednum = 1
    c_stop = 10
    n_samples = 1000
    epsilon_rej = 150
    # epsilon_total = 1e6 # for non-private, input epsilon_total = 1e6, then we set sigma_rej = 0.
    epsilon_total = 1.0 # for private, input epsilon_total = 1e6, then we set sigma_rej = 0.
    resample = 0

    random.seed(seednum)

    true_params = [5.88, 6.74, 0.09, 192]  # [R1, t1, R2, burden] from Table 2 in Lintusaari et al paper (https://wellcomeopenresearch.org/articles/4-14).
    d2 = 5.95

    # simulate observations using the true parameter values
    n = 1000

    R1_true = true_params[0]
    t1_true = true_params[1]
    R2_true = true_params[2]

    a2 = operator.mul(R2_true, d2)
    a1 = ops.Rt_to_a(R1_true, t1_true)
    d1 = ops.Rt_to_d(R1_true, t1_true)

    list_fm_obs = []
    for i in np.arange(0, n):
        print("producing %d th true observation" %(i))
        observed = simulator_func([true_params[3], [a2], [d2], [a1], [d1]])
        fm_observed = feature_map(observed)
        list_fm_obs.append(fm_observed)

    """ ABC """
    ss_clipping_norm_bound = 200
    Bm = ss_clipping_norm_bound/n # this is sensitivity
    print('our choice of Bm is', Bm)

    """ (2) Private reject-ABC """
    if epsilon_total ==1e6:
        sigma_rej = 0
    else:
        if resample==1:
            sigma_rej = 2*c_stop*Bm/epsilon_total
        else: # resample == 0
            sigma_rej = (c_stop+1)*Bm/epsilon_total

    """ (3) running ABCDP """
    eps_abc_noised = epsilon_rej + np.random.laplace(loc=0, scale=sigma_rej)

    mean_obs_bounds = (0, 350)
    t1_bound = 30
    a1_bound = 40

    indicators = np.zeros(n_samples)
    dis_vec = np.zeros(n_samples)
    counter = 0
    list_params = []

    for i in range(n_samples):

        param_i, others_i = prior(mean_obs_bounds, t1_bound, a1_bound, d2)
        param_i = np.array(param_i)

        others_i = np.expand_dims(np.array(others_i, dtype='float64'), axis=0)
        list_params.append(others_i)

        total_dis = np.zeros(n)
        for j in np.arange(0, n):
            # print("producing %d th pseudo observation" % (j))
            # a pseudo dataset
            pseudo_j = simulator_func(param_i)
            fm_pseudo = feature_map(pseudo_j)

            dis = ops.distance(fm_pseudo[0], fm_pseudo[1], fm_pseudo[2], fm_pseudo[3], fm_pseudo[4],
                               observed=list_fm_obs[j])

            if epsilon_total < 1e6:
                dis = min(dis, ss_clipping_norm_bound)  # then we have to clip each distance to match sensitivity Bm

            total_dis[j] = dis

        avg_dis = np.sum(total_dis)/n
        print(i, 'th parameter sample from the prior drawn')
        print('avg_dis is', avg_dis)
        dis_vec[i] = avg_dis + np.random.laplace(loc=0, scale=2 * sigma_rej)

        if (dis_vec[i] <= eps_abc_noised):
            print("%d th sample accepted" %(i))
            indicators[i] = 1.0
            counter += 1
            # We also have to update the threshold
            if resample==1:
                eps_abc_noised = epsilon_rej + np.random.laplace(loc=0, scale=sigma_rej)
            if counter >= c_stop:
                break
            else:
                pass
        else:
            pass

    prim_post_sam_abcdp_rej = np.concatenate(list_params, axis=0)

    # Get the weighted posterior samples from rejectionABCDP
    param_released = np.where(indicators == 1.0)[0]
    chosen_posterior_samps = prim_post_sam_abcdp_rej[param_released, :]
    posterior_mean_abcdp_rej = np.mean(chosen_posterior_samps, 0)
    print('posterior mean from rej-abcdp is', posterior_mean_abcdp_rej)
    print('true_params are', true_params)

    err =  mean_abs_error(true_params, posterior_mean_abcdp_rej)
    print('err is ', err)

    plt.figure(1)
    plt.subplot(221)
    plt.hist(chosen_posterior_samps[:, 0], label='R1', bins=10)
    plt.subplot(222)
    plt.hist(chosen_posterior_samps[:, 1], label='t1', bins=10)
    plt.subplot(223)
    plt.hist(chosen_posterior_samps[:, 2], label='R2', bins=10)
    plt.subplot(224)
    plt.hist(chosen_posterior_samps[:, 3], label='burden rate',bins=10)

    # plt.show()

    plot_name = os.path.join(Results_PATH, 'n=%s_epstot=%s_resample=%i' % (n, epsilon_total, resample))
    np.save(plot_name+'poster_samps.npy', chosen_posterior_samps)
    plt.savefig(plot_name+'.pdf')


if __name__ == '__main__':
    main()

#
    # reweighted_hist = param_mat * normed_W[:, None]
    # plt.subplot(221)
    # plt.hist(reweighted_hist[:,3], label='burden rate')
    # plt.subplot(222)
    # plt.hist(reweighted_hist[:,0], label='R1')
    # plt.subplot(223)
    # plt.hist(reweighted_hist[:,2], label='R2')
    # plt.subplot(224)
    # plt.hist(reweighted_hist[:,1], label='t1')
    # plt.show()

    # np.mean(dist_mat)
    # np.var(dist_mat)

