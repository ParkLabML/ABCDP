""" a toy example for showing """
# (a) Privacy/accuracy trade-off in ABC
# (b) Interplay between epsilon_abc and sigma_soft

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
import auxiliary_files.kernel as kernel
import auxiliary_files.feature as feature
import auxiliary_files.util as util
import auxiliary_files.abcbase as ab
import seaborn as sns
from autodp import privacy_calibrator
import sys
import pandas as pd
from auxiliary_files.gaussian_approx_dp import compute_epsilon_Gaussian_RDP_cgreater1 , compute_epsilon_Gaussian_RDP_c1


Results_PATH = "/".join([os.getenv("HOME"), "Covid_results/"])

# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 18
}

plt.rc('font', **font)
plt.rc('lines', linewidth=3)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html

# prior
prior = ab.Prior.from_scipy(
    # stats.multivariate_normal(mean=[0,0,0,0,0], cov=np.eye(5))
    stats.multivariate_normal(mean=[0,0,0,0], cov=np.eye(4))
    # stats.multivariate_normal(mean=[-1,-1,40], cov=5*np.eye(3))
)

# def prior(prior_params):
#     # prior_paras = [0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
#     param0 = np.random.normal(prior_params[0], np.sqrt(prior_params[1]))
#     param1 = np.random.normal(prior_params[2], np.sqrt(prior_params[3]))
#     param2 = np.random.normal(prior_params[4], np.sqrt(prior_params[5]))
#     param3 = np.random.normal(prior_params[6], np.sqrt(prior_params[7]))
#
#     sam = np.array((param0, param1, param2, param3)).T
#     return sam


def simulator_func(n, param, seed=0):
    """
    The simulator function can be implemented in however way you like.
    The only requirement is that the first three input arguments are: n, param, seed.

    n: sample size to draw
    param: a parameter vector drawn from the prior
    """
    time_survey = np.arange(n)
    # sam = param[0] * time_survey ** 4 + param[1] * time_survey ** 3 + param[2] * time_survey**2 + param[3] * time_survey + param[4]
    sam = param[0] * time_survey ** 3 + param[1] * time_survey ** 2 + param[2] * time_survey + param[3]
    # sam = param[0] * time_survey ** 2 + param[1] * time_survey + param[2]
    return sam

# def mean_abs_error(a, b):
#     return np.mean(np.abs(a-b))

# construct a simulator based on the function
simulator = ab.Simulator.from_func(simulator_func)

def main():


    """ setup for the private version II """
    seednum = 1
    epsilon_abc = 0.05
    epsilon_total = 1e6
    c_stop = 5
    n_samples = 50000
    mechanism = "lap"
    resample = 0

    """ true observation """
    observed = np.array([
        1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
        614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
    ])

    n = len(observed) # total number of observations
    print('number of observations: ', n)
    observed = np.expand_dims(observed, axis=1)


    # K2ABC with Fourier random features
    n_features = 50000
    med = util.meddistance(observed)
    sigma2 = med ** 2
    fm = feature.RFFKGauss(sigma2, n_features=n_features, seed=seednum)
    # fm = []

    B_k = 1  # Here we use a Gaussian kernel which has B_k=1
    Bm = 2.0 / n * np.sqrt(B_k)
    print('our choice of Bm is', Bm)


    if mechanism=="gau":

        sigma = 6.0 # 3.0 (around esp=3), 2.0 (around 6), 1.0 (around 14), 0.5 (around 40)
        sigma1 = Bm*sigma
        sigma2 = 2.0*sigma1
        sigma_rej = [sigma1, sigma2]

    else:

        if epsilon_total ==1e6:
            b = 0.0
        else:
            b = 2.0*c_stop*Bm/epsilon_total

        b1 = b # Threshold noise
        b2 = 2.0*b  # Distance noise
        # change this part based on which composition method we use for computing the final privacy loss

        sigma_rej = [b1, b2]  # Contains (b, 2b).

    """ (3) private rejection-ABC with threshold resampling """
    rej_abcdp_svt = ab.rejectABCDP_svt(prior, simulator, fm, epsilon_abc, pnorm=2, normpow=1, sigma_rej=sigma_rej,
                                       c_stop=c_stop, mechanism=mechanism, resample=resample)

    """ (4) private rejection-ABC """
    # rej_abcdp_svt = ab.rejectABCDP_svt(prior, simulator, fm, epsilon_abc, epsilon_total, Bm, pnorm=2, normpow=1, sigma_rej = sigma_rej, c_stop = c_stop)

    # Get the weighted posterior samples from rejectionABCDP
    prim_post_sam_abcdp_rej, prim_weights_abcdp_rej, abc_samples_no = rej_abcdp_svt.posterior_sample(observed, n=n_samples, seed=seednum)
    param_released = np.where(prim_weights_abcdp_rej == 1.0)[0]
    posterior_mean_abcdp_rej = np.mean(prim_post_sam_abcdp_rej[param_released, :], 0)
    print('posterior mean from rej-abcdp is', posterior_mean_abcdp_rej)
    print("Number of algorithm's runs: ", abc_samples_no)

    if mechanism=="gau":
        delta = 1/n
        if resample==1:
            epsilon_total = compute_epsilon_Gaussian_RDP_cgreater1(Bm, sigma1**2, sigma2**2, c_stop, abc_samples_no, delta)
        else:
            epsilon_total = compute_epsilon_Gaussian_RDP_c1(Bm, sigma1**2, sigma2**2, c_stop, abc_samples_no, delta)
        print('esp tot is', epsilon_total)

    """ simulate the final trajectory """

    accepted_samps = prim_post_sam_abcdp_rej[param_released, :]
    extend_time = np.arange(n)
    plt.figure(2)

    for i in np.arange(0,c_stop):
        y_inferred = simulator.cond_sample(n, accepted_samps[i,:], seednum)
        plt.plot(extend_time, y_inferred, 'o', color='grey')

    plt.title('Non-private ABC')
    plt.plot(extend_time, y_inferred, 'o', color='grey', label='simulated')

    y_inferred = simulator.cond_sample(n, posterior_mean_abcdp_rej, seednum)
    plt.plot(extend_time, y_inferred, 'x', color='blue', label='simulated|mean')
    plt.plot(extend_time, observed, 'o', color="red", label='observed')
    plt.xlabel('time')
    plt.legend()
    plt.ylabel('number of covid patients')
    plt.savefig('Non-private.png')

    # save_title = '%s_eps_tot=%i_n_runs=%i'%(mechanism,epsilon_total,abc_samples_no)
    # plt.title(save_title)
    # plt.legend(loc='upper left')
    # print(save_title)
    # plt.savefig(save_title+'.png')

    # """ hist """
    # final_samps = prim_post_sam_abcdp_rej[param_released, :]
    #
    # plt.figure(1)
    # plt.title('posterior samples (rejABC)')
    # plt.subplot(141)
    # plt.title('a_3')
    # plt.hist(final_samps[:,0], label='a_3')
    # plt.axvline(final_samps[:, 0].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(142)
    # plt.title('a_2')
    # plt.hist(final_samps[:,1], label='a_2')
    # plt.axvline(final_samps[:,1].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(143)
    # plt.title('a_1')
    # plt.hist(final_samps[:,2], label='a_1')
    # plt.axvline(final_samps[:, 2].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(144)
    # plt.title('a_0')
    # plt.hist(final_samps[:,3], label='a_0')
    # plt.axvline(final_samps[:, 3].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    #
    #
    # """ hist for prior """
    # final_samps = prim_post_sam_abcdp_rej
    #
    # plt.figure(3)
    # plt.title('prior samples (rejABC)')
    # plt.subplot(141)
    # plt.title('a_3')
    # plt.hist(final_samps[:,0], label='a_3')
    # plt.axvline(final_samps[:, 0].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(142)
    # plt.title('a_2')
    # plt.hist(final_samps[:,1], label='a_2')
    # plt.axvline(final_samps[:, 1].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(143)
    # plt.title('a_1')
    # plt.hist(final_samps[:,2], label='a_1')
    # plt.axvline(final_samps[:, 2].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    # plt.subplot(144)
    # plt.title('a_0')
    # plt.hist(final_samps[:,3], label='a_0')
    # plt.axvline(final_samps[:, 3].mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.xlim(-3, 3)
    #
    #
    #
    # plt.show()



if __name__ == '__main__':
    main()