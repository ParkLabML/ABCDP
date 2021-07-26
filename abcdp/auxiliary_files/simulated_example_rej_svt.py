""" a toy example for showing """
# (a) Privacy/accuracy trade-off in ABC
# (b) Interplay between epsilon_abc and sigma_soft

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
import kernel as kernel
import feature as feature
import util as util
import abcbase as ab
import seaborn as sns
#from autodp import privacy_calibrator
import sys


results_path_resampling = "/".join([os.getenv("HOME"), "laplace_resampling/"])
results_path_no_resampling= "/".join([os.getenv("HOME"), "laplace_no_resampling/"])
# Create the folder for containing the figures.
os.makedirs(results_path_resampling , exist_ok=True)
os.makedirs(results_path_no_resampling , exist_ok=True)

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

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html

# prior
prior = ab.Prior.from_scipy(
    stats.dirichlet(alpha=np.ones(5))
)


def simulator_func(n, param, seed=0):
    """
    The simulator function can be implemented in however way you like.
    The only requirement is that the first three input arguments are: n, param, seed.

    n: sample size to draw
    param: a parameter vector drawn from the Dirichlet
    """
    n_components = 5
    uniforms = [ab.MarginalSampler.from_scipy(
        stats.uniform(loc=i - 1, scale=1)
    ) for i in range(1, n_components + 1)]

    cond = ab.MixtureSampler(samplers=uniforms, pmix=param)
    sam = cond.sample(n, seed=seed)
    return sam

def mean_abs_error(a, b):
    return np.mean(np.abs(a-b))

# construct a simulator based on the function
simulator = ab.Simulator.from_func(simulator_func)

def main():

    seednum = int(sys.argv[1])
    n = int(sys.argv[2]) # how many samples we draw for both true and pseudo-observations
    n_samples = int(sys.argv[3])
    c_stop = int(sys.argv[4])
    epsilon_rej = float(sys.argv[5])
    epsilon_total = float(sys.argv[6]) # for non-private, input epsilon_total = 1e6, then we set sigma_rej = 0.

#    seednum = 0
#    n = 5000 # how many samples we draw for both true and pseudo-observations
#    c_stop = 3
#    n_samples = 10000
#    epsilon_rej = 0.1
#    epsilon_total = 1e6 # for non-private, input epsilon_total = 1e6, then we set sigma_rej = 0.
#    priv_param=200

    """ generate y* """
    true_param = np.array([0.25, 0.04, 0.33, 0.04, 0.34])
    observed = simulator.cond_sample(n=n, param=true_param, seed=seednum)


    # K2ABC with Fourier random features
    n_features = 100
    sigma2 = 0.6 ** 2
    fm = feature.RFFKGauss(sigma2, n_features=n_features, seed=seednum)

    B_k = 1.0  # Here we use a Gaussian kernel which has B_k=1
    Bm = 2.0 / n * np.sqrt(B_k)
    print('our choice of Bm is', Bm)
    
    sigma_rej_resampling=[]
    
    #Compute privacy_parameter
    priv_param_resampling = (2*c_stop)/epsilon_total
    priv_param_no_resampling = (c_stop + 1)/epsilon_total
    """ (2) Private reject-ABC """
    
    #Resampling threshold case
    b1 = Bm*priv_param_resampling #Threshold noise
    print("B1 with resampling: ", b1)
    b2=2.0*Bm*priv_param_resampling #Distance noise 
    print("B2 with resampling: ", b2 )
    # change this part based on which composition method we use for computing the final privacy loss
        
    sigma_rej_resampling.append(b1) #Threshold noise (b)
    sigma_rej_resampling.append(b2) #Distance noise (2b)
    sigma_rej_resampling=np.array(sigma_rej_resampling) #Contains (b, 2b).
    
    #Not resampling threshold case
    sigma_rej_no=[]
    
    b1_no = Bm*priv_param_no_resampling #Threshold noise
    print("B1 without resampling: ", b1_no)
    b2_no=2.0*Bm*priv_param_no_resampling #Distance noise 
    print("B2 without resampling: ", b2_no )
    # change this part based on which composition method we use for computing the final privacy loss
        
    sigma_rej_no.append(b1_no) #Threshold noise (b)
    sigma_rej_no.append(b2_no) #Distance noise (2b)
    sigma_rej_no=np.array(sigma_rej_no) #Contains (b, 2b).

    """ (3) private rejection-ABC with threshold resampling """
    rej_abcdp_svt = ab.rejectABCDP_svt(prior, simulator, fm, epsilon_rej, pnorm=2, normpow=1, sigma_rej = sigma_rej_resampling, c_stop = c_stop, mechanism="lap", resample=1)

    # Get the weighted posterior samples from rejectionABCDP
    prim_post_sam_abcdp_rej, prim_weights_abcdp_rej, abc_samples = rej_abcdp_svt.posterior_sample(observed, n=n_samples, seed=seednum)
    param_released = np.where(np.abs(prim_weights_abcdp_rej - 1.0) < 1e-7)[0]
    posterior_mean_abcdp_rej = np.mean(prim_post_sam_abcdp_rej[param_released, :], 0)
    print('posterior mean from rej-abcdp is', posterior_mean_abcdp_rej)

    print('True parameter: ')

    print(true_param)
    err = mean_abs_error(true_param, posterior_mean_abcdp_rej)

    print('err from abcdp is', err)
    
#    epsilon_total_c= 2*c_stop/priv_param #Compute the epsilon after c runs (with resampling noise)
#    print('total epsilon: ', epsilon_total_c)
    

    """ (4) Saving all the results """

    method = os.path.join(results_path_resampling , 'seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(method+'.npy', posterior_mean_abcdp_rej)

    method_err = os.path.join(results_path_resampling , 'err_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(method_err+'.npy', err)

#    eps_c=os.path.join(results_path_resampling , 'epsilontotal_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
#    np.save(eps_c+'.npy', epsilon_total_c)
    
    runs=os.path.join(results_path_resampling , 'runs_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(runs+'.npy', abc_samples)
    
    """ (5) private rejection-ABC with no resampling threshold """
    rej_abcdp_svt_no = ab.rejectABCDP_svt(prior, simulator, fm, epsilon_rej, pnorm=2, normpow=1, sigma_rej = sigma_rej_no, c_stop = c_stop, mechanism="lap", resample=0)

    # Get the weighted posterior samples from rejectionABCDP
    prim_post_sam_abcdp_rej_no, prim_weights_abcdp_rej_no, abc_samples_no = rej_abcdp_svt_no.posterior_sample(observed, n=n_samples, seed=seednum)
    param_released_no = np.where(np.abs(prim_weights_abcdp_rej_no - 1.0) < 1e-7)[0]
    posterior_mean_abcdp_rej_no = np.mean(prim_post_sam_abcdp_rej_no[param_released_no, :], 0)

    print('posterior mean from rej-abcdp with no resampling threshold is', posterior_mean_abcdp_rej_no)
    print('True parameter: ')
    print(true_param)

    err_no = mean_abs_error(true_param, posterior_mean_abcdp_rej_no)

    print('err from abcdp is', err_no)
    
#    epsilon_total_no= (1+c_stop)/priv_param #Compute the epsilon after c runs (with resampling noise)
#    print('total epsilon: ', epsilon_total_no)

    """ (6) Saving all the results """

    method_no = os.path.join(results_path_no_resampling , 'seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(method_no+'.npy', posterior_mean_abcdp_rej_no)

    method_err_no = os.path.join(results_path_no_resampling , 'err_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(method_err_no+'.npy', err_no)

#    eps_c_no=os.path.join(results_path_no_resampling , 'epsilontotal_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_privparam=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
#    np.save(eps_c_no+'.npy', epsilon_total_no)

    runs_no=os.path.join(results_path_no_resampling , 'runs_seed=%s_n=%s_nsamps=%s_c_stop=%s_epsabc=%s_epstot=%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    np.save(runs_no+'.npy', abc_samples_no)    
    
if __name__ == '__main__':
    main()



