import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import optimize
from scipy.optimize import Bounds
from scipy.special import comb


Results_PATH = "/".join([os.getenv("HOME"), "eps_vals/"])

def compute_epsilon_Gaussian_RDP_c1(Bm, var1, var2, c, k_max, delta_i):
    # compute the final epsilon based on Remark for Theorem 11 with only one draw for the threshold.
    # from paper: Improving sparse vector technique with RDP
    # https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf

    # Bm: sensitivity
    # var1: Gaussian noise "variance" for threshold (Remember: this is variance, not standard deviation!)
    # var2: Gaussian noise "variance" for noise to be added to distance
    # c: # of positive answers
    # k_max: maximum # of ABC runs
    # delta: failure probability of DP guarantee

    # op1 = (Bm ** 2) / (2 * var1) + (2 * c * Bm ** 2) / (var2)
    # op2 = 1 + c*np.log(k_max / c) #Upper bound for the logaritm of the sum...

    op1 = (Bm ** 2) / (2 * var1) + (2 * c *  Bm ** 2) / (var2)
    log_sum=float(sum([Decimal(math.comb(k_max, i)) for i in range(0, c)]).ln())
    op2 = op1*(1 + log_sum + np.log(1./delta_i))

    if op2 > 0:
        epsilon_val = op1 + 2*np.sqrt(op2)
        print('1 threshold draw for c=%d, k_max=%d, var1=%.2f, and var2=%.2f, epsilon is %.3f' %(c,k_max,var1, var2, epsilon_val))
    else:
        print("The squared root cannot be done. The number isn't greater or equal to 0")
        epsilon_val = np.nan
    return epsilon_val

def compute_epsilon_Gaussian_RDP_cgreater1(Bm, var1, var2, c, k_max, delta_i):
    # compute the final epsilon based on Remark for Theorem 11 for c redraws of the threshold.
    # from paper: Improving sparse vector technique with RDP
    # https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf

    # Bm: sensitivity
    # var1: Gaussian noise "variance" for threshold (Remember: this is variance, not standard deviation!)
    # var2: Gaussian noise "variance" for noise to be added to distance
    # c: # of positive answers
    # k_max: maximum # of ABC runs
    # delta: failure probability of DP guarantee

    # op1 = (Bm ** 2) / (2 * var1) + (2 * c * Bm ** 2) / (var2)
    # op2 = 1 + c*np.log(k_max / c) #Upper bound for the logaritm of the sum...

    op1 = (c*Bm ** 2) / (2 * var1) + (2 * c *  Bm ** 2) / (var2)
    op2 = op1*(c*np.log(k_max + 1) + np.log(1./delta_i))

    if op2 > 0:
        epsilon_val = op1 + 2*np.sqrt(op2)
        print('c threshold redraws for c=%d, k_max=%d, var1=%.2f, and var2=%.2f, epsilon is %.3f' %(c,k_max,var1, var2, epsilon_val))
    else:
        print("The squared root cannot be done. The number isn't greater or equal to 0")
        epsilon_val = np.nan
    return epsilon_val

def compute_variance_given_epsilon_Gaussian_RDP_c1(Bm, c, k_max, delta_i, target_epsilon):
    # compute_variance_given_epsilon_Gaussian_RDP(Bm, c, k_max, delta_i, target_epsilon)
    def func(param):

        var1 = (Bm*param)**2 # now we treat param as privacy parameter,
        var2 = (2*Bm*param)**2
        # var1 = 4*var
        # var2 = var

        # op1 = (Bm ** 2) / (2 * var1) + (2 * c * Bm ** 2) / (var2)
        # combination = comb(k_max, c) * c
        # op2 = np.log(1 / delta_i) + np.log(combination)

        op1 = (Bm ** 2) / (2 * var1) + (2 * c * Bm ** 2) / (var2)
        log_sum=float(sum([Decimal(math.comb(k_max, i)) for i in range(0, c)]).ln())
        op2 = op1*(1 + log_sum + np.log(1./delta_i))

        if op2 > 0:
            epsilon = op1 + 2*np.sqrt(op2)
            # print('for c=%d, k_max=%d, var1=%.2f, and var2=%.2f, epsilon is %.3f' % (c, k_max, var1, var2, epsilon_val))
        else:
            print("The squared root cannot be done. The number isn't greater or equal to 0")
            epsilon = np.nan

        epsilon = float(epsilon)
        return (epsilon-target_epsilon)**2

    sol = optimize.minimize_scalar(func, bounds=(0, 1e12), method='bounded')
    # x0 = np.array([5.0, 10.0])
    # bounds = Bounds([1e-6, 100], [1e-6, 100])
    # func1 = lambda x: np.abs(func(x))
    # var = optimize.fmin(func1, x0)
    # sol = optimize.fmin(func1, x0)
    var = sol.x
    # print(var)

    return var

def compute_variance_given_epsilon_Gaussian_RDP_cgreater1(Bm, c, k_max, delta_i, target_epsilon):
    # compute_variance_given_epsilon_Gaussian_RDP(Bm, c, k_max, delta_i, target_epsilon)
    def func(param):

        var1 = (Bm*param)**2 # now we treat param as privacy parameter,
        var2 = (2*Bm*param)**2
        # var1 = 4*var
        # var2 = var

        op1 = (c*Bm ** 2) / (2 * var1) + (2 * c *  Bm ** 2) / (var2)
        op2 = op1*(c*np.log(k_max + 1) + np.log(1./delta_i))


        if op2 > 0:
            epsilon= op1 + 2*np.sqrt(op2)
#            print('c threshold redraws for c=%d, k_max=%d, var1=%.2f, and var2=%.2f, epsilon is %.3f' %(c,k_max,var1, var2, epsilon))
        else:
            print("The squared root cannot be done. The number isn't greater or equal to 0")
            epsilon = np.nan

        epsilon = float(epsilon)
        return (epsilon-target_epsilon)**2

    sol = optimize.minimize_scalar(func, bounds=(0, 1e12), method='bounded')
    # x0 = np.array([5.0, 10.0])
    # bounds = Bounds([1e-6, 100], [1e-6, 100])
    # func1 = lambda x: np.abs(func(x))
    # var = optimize.fmin(func1, x0)
    # sol = optimize.fmin(func1, x0)
    var = sol.x
    # print(var)

    return var


def main():
    var1=1.0
    var2=4.0
    c_stop=[10, 100, 500, 1000, 5000]
    k_max=10000 #k_max i.e. number of max iterations on svt algorithm
    delta=[1./k_max]
    n=5000
    B_k = 1.0  # Here we use a Gaussian kernel which has B_k=1
    Bm = 2.0 / n * np.sqrt(B_k) # sensitivity of the MMD (distance)

    for delta_i in delta:

        epsilon_vals=[]
        epsilon_vals_cgreater=[]

        for c in c_stop:

            """ 1. Given a variance, compute the epsilon for 1 threshold draw """
            epsilon = compute_epsilon_Gaussian_RDP_c1(Bm, var1, var2, c, k_max, delta_i)
            epsilon_vals.append(epsilon)

            """ 2. Given a target epsilon, compute the variance  for 1 threshold draw"""
            # here we assume two variances are the same
            target_epsilon = float(epsilon)
            privacy_param = compute_variance_given_epsilon_Gaussian_RDP_c1(Bm, c, k_max, delta_i, target_epsilon)
            # print('variance1 = %.2f' % (4*estimated_variance))
            # print('variance2 = %.2f' % (estimated_variance))

            var1 = (Bm * privacy_param) ** 2
            var2 = (2 * Bm * privacy_param) ** 2
            print('For 1 threshold draw the estimated privacy parameter = %.4f' %(privacy_param))
            print('variance1 = %.6f' % (var1))
            print('variance2 = %.6f' % (var2))
            
            """ 3. Given a variance, compute the epsilon for c threshold redraws """
            epsilon2 = compute_epsilon_Gaussian_RDP_cgreater1(Bm, var1, var2, c, k_max, delta_i)
            epsilon_vals_cgreater.append(epsilon2)
            
            """ 4. Given a target epsilon, compute the variance  for c threshold redraws"""
            # here we assume two variances are the same
            target_epsilon = float(epsilon2)
            privacy_param2 = compute_variance_given_epsilon_Gaussian_RDP_cgreater1(Bm, c, k_max, delta_i, target_epsilon)
            # print('variance1 = %.2f' % (4*estimated_variance))
            # print('variance2 = %.2f' % (estimated_variance))

            var1_c = (Bm * privacy_param2) ** 2
            var2_c = (2 * Bm * privacy_param2) ** 2
            print('For c threshold redraws the estimated privacy parameter = %.4f' %(privacy_param))
            print('variance1 = %.6f' % (var1_c))
            print('variance2 = %.6f' % (var2_c))


        method = os.path.join(Results_PATH, 'delta=%s' % (delta_i))
        np.save(method+'_threshold=1.npy', np.array(epsilon_vals))
        
        method2 = os.path.join(Results_PATH, 'delta=%s' % (delta_i))
        np.save(method2+'_threshold=c.npy', np.array(epsilon_vals_cgreater))



if __name__ == '__main__':
    main()


