"""Module containing classes implementing basic concepts in ABC. """

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.stats as stats
import auxiliary_files.util as util
from future.utils import with_metaclass

import auxiliary_files.kernel as kernel
import warnings
import sys


class MarginalSampler(with_metaclass(ABCMeta, object)):
    """
    A marginal sampler.
    """
    @abstractmethod
    def sample(self, n, seed=1):
        """
        n: number of items to sample.

        Return a a numpy array of size (n, ...,...). The returned samples
        should be deterministic given the seed.
        """
        raise NotImplementedError()

    @staticmethod
    def from_scipy(dist):
        return MarginalSamplerScipy(dist)

# end MarginalSampler


class MixtureSampler(object):
    """
    A sampler for sampling from a mixture distributions. The components of the
    mixture are represented as MarginalSampler's.
    """
    def __init__(self, samplers, pmix=None):
        """
        samplers: a list of length k consisting of k MarginalSampler's
            representing the k components.
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k = len(samplers)
        if pmix is None:
            pmix = np.ones(k)/float(k)
        if len(pmix) != k:
            raise ValueError('The length of pmix is {}. But the lenght of samplers is {}. Must match'.format(len(pmix), len(samplers)))

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1. Was {}'.format(np.sum(pmix)))

        self.pmix = pmix
        self.samplers = samplers


    def sample(self, n, seed=29):
        pmix = self.pmix
        samplers = self.samplers

        sam_list = []
        with util.NumpySeedContext(seed=seed):
            # counts for each mixture component
            counts = np.random.multinomial(n, pmix, size=1)

            # counts is a 2d array
            counts = counts[0]

            sam_list = []
            # For each component, draw from its corresponding mixture component.
            for i, nc in enumerate(counts):
                # Sample from ith component
                i_sampler = samplers[i]
                Xi = i_sampler.sample(nc, seed+i+1)
                if len(Xi.shape) == 1:
                    Xi = Xi[:, np.newaxis]
                sam_list.append(Xi)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        return sample

# end MixtureSampler



class Prior(MarginalSampler):
    """
    An object representing a prior distribution over parameters of interest.
    """

    @staticmethod
    def from_scipy(dist):
        """
        Construct a Prior object from a distribution object of scipy.stats.

        dist: a distribution object from scipy.stats package.
            For example, this can be
                dist = stats.dirichlet(alpha=....)
        """
        return MarginalSampler.from_scipy(dist)

# end Prior

class MarginalSamplerScipy(Prior):
    """
    A Prior object from a distribution object of scipy.stats.
    """
    def __init__(self, dist):
        """
        dist: a distribution object from scipy.stats package.
            For example, this can be
                dist = stats.dirichlet(alpha=....)
        """
        self.dist = dist

    def sample(self, n, seed=2):
        dist = self.dist
        with util.NumpySeedContext(seed=seed):
            sam = dist.rvs(size=n)
        return sam

# end MarginalSamplerScipy


class CondSampler(with_metaclass(ABCMeta, object)):
    """
    A conditional sampler. This can represent a forward simulator in
    approximate Bayesian computation (ABC).
    Implement a sampler for p(x|param).
    """
    @abstractmethod
    def cond_sample(self, n, param, seed=1):
        """
        param: a parameter vector on which the sample is conditioned.
        n: sample size

        Return a numpy array of size (n,...,..) representing the draw samples.
        """
        raise NotImplementedError()

# end CondSampler

class CondSamplerFromFunc(CondSampler):
    """
    A CondSampler implemented by directly specifying a function.
        f: (n, param, seed) |-> samples.

    """
    def __init__(self, f):
        self.f = f

    def cond_sample(self, n, param, seed=2):
        f = self.f
        sam = f(n, param, seed)
        if sam.shape[0] != n:
            raise ValueError('f returns {} samples when n={} is specified.'.format(sam.shape[0], n))
        return sam

# end CondSamplerFromFunc


class Simulator(CondSampler):
    """
    An ABC simulator. The same as a CondSampler.
    """
    @staticmethod
    def from_func(f):
        """
        Implement a Simulator directly with a function f.
        f: (n, param, seed) |-> samples.
        """
        return CondSamplerFromFunc(f=f)

# end Simulator


class ABC(object):
    """
    A class representing an approximate Bayesian computation (ABC) algorithm.
    """

    @abstractmethod
    def posterior_sample(self, observed, n, seed=1):
        """
        observed: a numpy array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate

        Run the algorithm until n samples are generated from the posterior.
        Return a weighted empirical distribution. Deterministic given the seed.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A. Weights are in [0,1], and sum to 1.
        """
        raise NotImplementedError()

# end ABC

class K2ABC(ABC):
    """
    K2-ABC algorithm of

    K2-ABC: Approximate Bayesian Computation with Kernel Embeddings
    Mijung Park, Wittawat Jitkrittum, Dino Sejdinovic
    AISTATS 2016
    """
    def __init__(self, prior, simulator, k, epsilon):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        k: a kernel of type kernel.Kernel used to compute MMD
        epsilon: epsilon parameter in the K2-ABC (soft threshold)
        """
        self.prior = prior
        self.simulator = simulator
        self.k = k

        assert epsilon > 0, 'epsilon must be positive. Was {}'.format(epsilon)
        self.epsilon = epsilon

    def posterior_sample(self, observed, n, seed=1, pseudo_sample_size=None):
        """
        observed: a numpy array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.
        """
        m = observed.shape[0]
        if pseudo_sample_size is None:
            pseudo_sample_size = m

        prior = self.prior
        simulator = self.simulator
        # kernel k
        k = self.k
        epsilon = self.epsilon

        # begin K2-ABC
        mmd2s = np.zeros(n)
        list_params = []
        for i in range(n):
            # param_i.shape == (1, ...)
            param_i = prior.sample(n=1, seed=seed+286+i)
            list_params.append(param_i)
            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0],
                    seed=94+i)
            assert np.all(observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(observed.shape, pseudo_i.shape)

            # compute MMD^2
            mmd2 = kernel.mmd2_biased(observed, pseudo_i, k)

            # weight of the pseudo dataset i
            mmd2s[i] = mmd2

        unnorm_W = np.exp(-mmd2s/epsilon)
        normed_W = unnorm_W/np.sum(unnorm_W)
        assert np.all(normed_W >= 0), 'Weights contain negative numbers: {}'.format(normed_W)
        # print(list_params)
        params = np.concatenate(list_params, axis=0)
        return params, normed_W

# end K2ABC


class PrimalK2ABC(ABC):
    """
    K2-ABC algorithm of

    K2-ABC: Approximate Bayesian Computation with Kernel Embeddings
    Mijung Park, Wittawat Jitkrittum, Dino Sejdinovic
    AISTATS 2016

    with a finite-dimensional feature map instead of the quadratic-time MMD^2
    estimator. Random Fourier features can be used by specifying an appropriate
    FeatureMap object. In other words, the distance between the observed set
    and a pseudo dataset is given by

    p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow

    By default, p-norm = 2, normpow = 1.
    """
    def __init__(self, prior, simulator, fm, epsilon, pnorm=2, normpow=1):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        fm: a feature map function of type feature.FeatureMap
        epsilon: epsilon parameter in the K2-ABC (soft threshold)
        pnorm, normpow: p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow
        """
        self.prior = prior
        self.simulator = simulator
        self.fm = fm
        self.pnorm = pnorm
        self.normpow = normpow

        assert epsilon > 0, 'epsilon must be positive. Was {}'.format(epsilon)
        self.epsilon = epsilon


    def posterior_sample(self, observed, n, seed=1, pseudo_sample_size=None):
        """
        observed: a numpy array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.
        """
        m = observed.shape[0]
        if pseudo_sample_size is None:
            pseudo_sample_size = m

        prior = self.prior
        simulator = self.simulator
        # feature map
        fm = self.fm
        epsilon = self.epsilon

        # begin K2-ABC
        dis_vec = np.zeros(n)
        list_params = []

        observed_mean_feature = np.mean(fm(observed), axis=0)

        for i in range(n):
            # param_i.shape == (1, ...)
            if i%100==0:
                print(i, 'th pseudo data samples generated')
            param_i = prior.sample(n=1, seed=seed+286+i)
            if len(np.shape(param_i))==1:
                param_i = np.expand_dims(param_i, axis=0)
            list_params.append(param_i)
            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0],
                    seed=94+i)

            if len(np.shape(observed))==2:
                pseudo_i = np.expand_dims(pseudo_i, axis=1)
            assert np.all(observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(observed.shape, pseudo_i.shape)
            pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
            # find the p-norm distance
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature,
                    ord=self.pnorm)**self.normpow

            # weight of the pseudo dataset i
            dis_vec[i] = dis

        unnorm_W = np.exp(-dis_vec/epsilon)
        normed_W = unnorm_W/np.sum(unnorm_W)
        assert np.all(normed_W >= 0), 'Weights contain negative numbers: {}'.format(normed_W)
        # print(list_params)
        params = np.concatenate(list_params, axis=0)
        return params, normed_W

# end PrimalK2ABC


class softABCDP(ABC):
    """
    soft ABCDP algorithm : DP version of

    K2-ABC: Approximate Bayesian Computation with Kernel Embeddings
    Mijung Park, Wittawat Jitkrittum, Dino Sejdinovic
    AISTATS 2016

    with a finite-dimensional feature map instead of the quadratic-time MMD^2
    estimator. Random Fourier features can be used by specifying an appropriate
    FeatureMap object. In other words, the distance between the observed set
    and a pseudo dataset is given by

    p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow

    By default, p-norm = 2, normpow = 1.
    """

    def __init__(self, prior, simulator, fm, epsilon, pnorm=2, normpow=1, sigma_soft=0):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        fm: a feature map function of type feature.FeatureMap
        epsilon: epsilon parameter in the K2-ABC (soft threshold)
        pnorm, normpow: p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow
        sigma_soft: noise standard deviation in each sampling step
        """
        self.prior = prior
        self.simulator = simulator
        self.fm = fm
        self.pnorm = pnorm
        self.normpow = normpow

        assert epsilon > 0, 'epsilon must be positive. Was {}'.format(epsilon)
        self.epsilon = epsilon
        self.sigma_soft = sigma_soft

    def posterior_sample(self, observed, n, seed=1, pseudo_sample_size=None):
        """
        observed: a numpy array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.
        """
        m = observed.shape[0]
        if pseudo_sample_size is None:
            pseudo_sample_size = m

        prior = self.prior
        simulator = self.simulator
        # feature map
        fm = self.fm
        epsilon = self.epsilon

        # begin K2-ABC
        dis_vec = np.zeros(n)
        list_params = []

        observed_mean_feature = np.mean(fm(observed), axis=0)

        for i in range(n):
            # param_i.shape == (1, ...)
            param_i = prior.sample(n=1, seed=seed + 286 + i)
            if len(np.shape(param_i))==1:
                param_i = np.expand_dims(param_i, axis=0)
            list_params.append(param_i)
            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0],
                                             seed=94 + i)
            if len(np.shape(observed))==2:
                pseudo_i = np.expand_dims(pseudo_i, axis=1)
            assert np.all(
                observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(
                observed.shape, pseudo_i.shape)
            pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
            # find the p-norm distance
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature,
                                    ord=self.pnorm) ** self.normpow

            # weight of the pseudo dataset i
            # dis_vec[i] = dis

            # assuming the normpow = 1, and Gaussian kernel
            noisy_dis = dis/epsilon + self.sigma_soft*np.random.randn(1)
            dis_vec[i] = max(noisy_dis,0)

        # unnorm_W = np.exp(-dis_vec / epsilon)
        unnorm_W = np.exp(-dis_vec)
        normed_W = unnorm_W / np.sum(unnorm_W)
        assert np.all(normed_W >= 0), 'Weights contain negative numbers: {}'.format(normed_W)
        # print(list_params)
        params = np.concatenate(list_params, axis=0)
        return params, normed_W

# end softABCDP


class rejectABCDP(ABC):
    """
    DP version of rejection ABC using a finite-dimensional feature map as a similarity measure.
    Random Fourier features can be used by specifying an appropriate
    FeatureMap object. In other words, the distance between the observed set
    and a pseudo dataset is given by

    p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow

    By default, p-norm = 2, normpow = 1.
    """

    def __init__(self, prior, simulator, fm, epsilon, Bm, pnorm=2, normpow=1, sigma_rej = 0):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        fm: a feature map function of type feature.FeatureMap
        epsilon: epsilon parameter in the K2-ABC (soft threshold)
        pnorm, normpow: p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow
        """
        self.prior = prior
        self.simulator = simulator
        self.fm = fm
        self.pnorm = pnorm
        self.normpow = normpow

        assert epsilon > 0, 'epsilon must be positive. Was {}'.format(epsilon)
        self.epsilon = epsilon
        self.sigma_rej = sigma_rej
        self.Bm = Bm

    def posterior_sample(self, observed, n, seed=1, pseudo_sample_size=None, observed_mean_feature=None):
        """
        observed: a number array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.
        """
        m = observed.shape[0]
        if pseudo_sample_size is None:
            pseudo_sample_size = m

        prior = self.prior
        simulator = self.simulator
        # feature map
        fm = self.fm
        epsilon = self.epsilon

        # begin K2-ABC
        dis_vec = np.zeros(n)
        list_params = []

        if observed_mean_feature.shape[0]==0:
            observed_mean_feature = np.mean(fm(observed), axis=0)
        else:
            print('we got the embedding of observed data')

        for i in range(n):

            if i%100==0:
                print(i,'th pseudo-data generated')
            # param_i.shape == (1, ...)
            param_i = prior.sample(n=1, seed=seed + 286 + i)
            list_params.append(param_i)
            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0],
                                             seed=94 + i)
            assert np.all(
                observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(
                observed.shape, pseudo_i.shape)
            pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
            # find the p-norm distance
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature,
                                    ord=self.pnorm) ** self.normpow

            # weight of the pseudo dataset i
            if self.sigma_rej==0:
                dis_vec[i] = dis
            else:
                dis_clipped = min(dis, self.Bm)
                dis_vec[i] = dis_clipped + self.sigma_rej*np.random.randn(1)

        indicators = (dis_vec<=epsilon)*1
        # unnorm_W = np.exp(-dis_vec / epsilon)
        # normed_W = unnorm_W / np.sum(unnorm_W)
        # assert np.all(normed_W >= 0), 'Weights contain negative numbers: {}'.format(normed_W)
        # print(list_params)
        params = np.concatenate(list_params, axis=0)
        return params, indicators


    @staticmethod
    def log_eps_dp(N, Bk, sigma, eps_abc, Bm):
        """
        Compute the log(epsilon_DP) using the derived upper bound.

        N: the sample size
        Bk: bound on the kernel
        sigma: standard deviation of the Gaussian noise in epsilon DP
        eps_abc: epsilon ABC (rejection threshold)

        This method assumes that \Delta_\rho (bound on the difference of two MMDs computed on adjacent datasets.) = (2/N)*Bk.
        """

        # Bm = 2*np.sqrt(Bk)
        in_sqrt = np.log(Bk)-np.log(2*np.pi) - 2.0*np.log(sigma)
        log_fac = np.minimum(0,  0.5*in_sqrt +np.log(2.0)-np.log(N) )
        min_in = np.minimum(eps_abc-Bm, -eps_abc)
        log_Phi = stats.norm.logcdf(min_in/sigma)
        log_eps_dp = log_fac - log_Phi
    #     print(Phi)
        return log_eps_dp


    @staticmethod
    def binary_search_sigma(N, Bk, eps_dp, eps_abc, Bm, tol=1e-12, verbose=False,
                            abs_sigma_lb=None):
        """
        Use binary search to approximately invert the bound to get sigma, given
        the target epsilon DP.

        N: the sample size
        Bk: upper bound on the kernel
        eps_dp: Target epsilon DP
        eps_abc: epsilon ABC (rejection threshold)
        Bm: upper bound on the MMD (not squared)
        tol: error tolerance on the function values.
            If the actual error < tol, the assume found.
        verbose: set to True to print out information during the search
        """
        if abs_sigma_lb is not None:
            warnings.warn('No longer need to specify abs_sigma_lb. It can be safely removed')

        def f(x):
            return rejectABCDP.log_eps_dp(N, Bk, x, eps_abc, Bm)
        # Use the log version
        target = np.log(eps_dp)

        # Establish a lower bound. First start from an arbitrary sigma.
        arbitrary_sigma = 10.0
        f_a = f(arbitrary_sigma)
        while f_a < target:
            arbitrary_sigma /= 2.0
            f_a = f(arbitrary_sigma)
        lb = arbitrary_sigma

        # get an upper bound of possible sigma
        ub = lb
        f_ub = f(ub)
        if verbose:
            print('ub = {}, f_ub = {}, target = {:.4g}'.format(ub, f_ub, target))
        while f_ub > target:
            ub *= 2
            f_ub = f(ub)
            if verbose:
                print('ub = {}, f_ub = {}, target = {:.4g}'.format(ub, f_ub, target))

        if verbose:
            print('Begin search for sigma in the interval ({:.4g},{:.4g})'.format(lb, ub))

        # (lb, ub) defines current search interval
        cur = (lb+ub)/2.0
        f_cur = f(cur)

        while np.abs(f_cur - target) > tol:
            if f_cur < target:
                ub = cur
            else:
                lb = cur
            cur = (lb+ub)/2.0
            f_cur = f(cur)
            if verbose:
                print('sigma={:.4g}, cur eps_ep={:.4g}, target={:.4g}'.format(
                    cur, f_cur, target))
        # Now we have found a good cur = sigma
        return cur

# end rejectABCDP


class FindBm(ABC):
    """
    Find the empirical max of MMD using a finite-dimensional feature map
    """

    def __init__(self, prior, simulator, fm, percentile, pnorm=2, normpow=1):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        fm: a feature map function of type feature.FeatureMap
        epsilon: epsilon parameter in the K2-ABC (soft threshold)
        pnorm, normpow: p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow
        """
        self.prior = prior
        self.simulator = simulator
        self.fm = fm
        self.pnorm = pnorm
        self.normpow = normpow
        self.percentile = percentile


    def posterior_sample(self, n, seed=1, pseudo_sample_size=None):
        """
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.
        """

        prior = self.prior
        simulator = self.simulator
        # feature map
        fm = self.fm

        # begin K2-ABC
        dis_vec = np.zeros(n)
        list_params = []

        if pseudo_sample_size is None:
            pseudo_sample_size = n

        # we call it observed, but this is a simulated dataset which we use as if this is observed.
        param_ref = prior.sample(n=1, seed=seed)
        observed = simulator.cond_sample(pseudo_sample_size, param=param_ref[0], seed=seed)
        observed_mean_feature = np.mean(fm(observed), axis=0)

        for i in range(n):
            # param_i.shape == (1, ...)
            param_i = prior.sample(n=1, seed=seed + 286 + i)
            list_params.append(param_i)
            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0],
                                             seed=94 + i)
            assert np.all(
                observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(
                observed.shape, pseudo_i.shape)
            pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
            # find the p-norm distance
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature,
                                    ord=self.pnorm) ** self.normpow

            # weight of the pseudo dataset i
            dis_vec[i] = dis

        # empirical_Bm = np.max(dis_vec)
        # empirical_Bm = np.mean(dis_vec)
        # empirical_Bm = 0.1
        empirical_Bm = np.percentile(dis_vec, self.percentile)

        return empirical_Bm


# end findBm

class rejectABCDP_svt(ABC):
    """
    DP version of rejection ABC + SVT using a finite-dimensional feature map as a similarity measure.
    Random Fourier features can be used by specifying an appropriate
    FeatureMap object. In other words, the distance between the observed set
    and a pseudo dataset is given by

    p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow

    By default, p-norm = 2, normpow = 1.

    It stops after 1 query is released.
    """

    def __init__(self, prior, simulator, fm, epsilon, pnorm=2, normpow=1, sigma_rej=np.zeros(2), c_stop=1000, mechanism=None, resample=0):
        """
        prior: a prior distribution for the parameters. Instance of type
            abcbase.Prior.
        simulator: a simulator of type abcbase Simulator.
        fm: a feature map function of type feature.FeatureMap
        epsilon: epsilon parameter in the rejection ABC
        pnorm, normpow: p-norm( mean_feature(observed) - mean_feature(pseudo data) )^normpow
        sigma_rej: scale of the Laplace noise. The noise is added to the ABC
            rejection threshold.
            In the Gaussian case sigme_rej is going to be a list with the 2 differente variances.
        c_stop: the maximum number of thetas that are going to be released.
        
        mechanism: The noise distribution you want to add to the threshold and distance.
            By default is set to Laplace distribution but it can be change to Gaussian.
        replace: indicates if the noised threshold is being refreshed each time the condition is met (1) or not(0).
            By defalut is set to 0 (not replacement)
        """
        self.prior = prior
        self.simulator = simulator
        self.fm = fm
        self.pnorm = pnorm
        self.normpow = normpow

        assert epsilon > 0, 'epsilon must be positive. Was {}'.format(epsilon)
        self.epsilon = epsilon
        self.sigma_rej = sigma_rej
        self.c_stop = c_stop
        self.resample=resample

        if len(sigma_rej) > 2:
            print("Too many inputs on the scale parameters. Set 1 or 2 parameters only.")
            sys.exit()
            
        if mechanism is None:
            self.mechanism="laplace"
        elif mechanism.lower().startswith("lap") or mechanism.lower()=="l":
            self.mechanism="laplace"
        elif mechanism.lower().startswith("gau") or mechanism.lower()=="g":
            self.mechanism="gaussian"
        else:
            print("The added noise is neither Laplace or Gaussian. Please, set one of these two options.")
            sys.exit()

    def posterior_sample(self, observed, n, seed=1, pseudo_sample_size=None, observed_mean_feature=None):
        """
        observed: a number array of size (m,,...,..) representing the observed
            data of m points.
        n: number of samples to generate
        pseudo_sample_size: the sample size of each pseudo dataset generated.
            If None, use m.

        Return A, B
            * A: a numpy array of size (n, ...,) representing the samples
            generated.

            * B: a 1-d numpy array of length n containing the weights of
            the samples in A.

        """
        m = observed.shape[0]
        if pseudo_sample_size is None:
            pseudo_sample_size = m

        prior = self.prior
        simulator = self.simulator
        # feature map
        fm = self.fm
        epsilon = self.epsilon
        sigma_rej = self.sigma_rej
        mechanism=self.mechanism
        resample=self.resample

        
        sigma=sigma_rej[0] #The threshold scale for the Laplacian noise and the first std for the Gaussian noise.
        sigma2=sigma_rej[1] #The distance scale for the Laplacian noise and the second std for the Gaussian noise.

        # begin ABC
        dis_vec = np.zeros(n)
        list_params = []

        if fm==[]:
            observed_mean_feature = observed
        else:
            observed_mean_feature = np.mean(fm(observed), axis=0)

        # if observed_mean_feature.shape[0]==0:
        #     observed_mean_feature = np.mean(fm(observed), axis=0)
        # else:
        #     print('we got the embedding of observed data')
        
        print("Resample is set to: ", resample)

        #The noised soft threshold.
        if mechanism == "laplace":
            # eps_abc_noised= epsilon + np.random.laplace(loc=0, scale=2*self.Bm / self.epsilon_total)
            eps_abc_noised = epsilon + np.random.laplace(loc=0, scale=sigma)
        else:            
            eps_abc_noised = epsilon + np.random.normal(loc=0, scale=sigma) 

        # print("Epsilon_total: ", self.epsilon_total)
        # print("Epsilon_abc: ", epsilon)
        # print("Epsilon_abc_noised: ", eps_abc_noised)
        indicators=np.zeros(n)
        counter=0

        for i in range(n):

            if i%1000==0:
                print(i,'th pseudo-data generated')

            param_i = prior.sample(n=1, seed=seed + 286 + i)
#            print("param_i: ", param_i)
            if len(np.shape(param_i)) == 1:
                param_i = np.expand_dims(param_i, axis=0)

            list_params.append(param_i)

            # a pseudo dataset
            pseudo_i = simulator.cond_sample(pseudo_sample_size, param=param_i[0], seed=94 + i)


#<<<<<<< Updated upstream
            #
            # if fm==[]:
            #     pseudo_mean_feature = pseudo_i
            # else:
            #     pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
# =======

            if len(np.shape(observed)) ==2:
               pseudo_i = np.expand_dims(pseudo_i, axis=1)
            assert np.all(observed.shape == pseudo_i.shape), 'The shape of the observed dataset ({}) does not match the shape of the pseudo dataset ({})'.format(
               observed.shape, pseudo_i.shape)

            pseudo_mean_feature = np.mean(fm(pseudo_i), axis=0)
#>>>>>>> Stashed changes
            # find the p-norm distance
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature,
                                    ord=self.pnorm) ** self.normpow


            # weight of the pseudo dataset i
#            if self.sigma_rej==0:
#                dis_vec[i] = dis
#            else:
#            dis_clipped = min(dis, self.Bm)
            if mechanism == "laplace":
#                print("The mechanims that is used is: ", mechanism)  
                dis_vec[i] = dis + np.random.laplace(loc=0, scale=sigma2)
            else:
                dis_vec[i] = dis + np.random.normal(loc=0, scale=sigma2)
#                print("The mechanims that is used is: ", mechanism)  

            # print("Distance: ", dis)
            # print("Noised distance: ", dis_vec[i])

            if dis_vec[i] <= eps_abc_noised:
                print('sample accepted: ', i)
                
                indicators[i]=1.0
                counter+=1
                
                if resample == 1:
                    # We  have to update the threshold
                    if mechanism == "laplace":
                        print("We are resampling Laplace threshold")
                        eps_abc_noised = epsilon + np.random.laplace(loc=0,scale=sigma)
                    else: 
                        eps_abc_noised = epsilon + np.random.normal(loc=0,scale=sigma)
                else:
                    print("We are NOT resampling the threshold")
                    
                if counter >= self.c_stop:
                    break

        #indicators = (dis_vec<=eps_abc_noised)*1
        # unnorm_W = np.exp(-dis_vec / epsilon)
        # normed_W = unnorm_W / np.sum(unnorm_W)
        # assert np.all(normed_W >= 0), 'Weights contain negative numbers: {}'.format(normed_W)
        # print(list_params)
        params = np.concatenate(list_params, axis=0)
        return params, indicators, i


# end rejectABCDP_svt
