""" this is code is taken from https://github.com/xl0418/ABCer/blob/master/ABCer.py """

import numpy as np
from itertools import repeat
from itertools import starmap
from scipy.stats import norm


class ABCer:
    def __init__(self, iterations, particles, observations):
        self.iterations = iterations
        self.particles = particles
        self.observations = observations

    def initialize_model(self, model):
        self.model = model

    def initialize_parameters(self, paras):
        self.parameters = paras
        return self.parameters

    def normalized_norm(self, x):
        diff_norm = np.linalg.norm(x / self.observations - 1, axis=1)
        max_err = np.nanmax(diff_norm)
        return diff_norm / max_err

    def purterbation(self, index, weight, para):
        para_last_iteration = para[index]
        weight_update = weight[index] / sum(weight[index])
        mean_para_last = np.sum(para_last_iteration * weight_update)
        var_para_last = np.sum(
            (para_last_iteration - mean_para_last) ** 2 * weight_update)
        sample_index = np.random.choice(index, self.particles, p=weight_update)
        mean_sample_para = para[sample_index]
        propose_para = np.random.normal(mean_sample_para,
                                        np.sqrt(2 * var_para_last))
        evolve_weight = weight_update[index.searchsorted(sample_index)]

        evolve_weight_denominator = np.sum(evolve_weight * norm.pdf(
            propose_para, mean_sample_para, np.sqrt(2 * var_para_last)))
        evolve_weight_numerator = norm.pdf(propose_para, mean_para_last,
                                           np.sqrt(2 * var_para_last))
        evolve_weight = evolve_weight_numerator / evolve_weight_denominator
        evolve_weight = evolve_weight / sum(evolve_weight)

        return evolve_weight, propose_para

    def ABC(self, prior_paras):
        # initialize the first iteration
        number_parameters = len(self.parameters)
        if len(prior_paras) != number_parameters * 2:
            return print(
                "Provide the corresponding length of the prior information of the parameters!"
            )

        para_each_iteration = np.tile(self.parameters, (self.particles, 1))
        for i in range(number_parameters):
            para_each_iteration[:, i] = np.random.uniform(
                prior_paras[2 * i], prior_paras[2 * i + 1],
                para_each_iteration.shape[0])

        # store parameter evolution
        disct_parameters = dict.fromkeys(range(number_parameters), [])
        for key, value in disct_parameters.items():
            l = np.zeros(shape=(self.iterations + 1, self.particles))
            l[0, :] = para_each_iteration[:, key]

            disct_parameters[key] = l

        # fitness
        fitness = np.zeros(shape=(self.iterations, self.particles))

        # weights
        disct_parameter_weights = dict.fromkeys(range(number_parameters), [])
        for key, value in disct_parameter_weights.items():
            l = np.zeros(self.particles)
            l.fill(1 / self.particles)
            disct_parameter_weights[key] = l

        for g in range(self.iterations):
            packed_para = [[para_each_iteration[i, :]]
                           for i in range(para_each_iteration.shape[0])]
            simulation_each_iter_list = list(starmap(self.model, packed_para))
            distance = self.normalized_norm(simulation_each_iter_list)
            fitness[g, :] = 1 - distance

            q5 = np.argsort(
                fitness[g, :])[-int(self.particles // 4)]  # best 25%
            fit_index = np.where(fitness[g, :] > fitness[g, q5])[0]
            print('Mean estimates: parameters: %.3e ; %.3e ' %
                  (np.mean(para_each_iteration[fit_index, 0]),
                   np.mean(para_each_iteration[fit_index, 1])))

            for i in range(number_parameters):
                disct_parameter_weights[i], disct_parameters[i][
                                            g + 1, :] = self.purterbation(fit_index,
                                                                          disct_parameter_weights[i],
                                                                          disct_parameters[i][g, :])
                para_each_iteration[:, i] = disct_parameters[i][g + 1, :]

            disct_parameters['fitness'] = fitness

            # np.save(output, para_data)

        return disct_parameters
