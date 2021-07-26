""" This code is taken from https://github.com/xl0418/ABCer/blob/master/Multiple_virusmodels.py """

from ABCer import ABCer
import matplotlib.pyplot as plt
import numpy as np

def model4(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * time_survey**3 + para[1] * time_survey**2 + para[2] * time_survey + para[3]
    return y


observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

test_ABC4 = ABCer(50, 10000, observations=observations)
test_ABC4.initialize_model(model4)
test_ABC4.initialize_parameters([0.0, 1.0, 1.0, 1.0])
test_list4 = test_ABC4.ABC(prior_paras=[0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

plt.plot(time, observations, 'o')
para_inferred = []
para_inferred.append(np.mean(test_list4[0][20, :]))
para_inferred.append(np.mean(test_list4[1][20, :]))
para_inferred.append(np.mean(test_list4[2][20, :]))
para_inferred.append(np.mean(test_list4[3][20, :]))

extend_time = np.arange(18)
y_inferred = model4(para_inferred, np.arange(18))

plt.plot(extend_time, y_inferred, 'x', color='r')

plt.show()