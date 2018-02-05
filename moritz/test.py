import numpy as np

import readdy_learn.example.regulation_network as regulation_network
import readdy_learn.analyze.validation as validation

import matplotlib.pyplot as plt

rn = regulation_network.RegulationNetwork()
rn.noise_variance = .01
rn.timestep = 1e-0
val = validation.Validation(rn, True, 6)

result = val.validate(np.logspace(-6, -1, num=4), lambdas=np.linspace(0, 1, num=2), realizations=3)
validation.plot_validation_result(result)
plt.show()