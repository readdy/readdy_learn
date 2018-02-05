import numpy as np

import readdy_learn.example.regulation_network as regulation_network
import readdy_learn.analyze.validation as validation


rn = regulation_network.RegulationNetwork()
rn.noise_variance = .01
val = validation.Validation(rn, True, 2)

result = val.validate(np.logspace(-6, -1, num=10), lambdas=np.linspace(0, 1, num=2), realizations=2)

print(result)
