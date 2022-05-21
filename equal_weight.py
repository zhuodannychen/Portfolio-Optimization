import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(assets):
    optimal = [1/len(assets) for i in range(len(assets))]
    return optimal


def minimum_variance(ret):
    def find_port_variance(weights):
        cov = ret.cov()
        port_var = np.dot(weights.T, np.dot(cov, weights))
        return port_var

    bounds_lim = [(0, 1) for x in range(len(ret.columns))] # change to (-1, 1) if you want to short
    init = [1/len(ret.columns) for i in range(len(ret.columns))]

    optimal = minimize(fun=find_port_variance,
                       x0=init
                       # bounds=bounds_lim
                       )

    return list(optimal['x'])



