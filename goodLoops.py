# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:26:32 2021

@author: schue
"""

import matplotlib
import numpy as np
from scipy.stats import expon
import pandas as pd
from statistics import mean, median
from numpy.random import normal
from matplotlib import pyplot
from numpy.random.mtrand import exponential
from scipy import stats
import SteinschlagGrafiken


def FindmostfittingDistribution(MatrixColumn):

    # List of available Distributions for fitting in scipy
    list_of_dists = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto', 'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant',
                     'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max']

    results = []
    for i in list_of_dists:
        dist = getattr(stats, i)
        param = dist.fit(MatrixColumn)
        a = stats.kstest(MatrixColumn, i, args=param)
        results.append((i, a[0], a[1]))

    results.sort(key=lambda x: float(x[2]), reverse=True)
    for j in results:
        print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))




# Read first CSV File
dataFile1 = pd.read_csv("out_1.csv")
# Read second CSV File
dataFile2 = pd.read_csv("out_2.csv")

# Add Zone columns
dataFile1['zone'] = '1'
dataFile2['zone'] = '2'

#Get prepared Dataframe from Steinschlaggrafik
#Der Vorteil des DFs gegenüber dataFilex ist, dass hier zusätzliche Infos wie Energie, TimebeforeStone, formatierte Zeitangaben etc vorhanden sind.
#Für die Verwendung müsste Mainloop for zone Calculations ein bisschen angepasst werden.
mergedDataFile = SteinschlagGrafiken.PassDataframe()

FileZone = [dataFile1, dataFile2]
# listfeatures_distributions = [["mass","exponential"],["velocity","normal"],["time","exponential"]]
listfeatures_distributions = [["mass", "exponential"]]
sizeMonteCarloSim = 1_000_000


# Mainloop for zone calculations
for FileZone in FileZone:
    # calc fit dist features (for )
    print(FileZone['zone'])
    for featureDistribution in listfeatures_distributions:
        # calc fit dist features case when dist predefined
        print(featureDistribution[0])
        print(featureDistribution[1])
        if (featureDistribution[1] == "exponential"):
            explambda = mean(FileZone[featureDistribution[0]])
            # generate sample
            sample = exponential(explambda, sizeMonteCarloSim)
            print(sample)
            featureDistribution= np.append(featureDistribution,sample)
            print(featureDistribution)
        elif(featureDistribution[1] == "normal"):
            sample = normal(size=sizeMonteCarloSim)
            print(sample)

    # calc monte carlo
    # return calc monte carlo
