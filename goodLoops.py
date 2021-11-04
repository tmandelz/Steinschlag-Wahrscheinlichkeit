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
from scipy.stats.stats import describe
import SteinschlagGrafiken
import scipy as sc


def FindmostfittingDistribution(MatrixColumn):

    # List of available Distributions for fitting in scipy
    list_of_dists = ['beta','cauchy','chi','chi2','expon','logistic','norm', 'pareto', 't', 'uniform']

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
# dataFile1 = pd.read_csv("out_1.csv")
# Read second CSV File
# dataFile2 = pd.read_csv("out_2.csv")

# Add Zone columns
# dataFile1['zone'] = '1'
# dataFile2['zone'] = '2'

#Get prepared Dataframe from Steinschlaggrafik
#Der Vorteil des DFs gegenüber dataFilex ist, dass hier zusätzliche Infos wie Energie, TimebeforeStone, formatierte Zeitangaben etc vorhanden sind.
#Für die Verwendung müsste Mainloop for zone Calculations ein bisschen angepasst werden.
mergedDataFile = SteinschlagGrafiken.PassDataframe()
dataFile1 = mergedDataFile.loc[mergedDataFile['zone'] == "1"]
dataFile2 = mergedDataFile.loc[mergedDataFile['zone'] == "2"]



FileZones = [dataFile1, dataFile2]
listfeatures_distributions = [["mass","exponential"],["velocity","normal"],["TimebeforeStone","exponential"]]
#listfeatures_distributions = [["mass", "exponential"]]
sizeMonteCarloSim = 1_000_000

listfeatures_samples = pd.DataFrame()
zoneindex = 1

# Mainloop for zone calculations
for FileZone in FileZones:
    # calc fit dist features (for )
    for featureDistribution in listfeatures_distributions:
        # calc fit dist features case when dist predefined
        if (featureDistribution[1] == "exponential"):
            explambda = mean(FileZone[featureDistribution[0]])
            # generate sample
            sample = exponential(explambda, sizeMonteCarloSim)
            listfeatures_samples[featureDistribution[0]+ '_zone_{}'.format(zoneindex)] = sample
            featureDistribution= np.append(featureDistribution,sample)
        elif(featureDistribution[1] == "normal"):
            meanTruncated = mean(FileZone[featureDistribution[0]])
            stdTruncated = np.std(FileZone[featureDistribution[0]])
            # generate sample
            sample = normal(meanTruncated,stdTruncated, size=sizeMonteCarloSim)
            listfeatures_samples[featureDistribution[0]+ '_zone_{}'.format(zoneindex)] = sample

    zoneindex=zoneindex+1
print(listfeatures_samples)
print(listfeatures_samples.min())
print(listfeatures_samples.max())


print(listfeatures_samples['velocity_zone_1'].lt(0).sum())
    # calc monte carlo
    # return calc monte carlo
