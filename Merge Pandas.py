# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:08:49 2021

@author: schue
"""


from datetime import datetime
start_time = datetime.now()

import numpy as np
import pandas as pd
from statistics import mean
from numpy.random import normal
from numpy.random.mtrand import exponential
from scipy import stats
import SteinschlagGrafiken


def FindmostfittingDistribution(MatrixColumn):

    # List of available Distributions for fitting in scipy
    list_of_dists = ['beta', 'cauchy', 'chi', 'chi2',
                     'expon', 'logistic', 'norm', 'pareto', 't', 'uniform']

    results = []
    for i in list_of_dists:
        dist = getattr(stats, i)
        param = dist.fit(MatrixColumn)
        a = stats.kstest(MatrixColumn, i, args=param)
        results.append((i, a[0], a[1]))

    results.sort(key=lambda x: float(x[2]), reverse=True)
    for j in results:
        print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))


# Get prepared Dataframe from Steinschlaggrafik
# Der Vorteil des DFs gegenüber dataFilex ist, dass hier zusätzliche Infos wie Energie, TimebeforeStone, formatierte Zeitangaben etc vorhanden sind.
# Für die Verwendung müsste Mainloop for zone Calculations ein bisschen angepasst werden.
mergedDataFile = SteinschlagGrafiken.PassDataframe()
dataFile1 = mergedDataFile.loc[mergedDataFile['zone'] == "1"]
dataFile2 = mergedDataFile.loc[mergedDataFile['zone'] == "2"]

# FindmostfittingDistribution(dataFile2["TimebeforeStone"])

Time_Import = datetime.now()


FileZones = [dataFile1, dataFile2]
listfeatures_distributions_zone1 = [["mass", "exponential"], [
    "velocity", "normal"], ["TimebeforeStone", "exponential"]]
listfeatures_distributions_zone2 = [["mass", "exponential"], [
    "velocity", "normal"], ["TimebeforeStone", "exponential"]]

# Anzahl Durchgänge in der Monte Carlo Simulation

sizeMonteCarloSim = 10_000

listfeatures_samples = pd.DataFrame()

# Second Mainloop for zone calculations
# Zone 1
listfeatures_samples_zone_1 = pd.DataFrame()
zoneindex = 0

for featureDistribution in listfeatures_distributions_zone1:
    zoneindex = 1
    # calc fit dist features case when dist predefined
    if (featureDistribution[1] == "exponential"):
        explambda = mean(dataFile1[featureDistribution[0]])
        # generate sample
        sample = exponential(explambda, sizeMonteCarloSim)
        listfeatures_samples_zone_1[featureDistribution[0]] = sample
        featureDistribution_zone1 = np.append(featureDistribution, sample)
    elif(featureDistribution[1] == "normal"):
        meanTruncated = mean(dataFile1[featureDistribution[0]])
        stdTruncated = np.std(dataFile1[featureDistribution[0]])
        # generate sample
        sample = normal(meanTruncated, stdTruncated,
                        size=sizeMonteCarloSim)
        listfeatures_samples_zone_1[featureDistribution[0]]= sample

# Zone 2
listfeatures_samples_zone_2 = pd.DataFrame()
zoneindex = 0

for featureDistribution in listfeatures_distributions_zone2:
    zoneindex = 2
    # calc fit dist features case when dist predefined
    if (featureDistribution[1] == "exponential"):
        explambda = mean(dataFile1[featureDistribution[0]])
        # generate sample
        sample = exponential(explambda, sizeMonteCarloSim)
        listfeatures_samples_zone_2[featureDistribution[0]]= sample
        listfeatures_distributions_zone2 = np.append(featureDistribution, sample)
    elif(featureDistribution[1] == "normal"):
        meanTruncated = mean(dataFile1[featureDistribution[0]])
        stdTruncated = np.std(dataFile1[featureDistribution[0]])
        # generate sample
        sample = normal(meanTruncated, stdTruncated,
                        size=sizeMonteCarloSim)
        listfeatures_samples_zone_2[featureDistribution[0]] = sample

Timer_after_MonteCarlo = datetime.now()


# Berechnung der Energie der simulierten Steinschläge pro Zone
listfeatures_samples_zone_1['energy'] = (
    (listfeatures_samples_zone_1['mass']/2)*(listfeatures_samples_zone_1['velocity']**2) / 1000)
listfeatures_samples_zone_2['energy'] = (
    (listfeatures_samples_zone_2['mass']/2)*(listfeatures_samples_zone_2['velocity']**2) / 1000)
# # Markierung der Steine, die mit der Energie das Netz durchschlagen haben
listfeatures_samples_zone_1['direct_breakthrough'] = np.where(
    (listfeatures_samples_zone_1["energy"] >= 1000), 1, 0)
listfeatures_samples_zone_2['direct_breakthrough'] = np.where(
    (listfeatures_samples_zone_2["energy"] >= 1000), 1, 0)

listfeatures_samples_zone_1['Zone'] = 1
listfeatures_samples_zone_2['Zone'] = 2

Timer_after_Calc_Energy_directbreakthrough = datetime.now()


# Berechnung der Wahrscheinlichkeit, dass das Netz durchbrochen wird
#print(listfeatures_samples['ist ganz böse'].gt(0).sum() / sizeMonteCarloSim)

# Berechnung Masse im Netz

listfeatures_samples_zone_1["CumsumHoursbeforeStone"] = listfeatures_samples_zone_1["TimebeforeStone"].cumsum()
listfeatures_samples_zone_2["CumsumHoursbeforeStone"] = listfeatures_samples_zone_2["TimebeforeStone"].cumsum()

Timer_after_Cumsum1 = datetime.now()


Timer_after_Cumsum2 = datetime.now()






#listfeatures_samples_zone_1.set_index('CumsumHoursbeforeStone', inplace=True)
#listfeatures_samples_zone_2.set_index('CumsumHoursbeforeStone', inplace=True)

listfeatures_samples = listfeatures_samples_zone_1.append(listfeatures_samples_zone_2,ignore_index=True)

#listfeatures_samples = listfeatures_samples_zone_1.merge(listfeatures_samples_zone_2,how="outer")

Timer_after_Merge = datetime.now()

