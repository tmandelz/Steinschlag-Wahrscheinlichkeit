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

# Damit der Code läuft, müssen im gleichen Folder SteinschlagGrafiken.py und die beiden CSVs "out_1" und "out_2" vorhanden sein.

# Überprüfung welche Verteilungen zu den Faktoren passen (wurde nur einmal verwendet)
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


# Import der Daten läuft derzeit via mergedDataFile = SteinschlagGrafiken.PassDataframe() aus .py Code SteinschlagGrafiken
# Read first CSV File
# dataFile1 = pd.read_csv("out_1.csv")
# Read second CSV File
# dataFile2 = pd.read_csv("out_2.csv")

# Add Zone columns
# dataFile1['zone'] = '1'
# dataFile2['zone'] = '2'

# Get prepared Dataframe from Steinschlaggrafik
# Der Vorteil des DFs gegenüber dataFilex ist, dass hier zusätzliche Infos wie Energie, TimebeforeStone, formatierte Zeitangaben etc vorhanden sind.
# Für die Verwendung müsste Mainloop for zone Calculations ein bisschen angepasst werden.
mergedDataFile = SteinschlagGrafiken.PassDataframe()
dataFile1 = mergedDataFile.loc[mergedDataFile['zone'] == "1"]
dataFile2 = mergedDataFile.loc[mergedDataFile['zone'] == "2"]

# Erstellung von Listen für die Monte Carlo Simulation
FileZones = [dataFile1, dataFile2]
listfeatures_distributions = [["mass", "exponential"], [
    "velocity", "normal"], ["TimebeforeStone", "exponential"]]
#listfeatures_distributions = [["mass", "exponential"]]

# Anzahl Durchgänge in der Monte Carlo Simulation
sizeMonteCarloSim = 1_000

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
            listfeatures_samples[featureDistribution[0] +
                                 '_zone_{}'.format(zoneindex)] = sample
            featureDistribution = np.append(featureDistribution, sample)
        elif(featureDistribution[1] == "normal"):
            meanTruncated = mean(FileZone[featureDistribution[0]])
            stdTruncated = np.std(FileZone[featureDistribution[0]])
            # generate sample
            sample = normal(meanTruncated, stdTruncated,
                            size=sizeMonteCarloSim)
            listfeatures_samples[featureDistribution[0] +
                                 '_zone_{}'.format(zoneindex)] = sample
    print("Zone fertig")
    zoneindex = zoneindex+1


print(listfeatures_samples)
print(listfeatures_samples.min())
print(listfeatures_samples.max())

# Berechnung der Energie der simulierten Steinschläge pro Zone
listfeatures_samples['energy_zone_1'] = (
    (listfeatures_samples['mass_zone_1']/2)*(listfeatures_samples['velocity_zone_1']**2) / 1000)
listfeatures_samples['energy_zone_2'] = (
    (listfeatures_samples['mass_zone_2']/2)*(listfeatures_samples['velocity_zone_2']**2) / 1000)
# Markierung der Steine, die mit der Energie das Netz durchschlagen haben
listfeatures_samples['istböse_zone_1'] = np.where(
    (listfeatures_samples["energy_zone_1"] >= 1000), 1, 0)
listfeatures_samples['istböse_zone_2'] = np.where(
    (listfeatures_samples["energy_zone_2"] >= 1000), 1, 0)
listfeatures_samples['ist ganz böse'] = listfeatures_samples['istböse_zone_1'] + \
    listfeatures_samples['istböse_zone_2']

print(listfeatures_samples)
# Berechnung der Wahrscheinlichkeit, dass das Netz durchbrochen wird
print(listfeatures_samples['ist ganz böse'].gt(0).sum() / sizeMonteCarloSim)

# Berechnung Masse im Netz
# # Vorgehensweise funktioniert nicht mehr ab 100/- Steinen, da pd Zeitreihe nicht lang genug
listfeatures_samples["HoursbeforeStone_zone_1"] = pd.to_timedelta(listfeatures_samples["TimebeforeStone_zone_1"], unit ="s")
listfeatures_samples["CumsumHoursbeforeStone_zone_1"] = listfeatures_samples["HoursbeforeStone_zone_1"].cumsum()
listfeatures_samples["HoursbeforeStone_zone_2"] = pd.to_timedelta(listfeatures_samples["TimebeforeStone_zone_2"], unit ="s")
listfeatures_samples["CumsumHoursbeforeStone_zone_2"] = listfeatures_samples["HoursbeforeStone_zone_2"].cumsum()
listfeatures_samples = listfeatures_samples.set_index("CumsumHoursbeforeStone_zone_2")
listfeatures_samples["MassinNet"] = listfeatures_samples["mass_zone_2"].rolling("24h", min_periods=1).sum()

# # Berechnung ob Netz in Kombination im Masse darin reisst
# listfeatures_samples["BreachFullNet"] = np.where((listfeatures_samples["energy_zone_2"] >= 500) & (listfeatures_samples["MassinNet"] >= 2000), 1, 0)
# print(listfeatures_samples['BreachFullNet'].gt(0).sum() / sizeMonteCarloSim)


# To Do: Berechnung der Steine die wegen dem vollen Netz dieses durchschlagen haben (Daten vorhanden, Code noch offen),
#       Verknüpfung der Wahrscheinlichkeit, dass eine Auto getroffen wird und der Wahrscheinlichkeit, dass es dann zu einem Todesfall kommt.