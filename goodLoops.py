# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:26:32 2021

@author: schue
"""

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

# FindmostfittingDistribution(dataFile2["TimebeforeStone"])

FileZones = [dataFile1, dataFile2]
listfeatures_distributions_zone1 = [["mass", "exponential"], [
    "velocity", "normal"], ["TimebeforeStone", "exponential"]]
listfeatures_distributions_zone2 = [["mass", "exponential"], [
    "velocity", "normal"], ["TimebeforeStone", "exponential"]]

# Anzahl Durchgänge in der Monte Carlo Simulation
#sizeMonteCarloSim = 1_000_000

sizeMonteCarloSim = 1000000

listfeatures_samples = pd.DataFrame()
zoneindex = 1

# Mainloop for zone calculations
# for FileZone in FileZones:
#     # calc fit dist features (for )
#     for featureDistribution in listfeatures_distributions:
#         # calc fit dist features case when dist predefined
#         if (featureDistribution[1] == "exponential"):
#             explambda = mean(FileZone[featureDistribution[0]])
#             # generate sample
#             sample = exponential(explambda, sizeMonteCarloSim)
#             listfeatures_samples[featureDistribution[0] +
#                                  '_zone_{}'.format(zoneindex)] = sample
#             featureDistribution = np.append(featureDistribution, sample)
#         elif(featureDistribution[1] == "normal"):
#             meanTruncated = mean(FileZone[featureDistribution[0]])
#             stdTruncated = np.std(FileZone[featureDistribution[0]])
#             # generate sample
#             sample = normal(meanTruncated, stdTruncated,
#                             size=sizeMonteCarloSim)
#             listfeatures_samples[featureDistribution[0] +
#                                  '_zone_{}'.format(zoneindex)] = sample
#     print("Zone fertig")
#     zoneindex = zoneindex+1

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


# Berechnung der Wahrscheinlichkeit, dass das Netz durchbrochen wird
#print(listfeatures_samples['ist ganz böse'].gt(0).sum() / sizeMonteCarloSim)

# Berechnung Masse im Netz
# # Vorgehensweise funktioniert nicht mehr ab 100/- Steinen, da pd Zeitreihe nicht lang genug

listfeatures_samples_zone_1["HoursbeforeStone"] = pd.to_timedelta(listfeatures_samples_zone_1["TimebeforeStone"], unit ="s")
listfeatures_samples_zone_1["CumsumHoursbeforeStone"] = listfeatures_samples_zone_1["HoursbeforeStone"].cumsum()
listfeatures_samples_zone_2["HoursbeforeStone"] = pd.to_timedelta(listfeatures_samples_zone_2["TimebeforeStone"], unit ="s")
listfeatures_samples_zone_2["CumsumHoursbeforeStone"] = listfeatures_samples_zone_2["HoursbeforeStone"].cumsum()

index = 0
for x in listfeatures_samples_zone_1["CumsumHoursbeforeStone"]:
    x = ((x.days * 24 + x.seconds /3600)) *3600
    listfeatures_samples_zone_1["CumsumHoursbeforeStone"][index] = x
    # print(x.days * 24 + x.seconds /3600)
    index = index+1
index = 0
for x in listfeatures_samples_zone_2["CumsumHoursbeforeStone"]:
    x = ((x.days * 24 + x.seconds /3600)) *3600
    listfeatures_samples_zone_2["CumsumHoursbeforeStone"][index] = x
    # print(x.days * 24 + x.seconds /3600)
    index = index+1


print(listfeatures_samples_zone_1)
print(listfeatures_samples_zone_2)

listfeatures_samples = listfeatures_samples_zone_1.merge(listfeatures_samples_zone_2,how="outer")
listfeatures_samples = listfeatures_samples.sort_values(by='CumsumHoursbeforeStone')
listfeatures_samples = listfeatures_samples.reset_index(drop=True)


listfeatures_samples.insert(8,'Year','')
print(listfeatures_samples)
year = 0
yearhours = 8760
index = 0
for listfeatures_sample in listfeatures_samples['CumsumHoursbeforeStone']:
    if(listfeatures_sample >  yearhours):
        year = year +1
        yearhours = yearhours +8760
        listfeatures_samples['Year'][index] = year
    else:

        listfeatures_samples['Year'][index] = year

    index = index+1

print(listfeatures_samples)

listfeatures_samples = listfeatures_samples.set_index("CumsumHoursbeforeStone")
listfeatures_samples["MasseinNet"] = listfeatures_samples["mass"].rolling("24h", min_periods=1).sum()

# # Berechnung ob Netz in Kombination im Masse darin reisst
# listfeatures_samples["BreachFullNet"] = np.where((listfeatures_samples["energy_zone_2"] >= 500) & (listfeatures_samples["MassinNet"] >= 2000), 1, 0)
# print(listfeatures_samples['BreachFullNet'].gt(0).sum() / sizeMonteCarloSim)





# To Do: Berechnung der Steine die wegen dem vollen Netz dieses durchschlagen haben (Daten vorhanden, Code noch offen),
#       Verknüpfung der Wahrscheinlichkeit, dass eine Auto getroffen wird und der Wahrscheinlichkeit, dass es dann zu einem Todesfall kommt.
