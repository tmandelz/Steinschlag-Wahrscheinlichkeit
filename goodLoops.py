# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:26:32 2021

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

sizeMonteCarloSim = 1_000

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


listfeatures_samples = listfeatures_samples_zone_1.merge(listfeatures_samples_zone_2,how="outer")

Timer_after_Merge = datetime.now()

listfeatures_samples = listfeatures_samples.sort_values(by='CumsumHoursbeforeStone')
listfeatures_samples = listfeatures_samples.reset_index(drop=True)

Timer_after_Merge_clean = datetime.now()

listfeatures_samples["Year"] = listfeatures_samples['CumsumHoursbeforeStone'].floordiv(8760)


Timer_after_Calc_Year = datetime.now()

# Berechnet, wieviel Masse pro Tag im Netz ist, aber nicht, ob es wegen dem letzten gerissen ist...
listfeatures_samples["Tag"] = listfeatures_samples['CumsumHoursbeforeStone'] // 24
Netzvoll = listfeatures_samples.groupby("Tag")["mass"].agg("sum")
Netzvoll = pd.DataFrame({'Tag':Netzvoll.index, 'Tagesmasse':Netzvoll.values})
listfeatures_samples = listfeatures_samples.merge(Netzvoll, how="left", on = "Tag")
listfeatures_samples["PossibleBreachFullNet"] = np.where((listfeatures_samples["energy"] >= 500) & (listfeatures_samples["Tagesmasse"] >= 2000), 1, 0)



CountBreachFullNet = 0
ListPossibleBrechFullNet = listfeatures_samples[listfeatures_samples["PossibleBreachFullNet"] == 1]
ListPossibleBrechFullNet = ListPossibleBrechFullNet.reset_index()
for i in range(len(ListPossibleBrechFullNet)):
    Day = ListPossibleBrechFullNet.loc[i, "Tag"]
    ToCheck = listfeatures_samples[listfeatures_samples["Tag"] == Day]
    ToCheck = ToCheck.reset_index()
    ToCheck["CumsumMass"] = ToCheck["mass"].shift().cumsum()
    # print(ToCheck[["energy", "CumsumMass"]])
    for i in range(len(ToCheck)):
        if ToCheck.loc[i, "direct_breakthrough"] == 1:
            break #break weil wenn das Netz durchbrochen ist, die Strasse gesperrt wird
        else:
            if (ToCheck.loc[i, "energy"] >= 500) & (ToCheck.loc[i, "CumsumMass"] >= 2000):
                CountBreachFullNet += 1
                break
    # print(Day)


print("Number of years:", listfeatures_samples["Year"].max())
print("Number of BreachFullNet:", CountBreachFullNet)
print("Number of direct breakthrough:", listfeatures_samples["direct_breakthrough"].sum())


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("Time Import", Time_Import - start_time)
print("Time Monte Carlo", Timer_after_MonteCarlo - Time_Import)
print("Time Calc Energy direct break", Timer_after_Calc_Energy_directbreakthrough - Timer_after_MonteCarlo)
print("Time Cumsum1", Timer_after_Cumsum1 - Timer_after_Calc_Energy_directbreakthrough)
print("Time Cumsum2", Timer_after_Cumsum2 - Timer_after_Cumsum1)
print("Time Merge", Timer_after_Merge - Timer_after_Cumsum2)
print("Time Merge cleaned", Timer_after_Merge_clean - Timer_after_Merge)
print("Time Calc Year", Timer_after_Calc_Year - Timer_after_Merge_clean)
print("Time Calc Poss full Net", end_time - Timer_after_Calc_Year)



# # To Do: Berechnung der Steine die pro Jahr das Netz durchschlagen haben (für direct und Fullnet),
# #         Wirkliche Anzahl benötigte Jahre (10k oder 1mio)
# #       Verknüpfung der Wahrscheinlichkeit, dass eine Auto getroffen wird und der Wahrscheinlichkeit, dass es dann zu einem Todesfall kommt.
