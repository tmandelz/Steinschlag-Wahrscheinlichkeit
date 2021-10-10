# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:03:16 2021

@author: schue
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Read first CSV File
dataFile1 = pd.read_csv("out_1.csv")
##Read second CSV File
dataFile2 = pd.read_csv("out_2.csv")

#Add Zone columns
dataFile1['zone']='1'
dataFile2['zone']='2'

#Append File 2 to File 1
mergedDataFile = dataFile1.append(dataFile2)

#Add Energy in KJ column formula(KE=1/2mv^2)/1000 in m = kg and v = m/s
mergedDataFile['energy']=((mergedDataFile['mass']/2)*(mergedDataFile['velocity']**2) / 1000)

# Add Trigger if stone has fallen
mergedDataFile["Trigger"] = np.where(mergedDataFile["zone"] == 0, 0, 1) 

# Create Datetime Column
mergedDataFile["DateTime"] = pd.to_datetime(mergedDataFile['date'] + ' ' + mergedDataFile['timestamp'])

# Change Hour if two stones at same time
mergedDataFile = mergedDataFile.sort_values(by=['DateTime'])
mergedDataFile = mergedDataFile.reset_index(drop=True)
mergedDataFile.loc[1, 'timestamp'] = "10:00"
mergedDataFile.loc[44, 'timestamp'] = "13:00"
mergedDataFile.loc[89, 'timestamp'] = "13:00"

# Update Datetime Column
mergedDataFile["DateTime"] = pd.to_datetime(mergedDataFile['date'] + ' ' + mergedDataFile['timestamp'])


# Fill missing hours
dfTimeSerie = mergedDataFile
dfTimeSerie = dfTimeSerie.set_index(mergedDataFile["DateTime"])

dfTimeSerie = dfTimeSerie.resample('H').first().fillna(0)
dfTimeSerie['DateTime'] = dfTimeSerie.index

# Fill new missing days and hours in date and timestamp
dfTimeSerie["date"] = dfTimeSerie["DateTime"].dt.date
dfTimeSerie["timestamp"] = dfTimeSerie["DateTime"].dt.time

# Rolling 24h
dfTimeSerie["rollingEnergy24h"] = dfTimeSerie["energy"].rolling(24, min_periods=1).sum()
dfTimeSerie["rollingmass24h"] = dfTimeSerie["mass"].rolling(24, min_periods=1).sum()


# Calculate Breach
dfTimeSerie["BreachEnergy"] = np.where(dfTimeSerie["energy"] >= 1000, 1, 0)
dfTimeSerie["BreachFullNet"] = np.where((dfTimeSerie["energy"] >= 500) & (dfTimeSerie["rollingmass24h"] >= 2000), 1, 0)
# Theoretisch müsste man dfTimeSerie["rollingmass24h"] von der Reihe darüber nehmen, nun wird auch der neue Stein dazugezählt.

Uebersicht = mergedDataFile.describe()


dfTimeSerie["UebersichtEnergie"] = pd.cut(dfTimeSerie.energy, [0, 50, 150, 300, np.inf], labels=["Klein", "Mittel", "Gross", "Sehr gross"])
UebersichtEnergie = pd.pivot_table(dfTimeSerie, values='Trigger', columns=['UebersichtEnergie'], aggfunc=np.sum)
UebersichtZone = pd.pivot_table(dfTimeSerie, values='Trigger', columns=['zone'], aggfunc=np.sum)
UebersichtZoneEnergie = pd.pivot_table(dfTimeSerie, values='Trigger', columns=['zone', 'UebersichtEnergie'], aggfunc=np.sum)
## Achtung: ein Stein hat Energie 0 ##

# dfTimeSerie.boxplot(column = "mass", by = "zone")
# dfTimeSerie.boxplot(column = "velocity", by = "zone")
# dfTimeSerie.boxplot(column = "energy", by = "zone")
# dfTimeSerie.boxplot(column = "mass", by = "UebersichtEnergie")


# X-Achse: Uhrzeit 0:00 - 24:00
# ScatterplotTime = mergedDataFile.groupby(['timestamp']).agg({'Trigger':'sum'})
# plt.scatter(ScatterplotTime.index, ScatterplotTime['Trigger'])


# ScatterplotDate = dfTimeSerie.groupby(['date']).agg({'Trigger':'sum'})
# plt.scatter(ScatterplotDate.index, ScatterplotDate['Trigger'])

# mergedDataFile["energy"].plot.hist(bins = 40)
# dfTimeSerieHisto = dfTimeSerie[dfTimeSerie.rollingmass24h != 0]
# dfTimeSerieHisto["rollingmass24h"].plot.hist(bins = 40)


# dfTimeSerie.groupby(dfTimeSerie["date"]).sum()["energy"].plot()

# dfTimeSerie["rollingEnergy24h"].plot()
# dfTimeSerie["rollingmass24h"].plot()
# dfTimeSerie["BreachEnergy"].plot()
# dfTimeSerie["BreachFullNet"].plot()

# dfTimeSerie["rollingmass24h"].plot()
# dfTimeSerie["energy"].plot(secondary_y=True)


# print(dfTimeSerie)