# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:18:09 2021

@author: schue
"""

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html

import numpy as np
from scipy.stats import expon 
import pandas as pd



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

#Sortieren um aufsteigende Übersicht der Energie zu haben
mergedDataFile = mergedDataFile.sort_values("energy")


mergedDataFile["pdf"] = expon.pdf(mergedDataFile["energy"], loc = 0, scale = 1)
mergedDataFile["cdf"] = expon.cdf(mergedDataFile["energy"], loc = 0, scale = 1) * 100

# print(expon.expect(expon.pdf, args=mergedDataFile["energy"]))

mergedDataFile["energy"].plot.hist(bins = 40)
mergedDataFile.plot(x = "energy", y = "pdf")

