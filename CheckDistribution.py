# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:18:09 2021

@author: schue
"""

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html

import matplotlib
import numpy as np
from scipy.stats import expon 
import pandas as pd
from statistics import mean, median
from numpy.random import normal
from matplotlib import pyplot
from numpy.random.mtrand import exponential
from scipy import stats


def FindmostfittingDistribution(MatrixColumn):

	## List of available Distributions for fitting in scipy
	list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']


	results = []
	for i in list_of_dists:
		dist = getattr(stats, i)
		param = dist.fit(MatrixColumn)
		a = stats.kstest(MatrixColumn, i, args=param)
		results.append((i,a[0],a[1]))
    
    
	results.sort(key=lambda x:float(x[2]), reverse=True)
	for j in results:
    		print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))



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

#FindmostfittingDistribution(mergedDataFile["mass"])


#Monte Carlo Sim für Exponential Verteilung der Masse

# define the distribution
explambda = mean(mergedDataFile["mass"])

# generate monte carlo samples of differing size
size = 1000000
# generate sample

sample = exponential(explambda, size)

print(max(sample))
print(min(sample))
print(mean(sample))
print(median(sample))








#TODO: Monte Carlo Sim für Masse, Geschwindigkeit und Zeit modellieren pro Ablösungszone


