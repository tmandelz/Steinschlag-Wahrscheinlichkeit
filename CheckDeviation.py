# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:26:32 2021

@author: schue
"""

from scipy import stats
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def Test(simsize):
    # Globale Variablen
    # Anzahl Durchgänge in der Monte Carlo Simulation
    sizeMonteCarloSim = simsize
    sizeMonteCarloSimZone1 = int(sizeMonteCarloSim * 0.68)
    sizeMonteCarloSimZone2 = int(sizeMonteCarloSim * 0.32)
    # Grenzwert der Wahrscheinlichkeit
    ProbabilityLimit = 10 ** -4
    
    
    # Pandas Warnungen ausblenden für Ketten Aufrufe
    pd.options.mode.chained_assignment = None  # default='warn'
    
 # Funktion welche eine Verteilungart fitten kann
def FindmostfittingDistribution(ColumnName,MatrixColumn):

    # Unterdrückt Division durch 0 Warnungen
    with np.errstate(divide='ignore',invalid='ignore'):
        # Liste von Verteilungen welche gefittet werden können
        list_of_dists = ['cauchy','expon', 'logistic', 'norm','uniform','gamma']
    
        results = []
        # Kolmogorov-Smirnov Test für jede Verteilung um den Typ zu finden
        for i in list_of_dists:
            dist = getattr(stats, i)
            param = dist.fit(MatrixColumn)
            a = stats.kstest(MatrixColumn, i, args=param)
            results.append((i, a[0], a[1]))

        # sortieren der Resultate nach Höchstem PWert
        results.sort(key=lambda x: float(x[2]), reverse=True)
        # Plotten von Verteilung
        MatrixColumn.hist()
        plt.title(f"Histogram der Spalte: {ColumnName}")
        plt.xlabel(f"{ColumnName}")
        plt.ylabel("Dichte")
        plt.show()
        
        # Ausgabe der Resultate
        print(f"Die folgenden Verteilungen der Spalte {ColumnName} wurden geprüft und fitten am besten in absteigender Reihenfolge:")
        for j in results:
            print("{}: pvalue={}".format(j[0], j[1], j[2]))
        
        

    # Funktion um die CSV's zu lesen
    def ReadDataframe():
        # Lesen der ersten CSV Datei
        dataFile1 = pd.read_csv("out_1.csv")
        # Lesen der zweiten CSV Datei
        dataFile2 = pd.read_csv("out_2.csv")

        # Zonen werden als Spalte ergänzt
        dataFile1['zone']='1'
        dataFile2['zone']='2'

        # Zusammenführen der beiden Zonen Datenframes
        mergedDataFile = dataFile1.append(dataFile2)
        return mergedDataFile
        
    # Funktion zur Berechnung eines Zeitunterschieds und Ergänzung der jeder Stunde im Datenframe als Row
    def CalculateTimeDeltaHours(mergedDataFile):
        # Stunde dazwischen als Rows ergänzen
        TimebeforeStoneZone1 = mergedDataFile.loc[mergedDataFile["zone"] == "1","DateTime"].diff()
        TimebeforeStoneZone2 = mergedDataFile.loc[mergedDataFile["zone"] == "2","DateTime"].diff()
        
        # Zeitdelta ergänzen
        mergedDataFile.loc[mergedDataFile["zone"] == "1","TimebeforeStone"] = TimebeforeStoneZone1.astype('timedelta64[h]').fillna(0)     
        mergedDataFile.loc[mergedDataFile["zone"] == "2","TimebeforeStone"] = TimebeforeStoneZone2.astype('timedelta64[h]').fillna(0)     
        return mergedDataFile


    # Bereinigungen und Berechnungen der Datenframes
    def CalculateandUpdateColumns(mergedDataFile):
        
        #Energie mittels der Formel (KE=1/2mv^2)/1000 in m = kg und v = m/s -> Kj
        mergedDataFile['energy']=((mergedDataFile['mass']/2)*(mergedDataFile['velocity']**2) / 1000)

        # Auslöser Spalte ergänzen wenn ein Stein gefallen ist
        mergedDataFile["Trigger"] = np.where(mergedDataFile["zone"] == 0, 0, 1) 

        # Umformatierung zum Datums Datentyp
        mergedDataFile["DateTime"] = pd.to_datetime(mergedDataFile['date'] + ' ' + mergedDataFile['timestamp'])

        # Nach Datum ordnen
        mergedDataFile = mergedDataFile.sort_values(by=['DateTime'])
        mergedDataFile = mergedDataFile.reset_index(drop=True)
        # Verschieben der Steine um eine Stunde welche zur gleichen Stunde fallen
        mergedDataFile.loc[1, 'timestamp'] = "10:00"
        mergedDataFile.loc[44, 'timestamp'] = "13:00"
        mergedDataFile.loc[89, 'timestamp'] = "13:00"

        # Update der Datumsspalte
        mergedDataFile["DateTime"] = pd.to_datetime(mergedDataFile['date'] + ' ' + mergedDataFile['timestamp'])
        return mergedDataFile

    # Funktion zur Erstellung der Timeserie für die Plots
    def TimeSeriePlots(mergedDataFile):
            # Leere Stunden ergänzen
            dfTimeSerie = mergedDataFile
            dfTimeSerie = dfTimeSerie.set_index(mergedDataFile["DateTime"])
            dfTimeSerie = dfTimeSerie.resample('H').first().fillna(0)
            dfTimeSerie['DateTime'] = dfTimeSerie.index

            # Fehlende Tage und Stunden ergänzen
            dfTimeSerie["date"] = dfTimeSerie["DateTime"].dt.date
            dfTimeSerie["timestamp"] = dfTimeSerie["DateTime"].dt.time
            
            # berechnen der rollierenden Energie sowie masse über 24 Stunden
            dfTimeSerie["rollingEnergy24h"] = dfTimeSerie["energy"].rolling(24, min_periods=1).sum()
            dfTimeSerie["rollingmass24h"] = dfTimeSerie["mass"].rolling(24, min_periods=1).sum()


            # Berechnung ob Durchbrüche beobachtet wurden mittels Durchbruchkriterien
            dfTimeSerie["BreachEnergy"] = np.where(dfTimeSerie["energy"] >= 1000, 1, 0)
            dfTimeSerie["BreachFullNet"] = np.where((dfTimeSerie["energy"] >= 500) & (dfTimeSerie["rollingmass24h"] >= 2000), 1, 0)
            return dfTimeSerie   
        
        
    # Initieren von Datenframe
    listfeatures_samples = pd.DataFrame()
    # Lesen des CSV's
    mergedDataFile = ReadDataframe()
    # Bereinigungen und Berechnungen der Datenframes ausführen
    mergedDataFile = CalculateandUpdateColumns(mergedDataFile)
    # Berechnen der Zeitunterschiede
    mergedDataFile = CalculateTimeDeltaHours(mergedDataFile)
    # Bereinigungen und Berechnungen der Datenframes für Plots ausführen
    dfTimeSerie = TimeSeriePlots(mergedDataFile)

    # Splitten in Zone 1 und Zone 2
    dataFile1 = mergedDataFile.loc[mergedDataFile['zone'] == "1"]
    dataFile2 = mergedDataFile.loc[mergedDataFile['zone'] == "2"]
    
    # Definition der Verteilungsinformationen für die Zonen aus dem vorherigen Fitting
    listfeatures_distributions_zone1 = [["mass", "exponential"], ["velocity", "normal"], ["TimebeforeStone", "exponential"]]
    listfeatures_distributions_zone2 = [["mass", "exponential"], ["velocity", "normal"], ["TimebeforeStone", "gamma"]]
    
    # Mainloop for Monte Carlo Sim
    # Zone 1
    listfeatures_samples_zone_1 = pd.DataFrame()
    zoneindex = 0
    
    for featureDistribution in listfeatures_distributions_zone1:
        zoneindex = 1
        if (featureDistribution[1] == "exponential"):
            # Verteilungsparameter bestimmen
            meanExponential1 = mean(dataFile1[featureDistribution[0]])
            # Generieren von samples die exponential verteilt sind
            rng = default_rng()
            sample = rng.exponential(meanExponential1,size=sizeMonteCarloSimZone1)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_1[featureDistribution[0]] = sample
    
        elif(featureDistribution[1] == "normal"):
            # Verteilungsparameter bestimmen
            meanNormal1 = mean(dataFile1[featureDistribution[0]])
            stdNormal1 = np.std(dataFile1[featureDistribution[0]])
            # Generieren von samples die normal verteilt sind
            rng = default_rng()
            sample = rng.normal(meanNormal1, stdNormal1,size=sizeMonteCarloSimZone1)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_1[featureDistribution[0]] = sample
    
        elif(featureDistribution[1] == "gamma"):
            # Verteilungsparameter bestimmen
            meanGamma1 = mean(dataFile1[featureDistribution[0]])
            # Generieren von samples die gamma verteilt sind
            rng = default_rng()
            sample = rng.gamma(meanGamma1, size=sizeMonteCarloSimZone1)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_1[featureDistribution[0]] = sample
    
    # Zone 2
    listfeatures_samples_zone_2 = pd.DataFrame()
    zoneindex = 0
    
    for featureDistribution in listfeatures_distributions_zone2:
        zoneindex = 2
        if (featureDistribution[1] == "exponential"):
            # Verteilungsparameter bestimmen
            meanExponential2 = mean(dataFile2[featureDistribution[0]])
            # Generieren von samples die exponential verteilt sind
            rng = default_rng()
            sample = rng.exponential(meanExponential2, size=sizeMonteCarloSimZone2)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_2[featureDistribution[0]] = sample
    
        elif(featureDistribution[1] == "normal"):
            # Verteilungsparameter bestimmen
            meanNormal2 = mean(dataFile2[featureDistribution[0]])
            stdNormal2 = np.std(dataFile2[featureDistribution[0]])
            # Generieren von samples die normal verteilt sind
            rng = default_rng()
            sample = rng.normal(meanNormal2, stdNormal2,size=sizeMonteCarloSimZone2)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_2[featureDistribution[0]] = sample
    
        elif(featureDistribution[1] == "gamma"):
            # Verteilungsparameter bestimmen
            meanGamma2 = mean(dataFile2[featureDistribution[0]])
            # Generieren von samples die gamma verteilt sind
            rng = default_rng()
            sample = rng.gamma(meanGamma2, size=sizeMonteCarloSimZone2)
            # Generierte Werte ins Datenframe hinzufügen
            listfeatures_samples_zone_2[featureDistribution[0]] = sample
    
    # Berechnung der Energie der simulierten Steinschläge pro Zone
    listfeatures_samples_zone_1['energy'] = ((listfeatures_samples_zone_1['mass']/2)*(listfeatures_samples_zone_1['velocity']**2) / 1000)
    listfeatures_samples_zone_2['energy'] = ((listfeatures_samples_zone_2['mass']/2)*(listfeatures_samples_zone_2['velocity']**2) / 1000)
    
    # Markierung der Steine, die mit der Energie das Netz durchschlagen haben
    listfeatures_samples_zone_1['direct_breakthrough'] = np.where((listfeatures_samples_zone_1["energy"] >= 1000), 1, 0)
    listfeatures_samples_zone_2['direct_breakthrough'] = np.where((listfeatures_samples_zone_2["energy"] >= 1000), 1, 0)
    
    # Hinzufügen der Zone als Spalte
    listfeatures_samples_zone_1['Zone'] = 1
    listfeatures_samples_zone_2['Zone'] = 2
    
    # Kumulierte Summe der Zeitdifferenzen berechnen 
    listfeatures_samples_zone_1["CumsumHoursbeforeStone"] = listfeatures_samples_zone_1["TimebeforeStone"].cumsum()
    listfeatures_samples_zone_2["CumsumHoursbeforeStone"] = listfeatures_samples_zone_2["TimebeforeStone"].cumsum()
    
    # Kombinieren beider Sample Datenframes
    listfeatures_samples = listfeatures_samples_zone_1.append(listfeatures_samples_zone_2, ignore_index=True)
    
    # Sortieren und Index neu setzen
    listfeatures_samples = listfeatures_samples.sort_values(by='CumsumHoursbeforeStone')
    listfeatures_samples = listfeatures_samples.reset_index(drop=True)
    
    # Definition des Jahres in welcher der Stein gefallen ist (365d * 24h)
    listfeatures_samples["Year"] = listfeatures_samples['CumsumHoursbeforeStone'].floordiv(8760)
    
    
    # Berechnet, wieviel Masse pro Tag im Netz ist
    # Mittels floordivision den Tag, an dem der Stein heruntergefallen ist, bestimmen
    listfeatures_samples["Tag"] = listfeatures_samples['CumsumHoursbeforeStone'].floordiv(24) # // 24
    # Mittels groupby für jeden Tag die summierte Masse im Netz bererchnen
    Netzvoll = listfeatures_samples.groupby("Tag")["mass"].agg("sum")
    Netzvoll = pd.DataFrame({'Tag': Netzvoll.index, 'Tagesmasse': Netzvoll.values})
    # Mittels Merge für alle Steine, die an einem Tag mit vollem Netz heruntergefallen sind, dies markieren
    listfeatures_samples = listfeatures_samples.merge(Netzvoll, how="left", on="Tag")
    # Steine markieren, die genügend Energie hatten und bei denen am selben Tag das Netz voll war, markieren...
    listfeatures_samples["PossibleBreachFullNet"] = np.where((listfeatures_samples["energy"] >= 500) & (listfeatures_samples["Tagesmasse"] >= 2000), 1, 0)
    
    
    # ... auch wenn die Steine genügend Energie hatten, ist noch nicht bekannnt, ob zu ihrem Zeitpunkt an dem Tag das Netz voll war.
    # Dies muss nun separat überprüft werden.
    CountBreachFullNet = 0
    # Liste der Steine, die das Netz hätten durchschlagen könnne, erstellen
    ListPossibleBrechFullNet = listfeatures_samples[listfeatures_samples["PossibleBreachFullNet"] == 1]
    ListPossibleBrechFullNet = ListPossibleBrechFullNet.reset_index(drop = True)
    ListPossibleBrechFullNet = ListPossibleBrechFullNet.drop_duplicates(subset=['Tag'])
    ListPossibleBrechFullNet = ListPossibleBrechFullNet.reset_index(drop = True)
    # Durch diese Liste iterieren
    for i in range(len(ListPossibleBrechFullNet)):
        # Den betroffenen Tag selektieren
        Day = ListPossibleBrechFullNet.loc[i, "Tag"]
        # Liste mit alle Steinschlägen vom betroffenen Tag erstellen
        ToCheck = listfeatures_samples[listfeatures_samples["Tag"] == Day]
        ToCheck = ToCheck.reset_index()
        # Cumsum der Masse berechnen und diesen dem nächsten Stein hinzufügen (denn aus Aufgabestellung: "Falls bereits ein Stein mit...")
        ToCheck["CumsumMass"] = ToCheck["mass"].shift().cumsum()
        # Überprüfen ob beim Stein mit der genügend hohen Energie Cumsum bereits über der kritischen Masse war.
        for i in range(len(ToCheck)):
            if ToCheck.loc[i, "direct_breakthrough"] == 1:
                break  # break weil wenn das Netz durchbrochen ist, die Strasse gesperrt wird
            else:
                if (ToCheck.loc[i, "energy"] >= 500) & (ToCheck.loc[i, "CumsumMass"] >= 2000):
                    CountBreachFullNet += 1
                    # Annahme: Wird das Netz durchbrochen, wird die Strasse gesperrt. Der restliche Tag wird nicht mehr überprüft
                    break
    
    # Ausgabe der Simulationensresultate
    # print("Anzahl simulierter Jahre:", listfeatures_samples["Year"].max())
    # print("Anzahl Durchbrüche nach vollem Netz:", CountBreachFullNet)
    # print("Anzahl direkter Durchbrüche:",listfeatures_samples["direct_breakthrough"].sum())
    
    # Alle Durchbrüche berechnen
    AllBreaches = float(CountBreachFullNet + listfeatures_samples["direct_breakthrough"].sum())
    # Alle simulierten Jahre
    MaxYears = float(listfeatures_samples["Year"].max())
    
    # Wahrscheinlichkeit von einem Durchbruch pro Jahr
    ProbabilityNetBreach = AllBreaches / MaxYears
    # print('{0:.10f}'.format(ProbabilityNetBreach))
    
    # Autos pro Stunde
    ProbabilityCar = 1200/24
    # print('{0:.10f}'.format(ProbabilityCar))
    
    # Bremsweg definieren (Vollbremse = Bremsweg/2)
    BrakeWay = 36 / 2
    # Autolänge definieren
    AutoLength = 4.5
    
    # Gefahrenzone für ein Auto welches durchfährt
    DangerZone = BrakeWay + AutoLength
    # Geschwindigkeit eines Autos in m/s
    VelocityCar = 60 / 3.6
    
    # Wahrscheinlichkeit für ein Auto in der Gefahrenzone zu sein pro Tag
    ProbabilityCarImpact = (DangerZone / VelocityCar) / (60*60)
    # print('{0:.10f}'.format(ProbabilityCarImpact))
    
    # Wahrscheinlichkeit das ein Verunfallter stirbt
    ProbabilityDeath = float(4/14)
    # print('{0:.10f}'.format(ProbabilityDeath))
    
    # Durschnittspassagiere pro Fahrzeug
    MeanPassengers = float(1.66)
    # print('{0:.10f}'.format(MeanPassengers))
    
    #Berechnung Wahrscheinlichkeit für Todesfälle
    ProbabilityDeathCase = ProbabilityNetBreach  * ProbabilityCar * ProbabilityCarImpact * ProbabilityDeath * MeanPassengers
    
    print('{0:.10f}'.format(ProbabilityDeathCase))
    print('{0:.10f}'.format(ProbabilityLimit))
    
    # Ausgabe der Entscheidung
    if AllBreaches == 0:
        print("Keine Durchbrüche vorhanden. Die Simulationanzahl muss erhöht werden.")
    
    if ProbabilityDeathCase >= ProbabilityLimit :
        print(f"Wahrscheinlichkeitsgrenzwert {ProbabilityLimit} überschritten.")
        
    else:
        print(f"Wahrscheinlichkeitsgrenzwert {ProbabilityLimit} nicht überschritten. Die Strasse kann offen bleiben!")
    return '{0:.10f}'.format(ProbabilityDeathCase)


def Check(iterations, simsize):
    resultat = []
    for i in range(iterations):
        resultat.append(Test(simsize))
    print(resultat)

# Check(5, 100000)