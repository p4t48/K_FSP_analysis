from FSP_analysis import *
import numpy as np
import matplotlib.pyplot as plt

#
# Analyse data
#

# Info about input data file
inputRange = 5
samplingRate = 10**5
bits = 16
pump = 1
probe = 2

dataFiles = glob.glob("../20171207/F21_1uT_4_Pump_1_Probe")
print(dataFiles)


for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, inputRange, samplingRate, bits, pump, probe)
    an.FSPFullFit(1,1)
    # an.AnalyseNFSPs(10)
    #an.FSPNoiseLevelPlot(1, 1000, 4000, 10**6)
    print(an.ReturnPumpProbeLevels(1))
