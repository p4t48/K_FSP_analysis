from FSP_analysis import *
import numpy as np
import matplotlib.pyplot as plt

#
# Analyse data
#

# Info about input data file
inputRange = 5
samplingRate = 10**6
bits = 16
channel = 2

dataFiles = glob.glob("20171115/K_*_1MSPS")
print(dataFiles)


for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, inputRange, samplingRate, bits, channel)
    an.AnalyseNFSPs(20)
    #an.FSPNoiseLevelPlot(1, 1000, 4000, 10**6)

