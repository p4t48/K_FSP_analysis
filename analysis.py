from FSP_analysis import *
import numpy as np
import matplotlib.pyplot as plt

#
# Analyse data
#

# Info about input data file
samplingRate = 10**5
bits = 16
channelLayout = {'pump': 1, 'probe': 2, 'waveform': 3, 'trigger': 4}
channelRanges = {'pump': 10, 'probe': 5, 'waveform': 10, 'trigger': 10}
amplifierGains = {'pump': 10**5, 'probe': 10**6, 'waveform': 1, 'trigger': 1}

dataFiles = glob.glob("../20171207/F21_1uT_*")
print(dataFiles)


for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains)
    #an.FSPFullFit(1,1)
    # an.AnalyseNFSPs(10)
    #an.FSPNoiseLevelPlot(1, 1000, 4000, 10**6)
    print(an.ReturnPumpProbeLevels(1))
