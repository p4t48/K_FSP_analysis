from FSP_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import time

#
# Analyse data
#

# Info about input data file
samplingRate = 10**5
bits = 16
channelLayout = {'pump': 1, 'probe': 2, 'waveform': 3, 'trigger': 4}
channelRanges = {'pump': 10, 'probe': 5, 'waveform': 10, 'trigger': 10}
amplifierGains = {'pump': 10**5, 'probe': 10**6, 'waveform': 1, 'trigger': 1}

dataFiles = glob.glob("../20171207/F21_1uT_4_Pump_1_Probe")
print(dataFiles)

for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains)
    #an.FSPFitExponential(1,1)
    #an.FSPFitExponentialPlot(1)
    print("Coarse frequency: %g " % an.FSPCoarseFrequency(1))
    start = time.time()
    an.FSPFitDecayingSine(1,1)

    #an.FSPFullFit(1,1)
    print(time.time() - start)
    #an.AnalyseNFSPs(4)
    #an.FSPNoiseLevelPlot(1, 1000, 4000, 10**6)
    #print(an.ReturnPumpProbeLevels(1))
