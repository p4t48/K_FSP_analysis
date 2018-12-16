from FSP_analysis import *
from analyze_array import *
import numpy as np
import matplotlib.pyplot as plt
import time

#
# Analyse data
#

# Info about input data file for four channel binary with trigger
"""
samplingRate = 10**6
bits = 16
channelLayout = {'pump': 1, 'probe': 2, 'waveform': 3, 'trigger': 4}
channelRanges = {'pump': 10, 'probe': 10, 'waveform': 10, 'trigger': 10}
amplifierGains = {'pump': 10**6, 'probe': 10**6, 'waveform': 1, 'trigger': 1}

dataFiles = glob.glob("../../Raw_data/2018/20181211/noise_only")
resultsPath = "../../Analysis_results"
print(dataFiles)

for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains)

    n = an.NumberOfFSPs()
    an.ReturnFSP(1)
    #an.FSPFitDecayingSine(1)
    #an.AnalyseNFSPs(n, resultsPath)
    an.FSPNoiseLevelPlot(10000,20000)
"""


# Info about input data file for one channel binary without trigger
samplingRate = 2*10**5
bits = 16 
channelLayout = 0
channelRange = 1.25
amplifierGain = 10**6

dataFile = glob.glob("../../Raw_data/2018/20181213/calib_chAlpha_-4.980")[0]
resultsPath = "../../Analysis_results"
print(dataFile)

an = ArrayAnalysis(dataFile, samplingRate, bits, channelLayout, channelRange, amplifierGain)
n = an.NumberOfFSPs()
#plotData = an.ReturnFSP(90000)
#an.FSPFitExponential(1)
#print(an.FSPCoarseFrequency())
#an.FSPFitDecayingSine(1)

an.AnalyseNFSPs(n, resultsPath)

#plt.plot(plotData['time'], plotData['signal'], 'bo-')
#plt.show()

