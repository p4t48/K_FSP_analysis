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
channelLayout = {'pump': 2, 'probe': 1, 'waveform': 3, 'trigger': 4}
channelRanges = {'pump': 10, 'probe': 10, 'waveform': 10, 'trigger': 10}
amplifierGains = {'pump': 10**6, 'probe': 10**6, 'waveform': 1, 'trigger': 1}

dataFiles = glob.glob("../20180413/Cs_stream")
print(dataFiles)

for dataFile in dataFiles:

    print(dataFile)
    an = FSPAnalysis(dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains)

    n = an.NumberOfFSPs()
    #an.ReturnFSP(1)
    #an.FSPFitDecayingSine(1)
    an.AnalyseNFSPs(n)
