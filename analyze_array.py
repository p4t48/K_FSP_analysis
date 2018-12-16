from FSP_analysis import *
import numpy as np
"""
This fitting library for data streamed with the d-tacq DAQ (http://www.d-tacq.com/).
Specialized to fitting data without recorded trigger signal.

The DAQ can stream data with the following format:

- 16 bit words with 16 bit vertical resolution on the sampling ADC.

- 32 bit words with 18 bit vertical resolution on the sampling ADC.

"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy import signal as sg
import pandas as pd
import glob
import os
import time


class ArrayAnalysis:

    def __init__(self, dataFile, samplingRate, bits, channelLayout, channelRange, amplifierGain):
        self.dataFile = dataFile
        self.samplingRate = samplingRate
        self.amplifierGain = amplifierGain
        self.lag = 7706 #3764 #6952 #
        self.FSPLength = 10000.7 # 5000.3542 # # 
        self.pulseLength = 50

        # Depending on the data format, get signal in volts
        if bits == 16:
            
            self.data = np.fromfile(self.dataFile, dtype=np.int16)
            channelNorm = channelRange / 2**15 # To get voltages from 16 bit int
            self.channel = self.data * channelNorm
            
        self.totalSamples = self.channel.size

        # Data used for the fitting of FSP signals
        self.allTriggers = self.TriggerFSP()
        self.currentTrigger = []
        self.currentFSP = []


    def NumberOfFSPs(self):
        """ 
        Writes to screen how many FSPs are in the present file. Returns this value. 
        """

        nFSPs = np.size(self.allTriggers,0)
        print("This file contains: %i FSPs" % nFSPs)

        return nFSPs
        
    def TriggerFSP(self):

        start = self.lag
        triggerPoints = []

        while(start < self.totalSamples):

            tStart = start + self.pulseLength
            tEnd = start + self.FSPLength - self.pulseLength

            triggerPoints.append([int(tStart), int(tEnd)])

            start += self.FSPLength

        return np.array(triggerPoints[:-1])

    
    def ReturnFSP(self, N):
        """ 
        Returns data points of the Nth FSP with timing info. 
        """

        boundaries = self.allTriggers[N]
        self.currentTrigger = boundaries

        # Get signal and cut out the first few weird points in the most obscure way possible
        signal = self.channel[boundaries[0]:boundaries[1]]
        time = np.arange(0, len(signal))/self.samplingRate

        self.currentFSP = {'time': time, 'signal': signal}    

        return self.currentFSP

    
    def FSPFitExponential(self, report=0):
        """ 
        Fits a constant plus exponential to the Nth FSP. As default doesn't print report. 
        """

        def ExponentialModel(params, t, data):
            """ 
            Fitting model a + d * exp(-e * t). 
            """

            a = params['a'].value
            d = params['d'].value
            e = params['e'].value

            model = a + d * np.exp(-e * t)

            return model - data

        # Parameters of the fit (using results from my first fit)
        expParams = Parameters()
        expParams.add('a', value=0.75)
        expParams.add('d', value=0.17)
        expParams.add('e', value=76)

        data = self.currentFSP
        expResult = minimize(ExponentialModel, expParams, args=(data['time'], data['signal']))

        # Only print the fit report if needed
        if report == 1:
            report_fit(expResult)
            fitResult = data['signal'] + expResult.residual

            plt.xlabel("Time, t (s)", size=26)
            plt.ylabel("Voltage, V (V)", size=26)

            plt.plot(data['time'], data['signal'], label='FSP data')
            plt.plot(data['time'], fitResult, 'r-', linewidth=3.0, label='Exponential fit to FSP')
            plt.legend(loc=1)
            plt.show()            
        else:
            pass

        return expResult

    def FSPCoarseFrequency(self):
        """ 
        Very coarse estimate of the frequency of the FSP for initial parameters of next fit. 
        """

        # Remove the DC offset and the exponential decay for easier manipulation
        expResult = self.FSPFitExponential()
        a = expResult.params['a'].value
        d = expResult.params['d'].value
        e = expResult.params['e'].value

        data = self.currentFSP
        time = data['time'][0:self.samplingRate//100]
        signal = data['signal'][0:self.samplingRate//100] - (a + d * np.exp(-e*time))

        # Find location where the FSP is positive to find the location of the zero crossings to positive.
        peaks = np.squeeze(np.array(np.where(signal > 0)))
        startPeriod = []

        for i in range(peaks.size - 1):
            if((peaks[i+1] - peaks[i]) > 1):
                startPeriod.append(peaks[i+1])

        freqEstimate = len(startPeriod)/time[-1]

        return freqEstimate    
    

    def FSPFitDecayingSine(self, report=0):
        """ 
        Fit decaying sine with subtracted DC and exponential. 
        """

        def DecayingSine(params, t, data):
            """ 
            Define model of a decaying sine wave with in phase and quadrature components. 
            """

            bc = params['bc'].value
            bs = params['bs'].value
            f = params['f'].value
            c = params['c'].value
            h = params['h'].value

            model = (bc * np.cos(2*np.pi*f*t) + bs * np.sin(2*np.pi*f*t)) * np.exp(-c*t) + h

            return model - data

        # Remove the DC offset and the exponential decay for easier manipulation
        expResult = self.FSPFitExponential()
        a = expResult.params['a'].value
        d = expResult.params['d'].value
        e = expResult.params['e'].value

        data = self.currentFSP
        time = data['time']
        signalSubtracted = data['signal'] - (a + d * np.exp(-e*time))

        # Parameters of the fit (using results from my first fit)
        sineParams = Parameters()
        sineParams.add('bc', value=0.1)
        sineParams.add('bs', value=0.1)
        sineParams.add('f', value=self.FSPCoarseFrequency())
        sineParams.add('c', value=70)
        sineParams.add('h', value=0.0)

        sineResult = minimize(DecayingSine, sineParams, args=(data['time'], signalSubtracted))

        # Only print the fit report and plot result if needed
        if (report == 1):
            report_fit(sineResult)
            fit = signalSubtracted + sineResult.residual

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace':0})                        
            ax1.plot(time, signalSubtracted, 'bo', linestyle='-', markersize=2, label="FSP with subtracted DC and exponential")
            ax1.plot(time, fit, 'r-', label="Fit to the FSP")
            ax1.set_ylabel("Voltage, V (V)", size=20)
            ax2.plot(time, sineResult.residual, 'bo', linestyle='-', markersize=2)
            ax2.set_ylabel("Residual (V)")
            ax1.legend(loc=0)
            plt.xlabel("Time, t (s)", size=26)
            plt.tight_layout()
            plt.show()
        else:
            pass

        return {'sine': sineResult, 'exp': expResult, 'data': data}
    

    def AnalyseNFSPs(self, N, resultsPath):
        """ 
        Run the decaying sine wave routine for an FSP on N FSPs and save results in csv file. 
        """

        bcos, bsine, frequency, gamma = [], [], [], []
        samplingRate, pointsFSP, ampGain = [], [], []
       
        for i in range(N):
            self.ReturnFSP(i)
            print("Analysing FSP %i" % i)
            result = self.FSPFitDecayingSine()
            bcos.append(result['sine'].params['bc'].value)
            bsine.append(result['sine'].params['bs'].value)
            frequency.append(result['sine'].params['f'].value)
            gamma.append(result['sine'].params['c'].value)

            samplingRate.append(self.samplingRate)
            pointsFSP.append(len(self.currentFSP['signal']))
            ampGain.append(self.amplifierGain)

        
        d = {'bcos (V)': bcos, 'bsine (V)': bsine, 'frequency (Hz)': frequency, 'gamma (Hz)': gamma, 'sampling rates (SPS)': samplingRate, 'points per FSP': pointsFSP, 'amplifier gain': ampGain}
        df = pd.DataFrame(data=d)

        folder = self.dataFile.split("/")[-2]
        fileName = self.dataFile.split("/")[-1]
        folderPath = "%s/%s" % (resultsPath, folder)
        filePath = "%s/%s/%s.csv" % (resultsPath, folder, fileName)

        if os.path.isdir(folderPath):
            df.to_csv(filePath, index=False, sep='\t')
        else:
            os.mkdir(folderPath)
            df.to_csv(filePath, index=False, sep='\t')    
