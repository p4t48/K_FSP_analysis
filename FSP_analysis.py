"""
This fitting library for data streamed with the d-tacq DAQ (http://www.d-tacq.com/).

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


class FSPAnalysis:

    def __init__(self, dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains):
        self.dataFile = dataFile
        self.samplingRate = samplingRate
        self.amplifierGains = amplifierGains

        # Depending on the data format, get signal in volts
        if bits == 16:
            
            self.data = np.fromfile(self.dataFile, dtype=np.int16)
            pumpNorm = np.float16(channelRanges['pump'] / 2**15) # To get voltages from 16 bit int
            probeNorm = np.float16(channelRanges['probe'] / 2**15)
            waveformNorm = np.float16(channelRanges['waveform'] / 2**15)
            triggerNorm = np.float16(channelRanges['trigger'] / 2**15)

            # Cast to half precision float to save space in memory
            self.pumpCh = np.float16(self.data[channelLayout['pump']-1::4] * pumpNorm)
            self.probeCh = np.float16(self.data[channelLayout['probe']-1::4] * probeNorm)
            self.waveformCh = np.float16(self.data[channelLayout['waveform']-1::4] * waveformNorm)
            self.triggerCh = np.float16(self.data[channelLayout['trigger']-1::4] * triggerNorm)
            self.data = [] # Free up memory since data won't be accessed again

        elif bits == 32:
            
            self.data = np.fromfile(self.dataFile, dtype=np.int32)
            pumpNorm = channelRanges['pump'] / 2**31 # To get voltages from 32 bit int
            probeNorm = channelRanges['probe'] / 2**31 
            waveformNorm = channelRanges['waveform'] / 2**31 
            triggerNorm = channelRanges['trigger'] / 2**31

            # Cast to single precision float to save space in memory
            self.pumpCh = np.float32(self.data[channelLayout['pump']-1::4] * pumpNorm)
            self.probeCh = np.float32(self.data[channelLayout['probe']-1::4] * probeNorm)
            self.waveformCh = np.float32(self.data[channelLayout['waveform']-1::4] * waveformNorm)
            self.triggerCh = np.float32(self.data[channelLayout['trigger']-1::4] * triggerNorm)
            self.data = [] # Free up memory since data won't be accessed again
            
        else:
            print("Needs to be either 16 or 32 bit!")

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
        """ 
        Get the boundaries of all FSPs from raw data by using the trigger channel. 
        """

        def consecutive(data, stepsize=1):
            """ 
            Group consecutive numbers in array as a sub array. 
            """
            return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        
        triggerLevel = 1
        # Check whether the first FSP is complete or not.
        if self.triggerCh[0] < triggerLevel:
            initialState = 0
        else:
            initialState = 1

        # Numpy where returns a tuple as default but I want an array...    
        rangeFSP = np.squeeze(np.array(np.where(self.triggerCh <= triggerLevel)))
        locationFSP = consecutive(rangeFSP)

        if initialState == 0:
            # Skip the first incomplete FSP
            locationFSP = locationFSP[1:] 
        else:
            pass

        triggerPoints = []
        for arr in locationFSP:
            arrTrig = np.delete(arr, np.r_[slice(1,arr.size-1)])
            triggerPoints.append(arrTrig)

        # Return the boundary indices of all FSP signals except the last one
        return np.array(triggerPoints[:-1])

    def ReturnFSP(self, N):
        """ 
        Returns data points of the Nth FSP with timing info. 
        """

        boundaries = self.allTriggers[N]
        self.currentTrigger = boundaries

        # Get signal and cut out the first few weird points in the most obscure way possible
        signal = self.probeCh[boundaries[0]+50:boundaries[1]]
        time = np.arange(0, len(signal))/self.samplingRate

        self.currentFSP = {'time': time, 'signal': signal}

    def ReturnPump(self):
        """ 
        Returns data points of the pump beam during Nth FSP with timing info. 
        """

        boundaries = self.currentTrigger

        # Get signal and cut out the first few weird points in the most obscure way possible
        signal = self.pumpCh[boundaries[0]:boundaries[1]]
        time = np.arange(0, len(signal))/self.samplingRate

        return {'time': time, 'signal': signal}

    
    def ReturnPumpProbeLevels(self):
        """ 
        Returns the pump and probe levels of RF-FSP mode as averages during probing time. 
        """

        probeLevel = np.mean(self.currentFSP['signal']) / self.amplifierGains['probe']
        pumpLevel = np.mean(self.ReturnPump()['signal']) / self.amplifierGains['pump']

        return pumpLevel, probeLevel
        
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


    def FSPNoiseLevelPlot(self, noiseFreqLow, noiseFreqHigh):
        """ 
        This function will calculate the power spectral density (PSD) using the periodogram
        method. It will calculate the real noise level and the shot noise level of the FSP. Then
        it plots the Fourier spectra along with the different noise levels on the graph. 
        """

        electronCharge = 1.602 * 10**(-19) # In Coulomb

        # From a single FSP estimate the PSD
        data = self.currentFSP
        freq, powerDensity = sg.periodogram(data['signal'], self.samplingRate)
        fWelch, PSDWelch = sg.welch(data['signal'], self.samplingRate, nperseg=1024)

        # Use root PSD in units of A/sqrt(Hz) with gain set on Femto
        psd = np.sqrt(powerDensity)/self.amplifierGains['probe']
        psdWelch = np.sqrt(PSDWelch)/self.amplifierGains['probe']

        # Calculate the noise level of the signal according to a specified frequency range
        freqConditionLow = freq > noiseFreqLow
        freqConditionHigh = freq < noiseFreqHigh
        freqCondition = np.logical_and(freqConditionLow,freqConditionHigh)
        noiseRange = np.where(freqCondition)

        freqWConditionLow = fWelch > noiseFreqLow
        freqWConditionHigh = fWelch < noiseFreqHigh
        freqWCondition = np.logical_and(freqWConditionLow,freqWConditionHigh)
        noiseRangeWelch = np.where(freqWCondition)

        noiseLevelPSD = np.sqrt(np.mean(np.square(psd[noiseRange]))) 
        noiseLevelWelch = np.sqrt(np.mean(np.square(psdWelch[noiseRangeWelch]))) 
        shotNoiseLevel = np.sqrt(2*electronCharge*np.mean(data['signal'])/self.amplifierGains['probe'])

        print("The real noise level using the periodogram is %.3g A/sqrt(Hz)." % noiseLevelPSD)
        print("The real noise level using the Welch method is %.3g A/sqrt(Hz)." % noiseLevelWelch)
        print("The shot noise level is %.3g A/sqrt(Hz)." % shotNoiseLevel)

        plt.ylim([10**(-14), 10**(-5)])
        plt.ylabel("Root power spectral density, PSD ($A/\sqrt{Hz}$)", size=26)
        plt.xlabel("Frequency, f (Hz)", size=26)
        plt.semilogy(freq, psd, label="Periodogram")
        plt.semilogy(fWelch, PSDWelch, color='cyan', linewidth=2, label="Welch method")
        plt.plot((0, freq[-1]), (noiseLevelPSD, noiseLevelPSD), 'r-', linewidth=2, label="Noise level using the periodogram")
        plt.plot((0, freq[-1]), (noiseLevelWelch, noiseLevelWelch), 'k-.', linewidth=2, label="Noise level using the Welch method")
        plt.plot((0, freq[-1]), (shotNoiseLevel, shotNoiseLevel), 'k-', linewidth=2, label="Shot noise level")
        plt.plot((noiseFreqLow, noiseFreqLow), (10**(-14), 10**(-5)), 'k')
        plt.plot((noiseFreqHigh, noiseFreqHigh), (10**(-14), 10**(-5)), 'k')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()

        
    def FSPNoiseLevel(self, noiseFreqLow, noiseFreqHigh):
        """ 
        This function will calculate the power spectral density (PSD) using the periodogram
        method. It will calculate the real noise level and the shot noise level of the FSP and
        return those values. 
        """

        electronCharge = 1.602 * 10**(-19) # In Coulomb

        # From a single FSP estimate the PSD
        data = self.currentFSP
        freq, powerDensity = sg.periodogram(data['signal'], self.samplingRate)
        fWelch, PSDWelch = sg.welch(data['signal'], self.samplingRate, nperseg=1024)

        # Use root PSD in units of A/sqrt(Hz) with gain set on Femto
        psd = np.sqrt(powerDensity)/self.amplifierGains['probe']
        psdWelch = np.sqrt(PSDWelch)/self.amplifierGains['probe']

        # Calculate the noise level of the signal according to a specified frequency range
        freqConditionLow = freq > noiseFreqLow
        freqConditionHigh = freq < noiseFreqHigh
        freqCondition = np.logical_and(freqConditionLow,freqConditionHigh)
        noiseRange = np.where(freqCondition)

        noiseLevelPSD = np.sqrt(np.mean(np.square(psd[noiseRange]))) 
        noiseLevelWelch = np.sqrt(np.mean(np.square(psd[noiseRange]))) 
        shotNoiseLevel = np.sqrt(2*electronCharge*np.mean(data['signal'])/self.amplifierGains['probe'])

        return noiseLevelPSD, noiseLevelWelch, shotNoiseLevel


    def FSPFullFit(self, report=0):
        """ 
        Full fit of the FSP signal including the DC offset and its decay. 
        """

        previousResults = self.FSPFitDecayingSine()
        sineInitial = previousResults['sine']
        expInitial = previousResults['exp']
        data = previousResults['data']


        def FullFSP(params, t, data):
            """ Define model of a decaying sine wave with in phase and quadrature components plus DC offset and exponential decay of the offset. """

            bc = params['bc'].value
            bs = params['bs'].value
            f = params['f'].value
            gamma = params['gamma'].value
            dc = params['dc'].value
            d = params['d'].value
            e = params['e'].value

            model = (bc * np.cos(2*np.pi*f*t) + bs * np.sin(2*np.pi*f*t)) * np.exp(-gamma*t) + dc + d * np.exp(-e*t)

            return model - data

        # Parameters of the fit (using results from previous fits in initialParameters)
        fullParams = Parameters()
        fullParams.add('bc', value=sineInitial.params['bc'].value)
        fullParams.add('bs', value=sineInitial.params['bs'].value)
        fullParams.add('f', value=sineInitial.params['f'].value)
        fullParams.add('gamma', value=sineInitial.params['c'].value)
        fullParams.add('dc', value=0.84 ) #expInitial.params['a'].value)
        fullParams.add('d', value=14.5 ) #expInitial.params['d'].value)
        fullParams.add('e', value=0.001)

        fullResult = minimize(FullFSP, fullParams, args=(data['time'], data['signal']))
        
        # Only print the fit report and plot result if needed
        if (report == 1):
            report_fit(fullResult)
            fit = data['signal'] + fullResult.residual

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace':0})                        
            ax1.set_ylabel("Voltage, V (V)", size=20)
            ax2.set_ylabel("Residual (V)")
            ax1.plot(data['time'], data['signal'], 'bo', linestyle='-', markersize=2, label="Full FSP signal")
            ax1.plot(data['time'], fit, 'r-', label="Fit to the FSP")
            ax2.plot(data['time'], fullResult.residual, 'bo', linestyle='-', markersize=2)
            ax1.legend(loc=1)
            plt.xlabel("Time, t (s)", size=26)
            plt.tight_layout()
            plt.show()
        else:
            pass    

        return fullResult

    
    def AnalyseNFSPs(self, N):
        """ 
        Run the decaying sine wave routine for an FSP on N FSPs and save results in csv file. 
        """

        bcos, bsine, frequency, gamma = [], [], [], []
        pumpLevel, probeLevel, noisePeriod, noiseWelch, shotNoise = [], [], [], [], []
        samplingRate, pointsFSP, ampGain = [], [], []
       
        for i in range(N):
            self.ReturnFSP(i)
            print("Analysing FSP %i" % i)
            result = self.FSPFitDecayingSine()
            bcos.append(result['sine'].params['bc'].value)
            bsine.append(result['sine'].params['bs'].value)
            frequency.append(result['sine'].params['f'].value)
            gamma.append(result['sine'].params['c'].value)

            pumpL, probeL = self.ReturnPumpProbeLevels()
            pumpLevel.append(pumpL)
            probeLevel.append(probeL)
            nP, nW, sN = self.FSPNoiseLevel(3000, 4000)
            noisePeriod.append(nP)
            noiseWelch.append(nW)
            shotNoise.append(sN)
            samplingRate.append(self.samplingRate)
            pointsFSP.append(len(self.currentFSP['signal']))
            ampGain.append(self.amplifierGains['probe'])

        
        d = {'bcos (V)': bcos, 'bsine (V)': bsine, 'frequency (Hz)': frequency, 'gamma (Hz)': gamma, 'noiseP (A/sqrt(Hz))': noisePeriod, 'noiseW (A/sqrt(Hz))': noiseWelch, 'shotNoise (A/sqrt(Hz))': shotNoise, 'pumpL (A)': pumpLevel, 'probeL (A)': probeLevel, 'sampling rates (SPS)': samplingRate, 'points per FSP': pointsFSP, 'amplifier gain': ampGain}
        df = pd.DataFrame(data=d)

        folder = self.dataFile.split("/")[1]
        fileName = self.dataFile.split("/")[2]
        folderPath = "../results/%s" % folder
        filePath = "../results/%s/%s.csv" % (folder, fileName)

        if os.path.isdir(folderPath):
            df.to_csv(filePath, index=False, sep='\t')
        else:
            os.mkdir(folderPath)
            df.to_csv(filePath, index=False, sep='\t')

    def SNSensitivity(self):
        """ 
        Returns the shot noise sensitivity obtained with a single FSP signal with the formula frome W. Heils group (He3 paper). 
        """

        gainFemto = self.amplifierGains['probe']
        Dt = 1/self.samplingRate
        nPoints = len(self.currentFSP['signal'])
        T = Dt * nPoints

        result = self.FSPFitDecayingSine(N)
        amplitude = np.sqrt(result['sine'].params['bc'].value**2 + result['sine'].params['bs'].value**2)/self.amplifierGains['probe']
        T2 = 1/result['sine'].params['c'].value
        perNoise, welchNoise, shotNoise = self.FSPNoiseLevel(N, 3000, 4000)
        beta = Dt / T2
        z = np.exp(-beta)
        N = nPoints

        num = N**3 * (1 - z**2)**3 * (1 - z**(2*N))
        denom = 12 * (z**2 * (1 - z**(2*N))**2 - N**2 * z**(2*N) * (1 - z**2)**2)
        C =  num / denom

        sensitivity = np.sqrt(12 * C) / (2 * np.pi * amplitude/shotNoise * T**(3/2))

        Bsens = sensitivity/7 # Divide by 7 as 1uT = 7kHz for Potassium
        
        return Bsens
