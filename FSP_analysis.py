"""
This fitting library contains the following methods:


- TriggerFSP(N): Get the Nth FSP from raw data by using the trigger channel.

- ReturnFSP(N): Returns data points of the Nth FSP with timing info.

- FSPFitExponential(N, report=0): Fits a constant plus exponential to the Nth FSP. As default doesn't print report. 

- FSPFitExponentialPlot(N): Plot result of fitting simple exponential to the FSP. 

-  FSPCoarseFrequency(N): Very coarse estimate of the frequency of the FSP for initial parameters of next fit. 

- FSPFitDecayingSine(N, report=0): Fit decaying sine with subtracted DC and exponential. 

- FSPNoiseLevelPlot(N, noiseFreqLow, noiseFreqHigh, gainFemto): This function will calculate the power spectral density (PSD) using the periodogram method. It will calculate the real noise level and the shot noise level of the FSP. Then it plots the Fourier spectra along with the different noise levels on the graph.

- FSPNoiseLevel(N, noiseFreqLow, noiseFreqHigh, gainFemto): This function will calculate the power spectral density (PSD) using the periodogram method. It will calculate the real noise level and the shot noise level of the FSP and return those values.

- FSPFullFit(N, report=0): Full fit of the FSP signal including the DC offset and its decay.

- AnalyseNFSPs(N): Run the full fitting routine for and FSP on N FSPs and save results in csv file. 

"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy import signal as sg
import pandas as pd
import glob


class FSPAnalysis:

    def __init__(self, dataFile, inputRange, samplingRate, bits, channel):
        self.dataFile = dataFile
        self.inputRange = inputRange
        self.samplingRate = samplingRate
        self.bits = bits
        self.channel = channel

        # Depending on the data format, get signal in volts
        f = open("%s" % self.dataFile, "r")
        if self.bits == 16:
            self.data = np.fromfile(f, dtype=np.int16)
            normalization = self.inputRange / 2**15 # To get voltages from 16 bit integer
        elif self.bits == 32:
            self.data = np.fromfile(f, dtype=np.int32)
            normalization = self.inputRange / 2**31 # To get voltages from 32 bit integer
        else:
            print("Needs to be either 16 or 32 bit!")

        self.trigger = self.data[3::4] * normalization
        self.rawFSP = self.data[self.channel-1::4] * normalization

        
    def TriggerFSP(self, N):
        """ Get the Nth FSP from raw data by using the trigger channel. """

        def consecutive(data, stepsize=1):
            """ Group consecutive numbers in array as a sub array. """
        
            return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        
        triggerLevel = 0.005
        # Check whether the first FSP is complete or not.
        if self.trigger[0] < triggerLevel:
            initialState = 0
        else:
            initialState = 1

        # Numpy where returns a tuple as default but I want an array...    
        rangeFSP = np.squeeze(np.array(np.where(self.trigger <= triggerLevel)))
        locationFSP = consecutive(rangeFSP)

        if initialState == 0:
            # Skip the first uncomplete FSP
            locationFSP = locationFSP[1:] 
        else:
            pass

        # Return the boundary indices of the Nth FSP
        return [min(locationFSP[N]), max(locationFSP[N])]



    def ReturnFSP(self, N):
        """ Returns data points of the Nth FSP with timing info. """

        boundaries = self.TriggerFSP(N)

        # Get signal and cut out the first few weird points in the most obscure way possible
        signal = self.rawFSP[boundaries[0]:boundaries[1]]
        time = np.arange(0, len(signal))/self.samplingRate

        return {'time': time, 'signal': signal}


    def FSPFitExponential(self, N, report=0):
        """ Fits a constant plus exponential to the Nth FSP. As default doesn't print report. """

        def ExponentialModel(params, t, data):
            """ Fitting model a + d * exp(-e * t). """

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

        data = self.ReturnFSP(N)
        expResult = minimize(ExponentialModel, expParams, args=(data['time'], data['signal']))

        # Only print the fit report if needed
        if report == 1:
            report_fit(expResult)
        else:
            pass

        return expResult

    
    def FSPFitExponentialPlot(self, N):
        """ Plot result of fitting simple exponential to the FSP. """

        expResult = self.FSPFitExponential(N, 1)
        data = self.ReturnFSP(N)
        fitResult = data['signal'] + expResult.residual

        plt.xlabel("Time, t (s)", size=26)
        plt.ylabel("Voltage, V (V)", size=26)

        plt.plot(data['time'], data['signal'], label='FSP data')
        plt.plot(data['time'], fitResult, 'r-', linewidth=3.0, label='Exponential fit to FSP')
        plt.legend(loc=1)
        plt.show()


    def FSPCoarseFrequency(self, N):
        """ Very coarse estimate of the frequency of the FSP for initial parameters of next fit. """

        # Remove the DC offset and the exponential decay for easier manipulation
        expResult = self.FSPFitExponential(N)
        a = expResult.params['a'].value
        d = expResult.params['d'].value
        e = expResult.params['e'].value

        data = self.ReturnFSP(N)
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


    def FSPFitDecayingSine(self, N, report=0):
        """ Fit decaying sine with subtracted DC and exponential. """

        def DecayingSine(params, t, data):
            """ Define model of a decaying sine wave with in phase and quadrature components. """

            bc = params['bc'].value
            bs = params['bs'].value
            f = params['f'].value
            c = params['c'].value
            h = params['h'].value

            model = (bc * np.cos(2*np.pi*f*t) + bs * np.sin(2*np.pi*f*t)) * np.exp(-c*t) + h

            return model - data

        # Remove the DC offset and the exponential decay for easier manipulation
        expResult = self.FSPFitExponential(N)
        a = expResult.params['a'].value
        d = expResult.params['d'].value
        e = expResult.params['e'].value

        data = self.ReturnFSP(N)
        time = data['time']
        signalSubtracted = data['signal'] - (a + d * np.exp(-e*time))

        # Parameters of the fit (using results from my first fit)
        sineParams = Parameters()
        sineParams.add('bc', value=0.1)
        sineParams.add('bs', value=0.1)
        sineParams.add('f', value=self.FSPCoarseFrequency(N))
        sineParams.add('c', value=70)
        sineParams.add('h', value=0.0)

        sineResult = minimize(DecayingSine, sineParams, args=(data['time'], signalSubtracted))

        # Only print the fit report and plot result if needed
        if (report == 1):
            report_fit(sineResult)
            fit = signalSubtracted + sineResult.residual

            plt.xlabel("Time, t (s)", size=26)
            plt.ylabel("Voltage, V (V)", size=26)
            plt.plot(time, signalSubtracted, label="FSP with subtracted DC and exponential")
            plt.plot(time, fit, 'r-', label="Fit to the FSP")
            plt.legend(loc=1)
            plt.show()
        else:
            pass

        return {'sine': sineResult, 'exp': expResult, 'data': data}


    def FSPNoiseLevelPlot(self, N, noiseFreqLow, noiseFreqHigh, gainFemto):
        """ This function will calculate the power spectral density (PSD) using the periodogram
        method. It will calculate the real noise level and the shot noise level of the FSP. Then
        it plots the Fourier spectra along with the different noise levels on the graph. """

        electronCharge = 1.602 * 10**(-19) # In Coulomb

        # From a single FSP estimate the PSD
        data = self.ReturnFSP(N)
        freq, powerDensity = sg.periodogram(data['signal'], self.samplingRate)
        fWelch, PSDWelch = sg.welch(data['signal'], self.samplingRate, nperseg=1024)

        # Use root PSD in units of A/sqrt(Hz) with gain set on Femto
        psd = np.sqrt(powerDensity)/gainFemto
        psdWelch = np.sqrt(PSDWelch)/gainFemto

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
        shotNoiseLevel = np.sqrt(2*electronCharge*np.mean(data['signal'])/gainFemto)

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

        
    def FSPNoiseLevel(self, N, noiseFreqLow, noiseFreqHigh, gainFemto):
        """ This function will calculate the power spectral density (PSD) using the periodogram
        method. It will calculate the real noise level and the shot noise level of the FSP and
        return those values. """

        electronCharge = 1.602 * 10**(-19) # In Coulomb

        # From a single FSP estimate the PSD
        data = self.ReturnFSP(N)
        freq, powerDensity = sg.periodogram(data['signal'], self.samplingRate)
        fWelch, PSDWelch = sg.welch(data['signal'], self.samplingRate, nperseg=1024)

        # Use root PSD in units of A/sqrt(Hz) with gain set on Femto
        psd = np.sqrt(powerDensity)/gainFemto
        psdWelch = np.sqrt(PSDWelch)/gainFemto

        # Calculate the noise level of the signal according to a specified frequency range
        freqConditionLow = freq > noiseFreqLow
        freqConditionHigh = freq < noiseFreqHigh
        freqCondition = np.logical_and(freqConditionLow,freqConditionHigh)
        noiseRange = np.where(freqCondition)

        noiseLevelPSD = np.sqrt(np.mean(np.square(psd[noiseRange]))) 
        noiseLevelWelch = np.sqrt(np.mean(np.square(psd[noiseRange]))) 
        shotNoiseLevel = np.sqrt(2*electronCharge*np.mean(data['signal'])/gainFemto)

        return noiseLevelPSD, noiseLevelWelch, shotNoiseLevel


    def FSPFullFit(self, N, report=0):
        """ Full fit of the FSP signal including the DC offset and its decay. """

        previousResults = self.FSPFitDecayingSine(N)
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
        fullParams.add('dc', value=expInitial.params['a'].value)
        fullParams.add('d', value=expInitial.params['d'].value)
        fullParams.add('e', value=10)

        fullResult = minimize(FullFSP, fullParams, args=(data['time'], data['signal']))

        # Only print the fit report and plot result if needed
        if (report == 1):
            report_fit(fullResult)
            fit = data['signal'] + fullResult.residual

            plt.xlabel("Time, t (s)", size=26)
            plt.ylabel("Voltage, V (V)", size=26)
            plt.plot(data['time'], data['signal'], label="Full FSP signal")
            plt.plot(data['time'], fit, 'r-', label="Fit to the FSP")
            plt.legend(loc=1)
            plt.show()
        else:
            pass    

        return fullResult

    
    def AnalyseNFSPs(self, N):
        """ Run the full fitting routine for and FSP on N FSPs and save results in csv file. """

        bcos, bsine, frequency, gamma, dc, d, e = [], [], [], [], [], [], []
        Dbcos, Dbsine, Dfrequency, Dgamma, Ddc, Dd, De = [], [], [], [], [], [], []

        for i in range(N):
            print("Analysing FSP %i" % i)
            result = self.FSPFullFit(i)
            bcos.append(result.params['bc'].value)
            Dbcos.append(result.params['bc'].stderr)
            bsine.append(result.params['bs'].value)
            Dbsine.append(result.params['bs'].stderr)
            frequency.append(result.params['f'].value)
            Dfrequency.append(result.params['f'].stderr)
            gamma.append(result.params['gamma'].value)
            Dgamma.append(result.params['gamma'].stderr)
            dc.append(result.params['dc'].value)
            Ddc.append(result.params['dc'].stderr)
            d.append(result.params['d'].value)
            Dd.append(result.params['d'].stderr)
            e.append(result.params['e'].value)
            De.append(result.params['e'].stderr)

        Bsens = self.Sensitivity(N, 3000, 5000, 10**6)
        
        d = {'bcos': bcos, 'Dbcos': Dbcos, 'bsine': bsine, 'Dbsine': Dbsine, 'frequency': frequency, 'Dfrequency': Dfrequency, 'gamma': gamma, 'Dgamma': Dgamma, 'dc': dc, 'Ddc': Ddc, 'd': d, 'Dd': Dd, 'e': e, 'De': De, 'SNBSens': Bsens}
        df = pd.DataFrame(data=d)

        df.to_csv("results/%s_ch%i.csv" % (self.dataFile, self.channel), index=False, sep='\t')


    def Sensitivity(self, N, noiseLevelLow, noiseLevelHigh, gainFemto):

        Dt = 1/self.samplingRate
        nPoints = len(self.ReturnFSP(N)['signal'])
        T = Dt * nPoints

        result = self.FSPFullFit(N)
        amplitude = np.sqrt(result.params['bc'].value**2 + result.params['bs'].value**2)/gainFemto
        T2 = 1/result.params['gamma'].value
        perNoise, welchNoise, shotNoise = self.FSPNoiseLevel(N, noiseLevelLow, noiseLevelHigh, gainFemto)
        beta = Dt / T2
        z = np.exp(-beta)
        N = nPoints

        num = N**3 * (1 - z**2)**3 * (1 - z**(2*N))
        denom = 12 * (z**2 * (1 - z**(2*N))**2 - N**2 * z**(2*N) * (1 - z**2)**2)
        C =  num / denom

        sensitivity = np.sqrt(12 * C) / (2 * np.pi * amplitude/shotNoise * T**(3/2))

        Bsens = sensitivity/7/np.sqrt(7)
        
        return Bsens