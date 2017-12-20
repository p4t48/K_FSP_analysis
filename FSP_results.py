import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

class FSPResults:

    def __init__(self, resultFiles):
        self.files = glob.glob(resultFiles)
        print(self.files)


    def fmt(self, x, pos):
        a, b = '{:.0e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    def Sensitivity(self, gain, samplingRate, nPoints, bc, bs, gamma, shotNoise):
        """ Calculates the sensitivity of an FSP signal in nT/sqrt(Hz) according to Heil paper. """
        Dt = 1/samplingRate
        T = Dt * nPoints
        amplitude = np.sqrt(bc**2 + bs**2)/gain
        T2 = 1/gamma
        beta = Dt / T2
        z = np.exp(-beta)
        N = nPoints

        num = N**3 * (1 - z**2)**3 * (1 - z**(2*N))
        denom = 12 * (z**2 * (1 - z**(2*N))**2 - N**2 * z**(2*N) * (1 - z**2)**2)
        C =  num / denom

        sensitivity = np.sqrt(12 * C) / (2 * np.pi * amplitude/shotNoise * T**(3/2))        
        Bsens = sensitivity/(7*np.sqrt(0.5/T)) # Divide by 7 as 1nT = 7Hz for Potassium

        return Bsens
 
    def SensitivityPlot(self, save=0):
        """ Contour plot of sensitivities in fT/sqrt(Hz) as a function of laser powers. """
        
        pumpLevels, probeLevels, sensitivities = [], [], []
        for dataFile in self.files:
            df = pd.read_csv(dataFile, sep="\t")
            df.reindex(df.index.drop(1))            
            gain = np.mean(df['amplifier gain'])
            samplingRate = np.mean(df['sampling rates (SPS)'])
            nPoints = np.floor(np.mean(df['points per FSP']))
            bc = np.mean(df['bcos (V)'])
            bs = np.mean(df['bsine (V)'])
            gamma = np.mean(df['gamma (Hz)'])
            shotNoise = np.mean(df['shotNoise (A/sqrt(Hz))'])

            sensitivity = self.Sensitivity(gain, samplingRate, nPoints, bc, bs, gamma, shotNoise)
            sensitivities.append(sensitivity)
            pumpLevels.append(np.mean(df['pumpL (A)'])*10**6)
            probeLevels.append(np.mean(df['probeL (A)'])*10**6)

        sensitivities = np.multiply(10**6, np.array(sensitivities)) # The plot will show fT/sqrt(Hz)

        # Define the grid
        yi = np.linspace(min(pumpLevels),max(pumpLevels),400)
        xi = np.linspace(min(probeLevels),max(probeLevels),400)
        sens = griddata(probeLevels, pumpLevels, sensitivities, xi, yi, interp='linear')
        sensMin = min(sensitivities)
        sensMax = max(sensitivities)
        levels = np.linspace(sensMin, sensMax, 15, endpoint=True)

        plt.contourf(xi, yi, sens, levels, cmap='cubehelix')
        plt.title('RF-FSP sensitivity', size=20)
        plt.ylabel(r'Pump power, $P_{\mathrm{Pump}}$ ($\mu$A)', size=20)
        plt.xlabel(r'Probe power, $P_{\mathrm{Probe}}$ ($\mu$A)', size=20)
        cbarx = plt.colorbar(format=ticker.FuncFormatter(self.fmt))
        cbarx.ax.tick_params(labelsize=16) 
        cbarx.ax.set_ylabel(r'Sensitivity, $\sigma_{\mathrm{CRLB}}$ ($\frac{fT}{\sqrt{Hz}}$)', size=20)
        plt.scatter(probeLevels, pumpLevels, marker='o', s=0.5)
        plt.tight_layout()
        
        if save == 0:
            plt.show()
            plt.clf()
        else:
            matplotlib.rc('pdf', fonttype=42)
            plt.savefig("sensitivity.pdf")

    def GammaPlot(self, save=0):
        """ Contour plot of the linewidths in Hz as a function of laser powers. """
        
        pumpLevels, probeLevels, gammas = [], [], []
        for dataFile in self.files:
            df = pd.read_csv(dataFile, sep="\t")
            df.reindex(df.index.drop(1))            
            gain = np.mean(df['amplifier gain'])
            samplingRate = np.mean(df['sampling rates (SPS)'])
            nPoints = np.floor(np.mean(df['points per FSP']))
            bc = np.mean(df['bcos (V)'])
            bs = np.mean(df['bsine (V)'])
            gamma = np.mean(df['gamma (Hz)'])
            shotNoise = np.mean(df['shotNoise (A/sqrt(Hz))'])

            gammas.append(gamma)
            pumpLevels.append(np.mean(df['pumpL (A)'])*10**6)
            probeLevels.append(np.mean(df['probeL (A)'])*10**6)

        # Define the grid
        yi = np.linspace(min(pumpLevels),max(pumpLevels),400)
        xi = np.linspace(min(probeLevels),max(probeLevels),400)
        sens = griddata(probeLevels, pumpLevels, gammas, xi, yi, interp='linear')
        gammaMin = min(gammas)
        gammaMax = max(gammas)
        levels = np.linspace(gammaMin, gammaMax, 15, endpoint=True)

        plt.contourf(xi, yi, sens, levels, cmap='cubehelix')
        plt.title('RF-FSP linewidth', size=20)
        plt.ylabel(r'Pump power, $P_{\mathrm{Pump}}$ ($\mu$A)', size=20)
        plt.xlabel(r'Probe power, $P_{\mathrm{Probe}}$ ($\mu$A)', size=20)
        cbarx = plt.colorbar(format=ticker.FuncFormatter(self.fmt))
        cbarx.ax.tick_params(labelsize=16) 
        cbarx.ax.set_ylabel(r'Linewidth, $\Gamma_2$ (Hz)', size=20)
        plt.scatter(probeLevels, pumpLevels, marker='o', s=0.5)
        plt.tight_layout()
        
        if save == 0:
            plt.show()
            plt.clf()
        else:
            matplotlib.rc('pdf', fonttype=42)
            plt.savefig("linewidth.pdf")

    def AmplitudePlot(self, save=0):
        """ Contour plot of the amplitudes in V as a function of laser powers. """
        
        pumpLevels, probeLevels, amplitudes = [], [], []
        for dataFile in self.files:
            df = pd.read_csv(dataFile, sep="\t")
            df.reindex(df.index.drop(1))            
            gain = np.mean(df['amplifier gain'])
            samplingRate = np.mean(df['sampling rates (SPS)'])
            nPoints = np.floor(np.mean(df['points per FSP']))
            bc = np.mean(df['bcos (V)'])
            bs = np.mean(df['bsine (V)'])
            gamma = np.mean(df['gamma (Hz)'])
            shotNoise = np.mean(df['shotNoise (A/sqrt(Hz))'])

            amplitudes.append(np.sqrt(bc**2 + bs**2))
            pumpLevels.append(np.mean(df['pumpL (A)'])*10**6)
            probeLevels.append(np.mean(df['probeL (A)'])*10**6)

        # Define the grid
        yi = np.linspace(min(pumpLevels),max(pumpLevels),400)
        xi = np.linspace(min(probeLevels),max(probeLevels),400)
        sens = griddata(probeLevels, pumpLevels, amplitudes, xi, yi, interp='linear')
        amplitudeMin = min(amplitudes)
        amplitudeMax = max(amplitudes)
        levels = np.linspace(amplitudeMin, amplitudeMax, 15, endpoint=True)

        plt.contourf(xi, yi, sens, levels, cmap='cubehelix')
        plt.title('RF-FSP amplitudes', size=20)
        plt.ylabel(r'Pump power, $P_{\mathrm{Pump}}$ ($\mu$A)', size=20)
        plt.xlabel(r'Probe power, $P_{\mathrm{Probe}}$ ($\mu$A)', size=20)
        cbarx = plt.colorbar(format=ticker.FuncFormatter(self.fmt))
        cbarx.ax.tick_params(labelsize=16) 
        cbarx.ax.set_ylabel(r'Amplitude, A (V)', size=20)
        plt.scatter(probeLevels, pumpLevels, marker='o', s=0.5)
        plt.tight_layout()
        
        if save == 0:
            plt.show()
            plt.clf()
        else:
            matplotlib.rc('pdf', fonttype=42)
            plt.savefig("amplitude.pdf")            

