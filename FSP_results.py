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

    def Sensitivity(self, samplingRate, nPoints, bc, bs, gamma, shotNoise):

        gainFemto = 10**6
        Dt = 1/samplingRate
        T = Dt * nPoints

        amplitude = np.sqrt(bc**2 + bs**2)/gainFemto
        T2 = 1/gamma
        beta = Dt / T2
        z = np.exp(-beta)
        N = nPoints

        num = N**3 * (1 - z**2)**3 * (1 - z**(2*N))
        denom = 12 * (z**2 * (1 - z**(2*N))**2 - N**2 * z**(2*N) * (1 - z**2)**2)
        C =  num / denom

        sensitivity = np.sqrt(12 * C) / (2 * np.pi * amplitude/shotNoise * T**(3/2))
        
        Bsens = sensitivity/7/np.sqrt(500/70) # Divide by 7 as 1uT = 7kHz for Potassium

        return Bsens
 
    def SensitivityPlot(self):
        
        pumpLevels, probeLevels, sensitivities = [], [], []
        samplingRates, numberPoints, bcs, bss, gammas, shotNoises = [], [], [], [], [], []        
        for dataFile in self.files:
            a = dataFile.split("_")
            df = pd.read_csv(dataFile, sep="\t")
            pumpLevels.append(float(a[2]))
            probeLevels.append(float(a[4]))
            #pumpLevels.append(np.mean(df['pumpL (A)'])*10**6)
            #probeLevels.append(np.mean(df['probeL (A)'])*10**6)

            df.reindex(df.index.drop(1))
            samplingRate = np.mean(df['sampling rates (SPS)'])
            samplingRates.append(samplingRate)
            nPoints = np.floor(np.mean(df['points per FSP']))
            numberPoints.append(nPoints)
            bc = np.mean(df['bcos (V)'])
            bcs.append(bc)
            bs = np.mean(df['bsine (V)'])
            bss.append(bs)
            gamma = np.mean(df['gamma (Hz)'])
            gammas.append(gamma)
            shotNoise = np.mean(df['shotNoise (A/sqrt(Hz))'])
            shotNoises.append(shotNoise)

            sensitivity = self.Sensitivity(samplingRate, nPoints, bc, bs, gamma, shotNoise)
            sensitivities.append(sensitivity)


        sensitivities = np.multiply(10**6, np.array(sensitivities))
        print(sensitivities)

        # Define the grid
        yi = np.linspace(min(pumpLevels),max(pumpLevels),400)
        xi = np.linspace(min(probeLevels),max(probeLevels),400)
        sens = griddata(probeLevels, pumpLevels, sensitivities, xi, yi, interp='linear')

        print(sens)

        levels = np.linspace(1.0*10**(0), 1*10**(1), 30, endpoint=True)

        plt.contourf(xi, yi, sens, levels, cmap='cubehelix')
        plt.title('RF-FSP sensitivity at 1 $\mu$T and 57$^{\circ}$C', size=20)
        plt.ylabel(r'Pump power, $P_{\mathrm{Pump}}$ ($\mu$A)', size=20)
        plt.xlabel(r'Probe power, $P_{\mathrm{Probe}}$ ($\mu$A)', size=20)
        cbarx = plt.colorbar(format=ticker.FuncFormatter(self.fmt))
        cbarx.ax.tick_params(labelsize=16) 
        cbarx.ax.set_ylabel(r'Sensitivity, $\sigma_{\mathrm{CRLB}}$ ($\frac{fT}{\sqrt{Hz}}$)', size=20)
        plt.scatter(probeLevels, pumpLevels, marker='o', s=0.5)
        plt.tight_layout()
        plt.show()
        #matplotlib.rc('pdf', fonttype=42)
        #plt.savefig("sensitivity.pdf")
        plt.clf()        
