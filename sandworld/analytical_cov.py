import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(1,'C:\\Users\\utilisateur\\Desktop\\Clément\\L2IT\\Python\\Code\\non_stationarity\\tools')
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity, rotate

PsdPlotFlag = False

# --- Main --- #

A = 1
B = 1

fmin = 2e-5
fmax = 0.1

delta_f = 1e-5

freq = np.arange(fmin,fmax,delta_f)
N_f = len(freq)

S_n = PowerSpectralDensity(freq)[0]
S_c = PowerSpectralDensity(freq)[1]

PSD = S_n + S_c

#COV_t = np.zeros((N_f,N_f))
'''
diag_value = A*0.25*PSD[]
upper_diag_value = 
lower_diag_value = 
np.fill_diagonal(COV_t,)
'''

# --- Plot the PSD --- #

if PsdPlotFlag == True :

    plt.plot(freq,S_n,label="instrumental")
    plt.plot(freq,S_c,label="confusion")
    plt.plot(freq,PSD,'--',label="instrumental+confusion")
    plt.xscale('log')
    plt.ylim(1e-42)
    plt.xlabel('Frequency (Hz)')
    plt.yscale('log')
    plt.ylabel('PSD (s)')
    #plt.title("PSD : Instrumental + Confusion background noises")
    plt.legend()
    plt.show()

diag = A*0.25*(PowerSpectralDensity(freq-delta_f)[0] + PowerSpectralDensity(freq-delta_f)[1] + PowerSpectralDensity(freq+delta_f)[0] + PowerSpectralDensity(freq+delta_f)[1])

up_diag = A*0.25*(PowerSpectralDensity(freq-delta_f)[0] + PowerSpectralDensity(freq-delta_f)[1])[2:]

low_diag = A*0.25*(PowerSpectralDensity(freq+delta_f)[0] + PowerSpectralDensity(freq+delta_f)[1])[:-2]

m = np.diag(np.log(diag), 0) + np.diag(np.log(low_diag), -2) + np.diag(np.log(up_diag), 2)

plt.matshow(m)
plt.show()