import numpy as np
import scipy as sp
from scipy.signal import tukey as tuk
import matplotlib.pyplot as plt
from tqdm import tqdm
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity

from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----- Fit parameters dictionnary gathering values for 6 months and one/two/four year(s) of observation. ----- #

fit_parameters = {"half_year": [0.133,243,482,917,0.00258], "one_year" : [0.171,292,1020,1680,0.00215],
                  "two_years" : [0.165,299,611,1340,0.00173], "four_years" : [0.138,-221,521,1680,0.00113]}

# --- Flags section --- #

PsdTaperedFlag = False
TimeNoiseFlag = False
CovPlotFlag = True
NoisePlotFlag = False
PsdPlotFlag = True

# ----- Main section ----- #

fmin = 1e-5
fmax = 1

delta_f = 1e-6

freq = np.arange(fmin,fmax,delta_f)
N_f = len(freq)

S_n = PowerSpectralDensity(freq)[0]
S_c = PowerSpectralDensity(freq)[1]

PSD = S_n + S_c

variance_instr_noise_f = S_n / (2 * delta_f)    # Calculate variance of instrumental noise, real and imaginary. 
variance_conf_noise_f = S_c / (2 * delta_f)     # Calculate variance of confusion background noise, real and imaginary.

Instr_noise_matrix = []
Conf_noise_matrix = []

np.random.seed(1234)                                # Set the seed

N_iter = 1
for i  in tqdm(range (0,N_iter)):


    instr_noise_f = np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f)
    conf_noise_f = np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f)

    Instr_noise_matrix.append(instr_noise_f)
    Conf_noise_matrix.append(conf_noise_f)


Instr_COV = np.cov(Instr_noise_matrix,rowvar=False)
Conf_COV = np.cov(Conf_noise_matrix,rowvar=False)

# Taking the diagonal terms of the COV matrix we should refind the PSD

'''
Instr_COV_diag = np.diag(Instr_COV)
Conf_COV_diag = np.diag(Conf_COV)


plt.plot(freq,Instr_COV_diag+Conf_COV_diag)
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
# --- Plot the covariance matrices --- #

if CovPlotFlag == True :

    fig1,axe1 = plt.subplots()
    plt.title('Instrumental noise covariance matrix')
    im1 = axe1.imshow(np.log(abs(Instr_COV)))
    divider = make_axes_locatable(axe1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axe1.text(300,1100,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    plt.colorbar(im1, cax=cax)
    fig1.show()

    fig2,axe2 = plt.subplots()
    plt.title('Confusion background noise covariance matrix')
    im2 = axe2.imshow(np.log(abs(Conf_COV)))
    divider = make_axes_locatable(axe2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axe2.text(300,1100,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    plt.colorbar(im2, cax=cax)
    fig2.show()

    plt.show(block=True)

    

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

#breakpoint()