import numpy as np
import scipy as sp
from scipy.signal import tukey as tuk
import matplotlib.pyplot as plt
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity

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

fmin = 1e-4
fmax = 1e-2
N = 65536

T = N/(2*fmax)
delta_f = 1/T
delta_t = T/N

t = np.arange(0,T,delta_t)
freq = np.arange(0,fmax,delta_f)

N_f = len(freq)

S_n = np.empty(N_f)
S_c = np.empty(N_f)

N_plateau = len(freq[freq<fmin])

S_n[:N_plateau] = PowerSpectralDensity(freq[N_plateau])[0]
S_c[:N_plateau] = PowerSpectralDensity(freq[N_plateau])[1]

S_n[N_plateau:] = PowerSpectralDensity(freq[N_plateau:])[0]
S_c[N_plateau:] = PowerSpectralDensity(freq[N_plateau:])[1]

raw_PSD = S_n + S_c

if PsdTaperedFlag == True :
    gaussian = sp.signal.gaussian(N_plateau*2+1,2)
    blackman = np.blackman(N_plateau*2+1)
    hamming = np.hamming(N_plateau*2+1)
    tukey = tuk(N_plateau*2+1,1)

    gaussian_taper = np.empty(N_f)
    blackman_taper = np.empty(N_f)
    hamming_taper = np.empty(N_f)
    tukey_taper = np.empty(N_f)

    gaussian_taper[0] = 0
    gaussian_taper[0:N_plateau] = gaussian[0:N_plateau]
    gaussian_taper[N_plateau:] = 1

    blackman_taper[0] = 0
    blackman_taper[0:N_plateau] = blackman[0:N_plateau]
    blackman_taper[N_plateau:] = 1

    hamming_taper[0] = 0
    hamming_taper[0:N_plateau] = hamming[0:N_plateau]
    hamming_taper[N_plateau:] = 1

    tukey_taper[0] = 0
    tukey_taper[0:N_plateau] = tukey[0:N_plateau]
    tukey_taper[N_plateau:] = 1


    b_PSD = raw_PSD*blackman_taper
    g_PSD = raw_PSD*gaussian_taper
    h_PSD = raw_PSD*hamming_taper
    t_PSD = raw_PSD*tukey_taper



variance_instr_noise_f = N * S_n / (4 * delta_t)    # Calculate variance of instrumental noise, real and imaginary. (3.34) p.58
variance_conf_noise_f = N * S_c / (4 * delta_t)     # Calculate variance of confusion background noise, real and imaginary. (3.34) p.58

Instr_noise_matrix = []
Conf_noise_matrix = []

np.random.seed(1234)                                # Set the seed

for i  in range (0,100):

    instr_noise_f = np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f)
    conf_noise_f = np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f)

    Instr_noise_matrix.append(instr_noise_f)
    Conf_noise_matrix.append(conf_noise_f)
    
    
if TimeNoiseFlag == True :

    instr_noise_t = np.fft.irfft(instr_noise_f) 
    conf_noise_t = np.fft.irfft(conf_noise_f)


Instr_noise_matrix_T = np.transpose(np.matrix(Instr_noise_matrix))
Conf_noise_matrix_T =np.transpose(np.matrix(Conf_noise_matrix))

#print(Instr_noise_matrix_T[0,1])

Instr_noise_matrix_H = Instr_noise_matrix_T.getH()
Conf_noise_matrix_H = Conf_noise_matrix_T.getH()

#print(Instr_noise_matrix_H[1,0])

Instr_COV = Instr_noise_matrix_T*Instr_noise_matrix_H
Conf_COV = Conf_noise_matrix_T*Conf_noise_matrix_H


# --- Plot the covariance matrices --- #

if CovPlotFlag == True :

    plt.matshow(Instr_COV.imag)
    plt.title('Instrumental noise covariance matrix')
    plt.colorbar()
    plt.show()

    plt.matshow(Conf_COV.imag)
    plt.title('Confusion background noise covariance matrix')
    plt.colorbar()
    plt.show()

# --- Plot noise realisation in time domain --- #

if NoisePlotFlag == True :

    plt.plot(t[:len(instr_noise_t)],instr_noise_t)
    plt.title('Instrumental noise')
    plt.xlabel('Time (s)')
    plt.show()

    plt.plot(t[:len(conf_noise_t)],conf_noise_t)
    plt.xlabel('Time (s)')
    plt.title('Confusion background noise')
    plt.show()


# --- Plot the PSD --- #

if PsdPlotFlag == True :

    #plt.plot(freq,S_n)
    #plt.plot(freq,S_c)
    plt.plot(freq,raw_PSD)
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.yscale('log')
    plt.ylabel('Strain (m-1)')
    plt.title("PSD : Instrumental + Confusion background noises")
    plt.show()


