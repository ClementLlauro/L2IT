import numpy as np

from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation,PowerSpectralDensity
from mcmc_fun import MCMC_run
import matplotlib.pyplot as plt
from corner import corner
from scipy.signal import tukey as tuk
import scipy as sp
np.random.seed(1234)


# --- Generate plot of the PSD for the given band range (set f<fmin to a constant) --- #

fmin = 1e-5
fmax = 0.1
N = 131072


T = N/(2*fmax)
delta_f = 1/T

freq = np.arange(0,fmax,delta_f)
N_f = len(freq)

S_n = np.empty(N_f)
S_c = np.empty(N_f)

N_plateau = len(freq[freq<fmin])

S_n[:N_plateau] = PowerSpectralDensity(freq[N_plateau])[0]
S_c[:N_plateau] = PowerSpectralDensity(freq[N_plateau])[1]

S_n[N_plateau:] = PowerSpectralDensity(freq[N_plateau:])[0]
S_c[N_plateau:] = PowerSpectralDensity(freq[N_plateau:])[1]

PSD = S_n + S_c


#plt.plot(freq,S_n)
#plt.plot(freq,S_c)
plt.plot(freq,PSD)
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.yscale('log')
plt.ylabel('Strain (m-1)')
plt.title("PSD : Instrumental + Confusion background noises (f<fmin -> constant)")

plt.show()

# --- Generate plot of the PSD for the given band range (use a gaussian handmade window to taper the PSD at f=0) --- #

fmin = 1e-5
fmax = 0.1
N = 32268


T = N/(2*fmax)
delta_f = 1/T

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
g_Sn = S_n*gaussian_taper
g_Sc = S_c*gaussian_taper
g_Snc = g_Sn + g_Sc
h_PSD = raw_PSD*hamming_taper
t_PSD = raw_PSD*tukey_taper

#plt.plot(freq,S_n)
#plt.plot(freq,S_c)
plt.plot(freq,b_PSD,label='Blackman')
plt.plot(freq,g_PSD,label='Gaussian')
plt.plot(freq,h_PSD,label='Hamming')
plt.plot(freq,t_PSD,label='Tukey')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.yscale('log')
plt.ylabel('Strain (m-1)')
plt.title(f"PSD : Instrumental + Confusion background noises (f<fmin -> tapered)")
plt.legend()

plt.show()





'''

M = []
for i  in range (1,4):
    L = [k*i + 1j*k*i for k in range (2,5)]
    M.append(L)
M = np.matrix(M)
print(M)

M_T = np.transpose(M)
print(M_T)

M_H = M_T.getH()

print(M_H)

COV = M_T*M_H

plt.matshow(COV.real)
plt.show()
'''