import numpy as np
import scipy as sp
from scipy.signal import tukey as tuk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import matplotlib.ticker as ticker

import sys
sys.path.insert(1,'C:\\Users\\utilisateur\\Desktop\\Clément\\L2IT\\Python\\Code\\non_stationarity\\tools')
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity, rotate

# conda activate C:\Users\utilisateur\miniconda3\envs\mcmc_tutorial
# --- Flag section --- #

PsdPlotFlag = False
CovPlotFlag = False
TimeNoisePlotFlag = False

# --- Main --- #

fmin = 2e-4
fmax = 5e-3
N = 8192


T = N/(2*fmax)
print(f'T = {T}')

delta_f = 1/T
print(f'delta_f = {delta_f}')
delta_t = T/N
print(f'delta_t = {delta_t}')


t = np.arange(0,T,delta_t)
print(f'time array length = {len(t)}')

freq = np.arange(0,fmax + delta_f,delta_f)
print(f'freq array length = {len(freq)}')

N_f = len(freq)

S_n = np.empty(N_f)
S_c = np.empty(N_f)
# PSD not defined at f = 0 Hz
S_n[0] = 100*PowerSpectralDensity(freq[1])[0] # arbitrary value choosen to conserve the behaviour of instrumental PSD
S_c[0] = PowerSpectralDensity(freq[1])[1]

S_n[1:] = PowerSpectralDensity(freq[1:])[0]
S_c[1:] = PowerSpectralDensity(freq[1:])[1]

raw_PSD = S_n + S_c

#print(f'Instr_PSD(f=1e-5) = {PowerSpectralDensity(1e-5)[0]}')
#print(f'Conf_PSD(f=1e-5) = {PowerSpectralDensity(1e-5)[1]}')

variance_instr_noise_f = N * S_n / (4 * delta_t)    # Calculate variance of instrumental noise, real and imaginary. (3.34) p.58
variance_conf_noise_f = N * S_c / (4 * delta_t)     # Calculate variance of confusion background noise, real and imaginary. (3.34) p.58

Stat_matrix = []
Non_Stat_matrix = []

np.random.seed(1234)    # Set the seed


A = 1
B = 0.5
nodiag_mean_array = []
nodiag_median = []
N_iter = 20000       

#breakpoint()
for i  in tqdm(range (0,N_iter)):
    instr_noise_f = np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) # draw gaussian instr noise
    conf_noise_f = np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) # draw gaussian conf noise
    #Stat_matrix.append(instr_noise_f+conf_noise_f)
        
        
    #Instr_noise_matrix.append(instr_noise_f)
    #Conf_noise_matrix.append(conf_noise_f)

    # instr_noise_t = np.fft.irfft(instr_noise_f,N) 
    # conf_noise_t = np.fft.irfft(conf_noise_f,N)
        
    instr_noise_t = np.fft.irfft(instr_noise_f) # Converting back to add the modulation
    conf_noise_t = np.fft.irfft(conf_noise_f)
    non_stat_conf_noise_t = conf_noise_t*Modulation(1,0.5,T,t) # Modulation parameters need to be explored !!

    non_stat_noise_t = instr_noise_t + non_stat_conf_noise_t # Computing the non-stationary noise in the time domain


    non_stat_noise_f = np.fft.rfft(non_stat_noise_t) # Go back to frequency domain

    Non_Stat_matrix.append(non_stat_noise_f) # Compute the final non-stationary noise covariance matrix
        

    

#Stat_COV = np.cov(Stat_matrix,rowvar=False)
Non_Stat_COV = np.cov(Non_Stat_matrix,rowvar=False)


    
# ---- Main Diagonal ratio checking ---- #
    

#d_s = np.diag(abs(Stat_COV))
d_ns = np.diag(abs(Non_Stat_COV))

stat_diag = (N/delta_t)*(1/4)*(PowerSpectralDensity(freq)[0] + PowerSpectralDensity(freq)[1])
non_stat_diag = (N/delta_t)*((1/2)*PowerSpectralDensity(freq)[0] + (A*A/2)*PowerSpectralDensity(freq)[1] + ((B*B)/8)*PowerSpectralDensity(freq+delta_f)[1] + ((B*B)/8)*PowerSpectralDensity(freq-delta_f)[1])

# ---- 1st upper Diagonal ratio checking ---- #

d_ns1 = np.diag(abs(Non_Stat_COV),1)
d_ns2 = np.diag(abs(Non_Stat_COV),2)

#ns_d1 = (N/(2*delta_t))*A*B*(PowerSpectralDensity(freq)[1])[1:]
ns_d1 = (N/(delta_t))*(PowerSpectralDensity(freq)[1])[1:]
ns_d2 = (N/(delta_t))*(PowerSpectralDensity(freq)[1])[2:]

plt.figure(figsize=(8,8))
plt.plot(freq[1:],d_ns1/ns_d1,label='1st off-diagonal ratio')
plt.plot(freq[2:],d_ns2/ns_d2,label='2nd off-diagonal ratio')
plt.xlabel('Frequency [Hz]', fontsize = 15)
plt.ylabel('e_diag/t_diag', fontsize = 15)
plt.ylim((-0.75,1.25))
plt.xlim((2e-4,3e-3))
plt.text(x=0.0031,y=1/32,s = r'$\frac{B^2}{8}$',fontsize=15,color='black')
plt.text(x=0.0031,y=1/4,s = r'$\frac{AB}{2}$',fontsize=15,color='black')
plt.title('Ratio of non-stationary noise covariance matrix off-diagonals \n (estimated/theoretical)',y=1.05,fontsize = 15, fontweight='bold')
plt.legend()
plt.show()


'''
plt.figure(figsize=(4,4))
plt.plot(freq[1:],(d_s/stat_diag)[1:])
plt.xlabel('Frequency [Hz]', fontsize = 15)
plt.ylabel('e_diag/t_diag', fontsize = 15)
plt.ylim((0.5,1.5))
plt.title('Ratio of stationary noise covariance matrix diagonals \n (estimated/theoretical)',y=1.05,fontsize = 15, fontweight='bold')
plt.show()

plt.figure(figsize=(4,4))
plt.plot(freq[2:-2],(d_ns/non_stat_diag)[2:-2])
plt.xlabel('Frequency [Hz]', fontsize = 15)
plt.ylabel('e_diag/t_diag', fontsize = 15)
plt.ylim((0,2))
plt.title('Ratio of non-stationary noise covariance up matrix diagonals \n (estimated/theoretical)',y=1.05,fontsize = 15, fontweight='bold')
plt.show()
''' 