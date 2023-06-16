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

PsdPlotFlag = True
CovPlotFlag = True
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

Instr_noise_matrix = []
Conf_noise_matrix = []
Non_Stat_matrix = []

np.random.seed(1234)    


A = 1
B = 0.5
                            # Set the seed
N_iter = 5000 # Number of noise realisations 

#breakpoint()
for i  in tqdm(range (0,N_iter)):
    instr_noise_f = np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) # draw gaussian instr noise
    conf_noise_f = np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) # draw gaussian conf noise

    Instr_noise_matrix.append(instr_noise_f)
    Conf_noise_matrix.append(conf_noise_f)

    # instr_noise_t = np.fft.irfft(instr_noise_f,N) 
    # conf_noise_t = np.fft.irfft(conf_noise_f,N)
    
    instr_noise_t = np.fft.irfft(instr_noise_f) # Converting back to add the modulation
    conf_noise_t = np.fft.irfft(conf_noise_f)
    non_stat_conf_noise_t = conf_noise_t*Modulation(1,0.5,T,t) # Modulation parameters need to be explored !!

    non_stat_noise_t = instr_noise_t + non_stat_conf_noise_t # Computing the non-stationary noise in the time domain


    non_stat_noise_f = np.fft.rfft(non_stat_noise_t) # Go back to frequency domain

    Non_Stat_matrix.append(non_stat_noise_f) # Compute the final non-stationary noise covariance matrix
    

print(f'Time noise length = {len(instr_noise_t)}')
print(f'Length of non-stationary noise in f domain = {len(non_stat_noise_f)}')

# --- Plot noise realisation in the time domain --- #
if TimeNoisePlotFlag == True :

    #plt.plot(t,conf_noise_t)
    
    #plt.plot(t,instr_noise_t)

    plt.show()

# --- Plot the covariance matrices --- #

if CovPlotFlag == True :

    #Instr_COV = np.cov(Instr_noise_matrix,rowvar=False)
    #Conf_COV = np.cov(Conf_noise_matrix,rowvar=False)
    Stat_COV = np.cov(Instr_noise_matrix+Conf_noise_matrix,rowvar=False)
    Non_Stat_COV = np.cov(Non_Stat_matrix,rowvar=False)
    
    # ---- Diagonal ratio checking ---- #
    
    d_s = np.diag(abs(Stat_COV))
    d_ns = np.diag(abs(Non_Stat_COV))

    stat_diag = (N/delta_t)*(1/4)*(PowerSpectralDensity(freq)[0] + PowerSpectralDensity(freq)[1])
    non_stat_diag = (N/delta_t)*((1/2)*PowerSpectralDensity(freq)[0] + (A*A/2)*PowerSpectralDensity(freq)[1] + ((B*B)/8)*PowerSpectralDensity(freq+delta_f)[1] + ((B*B)/8)*PowerSpectralDensity(freq-delta_f)[1])

    
    plt.figure(figsize=(4,4))
    plt.plot(freq[1:],(d_s/stat_diag)[1:])
    plt.xlabel('Frequency [Hz]', fontsize = 15)
    plt.ylabel('e_diag/t_diag', fontsize = 15)
    plt.ylim((0.5,1.5))
    plt.title('Ratio of stationary noise covariance matrix diagonals \n (estimated/theoretical)',y=1.05,fontsize = 15, fontweight='bold')
    plt.show()
    
    plt.figure(figsize=(4,4))
    plt.plot(freq[5:-5],(d_ns/non_stat_diag)[5:-5])
    plt.xlabel('Frequency [Hz]', fontsize = 15)
    plt.ylabel('e_diag/t_diag', fontsize = 15)
    plt.ylim((0.5,1.5))
    plt.title('Ratio of non-stationary noise covariance matrix diagonals \n (estimated/theoretical)',y=1.05,fontsize = 15, fontweight='bold')
    plt.show()
    
    '''
    fig1,axe1 = plt.subplots()
    plt.title('Instrumental noise covariance matrix')
    im1 = axe1.imshow(np.log(abs(Instr_COV)))
    #divider = make_axes_locatable(axe1)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    axe1.text(300,1100,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    #plt.colorbar(im1, cax=cax)
    fig1.show()

    fig2,axe2 = plt.subplots()
    plt.title('Confusion background noise covariance matrix')
    im2 = axe2.imshow(np.log(abs(Conf_COV)))
    #divider = make_axes_locatable(axe2)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    axe2.text(300,1100,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    #plt.colorbar(im2, cax=cax)
    fig2.show()

    fig3,axe3 = plt.subplots()
    plt.title('Instr +Conf noise covariance matrix')
    im3 = axe3.imshow(np.log(abs(Conf_COV+Instr_COV)))
    #divider = make_axes_locatable(axe2)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    axe3.text(300,1100,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    #plt.colorbar(im2, cax=cax)
    fig3.show()
    '''
    fig= plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    rotated = (np.log(abs(Stat_COV))[::-1])
    im=ax.imshow(rotated,extent=[0,N_f,0,N_f],origin="lower")
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format(x*delta_f))
    ax.xaxis.set_major_formatter(ticks_x)
    ax.xaxis.tick_top()
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format((N_f- x)*delta_f))
    ax.yaxis.set_major_formatter(ticks_y)
    #ax.text(0.45*N,0.56*N,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    plt.xticks(fontsize = 10,rotation = 45)
    plt.yticks(fontsize = 10,rotation = 45)
    ax.set_xlabel("Frequency [Hz]",fontsize = 15)
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Frequency [Hz]",fontsize = 15)
    plt.title(r"Estimated noise covariance matrix : $\log_{10}\left |\Sigma_{N}(f,f')\right |$",y = -0.1,fontsize = 15, fontweight='bold')
    plt.colorbar(im,fraction = 0.04)
    # inset axes....
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.imshow(rotated,extent=[0,N_f,0,N_f],origin="lower")
    # subregion of the original image
    x1, x2, y1, y2 = (N_f/10)-20,(N_f/10)+20,(9*N_f/10)-20,(9*N_f/10)+20
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()
# --- Plot the PSD --- #

#breakpoint()
if PsdPlotFlag == True :

    plt.plot(freq,S_n,'*',label = 'noise')
    plt.plot(freq,S_c,'*',label = 'confusion')
    #plt.plot(freq,raw_PSD)
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.yscale('log')
    plt.ylabel('Strain (m-1)')
    plt.title("PSD : Instrumental + Confusion background noises")
    plt.legend()
    plt.show()

