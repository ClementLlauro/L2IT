import numpy as np
import scipy as sp
from scipy.signal import tukey as tuk
import matplotlib.pyplot as plt
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity, rotate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import matplotlib.ticker as ticker

PsdPlotFlag = True
CovPlotFlag = True
TimeNoisePlotFlag = False

fmin = 1e-4
fmax = 8e-3
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

S_n[0] = 100*PowerSpectralDensity(freq[1])[0]
S_c[0] = PowerSpectralDensity(freq[1])[1]

S_n[1:] = PowerSpectralDensity(freq[1:])[0]
S_c[1:] = PowerSpectralDensity(freq[1:])[1]

raw_PSD = S_n + S_c
print(f'Instr_PSD(f=1e-5) = {PowerSpectralDensity(1e-5)[0]}')
print(f'Conf_PSD(f=1e-5) = {PowerSpectralDensity(1e-5)[1]}')

variance_instr_noise_f = N * S_n / (4 * delta_t)    # Calculate variance of instrumental noise, real and imaginary. (3.34) p.58
variance_conf_noise_f = N * S_c / (4 * delta_t)     # Calculate variance of confusion background noise, real and imaginary. (3.34) p.58

Instr_noise_matrix = []
Conf_noise_matrix = []
Non_Stat_matrix = []
np.random.seed(1234)                                # Set the seed
N_iter = 2000

#breakpoint()
for i  in tqdm(range (0,N_iter)):
    instr_noise_f = np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_instr_noise_f),N_f)
    conf_noise_f = np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_conf_noise_f),N_f)

    Instr_noise_matrix.append(instr_noise_f)
    Conf_noise_matrix.append(conf_noise_f)

    # instr_noise_t = np.fft.irfft(instr_noise_f,N) 
    # conf_noise_t = np.fft.irfft(conf_noise_f,N)
    
    instr_noise_t = np.fft.irfft(instr_noise_f) 
    conf_noise_t = np.fft.irfft(conf_noise_f)
    non_stat_conf_noise_t = conf_noise_t*Modulation(1,100,T,t)

    non_stat_noise_t = instr_noise_t + non_stat_conf_noise_t

    non_stat_noise_f = np.fft.rfft(non_stat_noise_t)
    Non_Stat_matrix.append(non_stat_noise_f)


print(f'Time noise length = {len(instr_noise_t)}')
print(f'Length of non-stationary noise in f domain = {len(non_stat_noise_f)}')
# --- Plot noise realisation in the time domain --- #
if TimeNoisePlotFlag == True :

    plt.plot(t,conf_noise_t)
    plt.plot(t,non_stat_conf_noise_t)
    #plt.plot(t,instr_noise_t)

    plt.show()

# --- Plot the covariance matrices --- #

if CovPlotFlag == True :

    #Instr_COV = np.cov(Instr_noise_matrix,rowvar=False)
    #Conf_COV = np.cov(Conf_noise_matrix,rowvar=False)

    Non_Stat_COV = np.cov(Non_Stat_matrix,rowvar=False)
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
    rotated = (np.log(abs(Non_Stat_COV))[::-1])
    im=ax.imshow(rotated,extent=[0,N_f,0,N_f],origin="lower")
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format(x*delta_f))
    ax.xaxis.set_major_formatter(ticks_x)
    ax.xaxis.tick_top()
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format((N_f- x)*delta_f))
    ax.yaxis.set_major_formatter(ticks_y)
    #ax.text(0.45*N,0.56*N,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
    plt.xticks(fontsize = 6,rotation = 45)
    plt.yticks(fontsize = 6,rotation = 45)

    ax.set_xlabel("Frequency [Hz]")
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('Frequency [Hz]')
    plt.title('Modulated Noise covariance matrix',y = -0.1,fontsize = 15)
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

