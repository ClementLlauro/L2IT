import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

import sys
sys.path.insert(1,'C:\\Users\\utilisateur\\Desktop\\Clément\\L2IT\\Python\\Code\\non_stationarity\\tools')
from non_stationary_fun import zero_pad, FFT, freq_PSD, Complex_plot, Modulation, PowerSpectralDensity, rotate

PsdPlotFlag = False

# --- Main --- #

A = 1
B = 0.5

fmin = 1e-4
fmax = 1e-2

delta_f = 1e-5
delta_t = 1/(2*fmax)
N = 1/(delta_f*delta_t)

freq = np.arange(fmin,fmax,delta_f)
N_f = len(freq)

S_n = PowerSpectralDensity(freq)[0]
S_c = PowerSpectralDensity(freq)[1]

PSD = S_n + S_c

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

stat_diag = (N/delta_t)*(1/2)*(PowerSpectralDensity(freq)[0] + PowerSpectralDensity(freq)[1])


diag = (N/delta_t)*((1/2)*PowerSpectralDensity(freq)[0] + (A*A)*PowerSpectralDensity(freq)[1] + ((B*B)/4)*PowerSpectralDensity(freq+delta_f)[1] + ((B*B)/4)*PowerSpectralDensity(freq-delta_f)[1])
up_diag_1 = (N/delta_t)*A*B*(PowerSpectralDensity(freq)[1])[:-1]
low_diag_1 = (N/delta_t)*A*B*(PowerSpectralDensity(freq)[1])[1:]
up_diag_2 = (N/delta_t)*((B*B)/4)*(PowerSpectralDensity(freq-delta_f)[1])[:-2]
low_diag_2 = (N/delta_t)*((B*B)/4)*(PowerSpectralDensity(freq+delta_f)[1])[2:]
#m = np.diag(np.log(diag), 0) + np.diag(np.log(low_diag), -2) + np.diag(np.log(up_diag), 2)
#m = np.log(np.diag(diag,0) + np.diag(up_diag_1, 1) + np.diag(low_diag_1, -1) + np.diag(low_diag_2, -2) + np.diag(up_diag_2,2) )
m = np.log(np.diag(stat_diag,0))
#+ np.diag(low_diag, -2) + np.diag(up_diag,2)
fig= plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
rotated = (m[::-1])
im=ax.imshow(rotated,extent=[0,N_f,0,N_f],origin="lower")
    
ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format((x)*delta_f))
ax.xaxis.set_major_formatter(ticks_x)
ax.xaxis.tick_top()
ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format((N_f- x)*delta_f))
ax.yaxis.set_major_formatter(ticks_y)
#ax.text(0.45*N,0.56*N,f"# iterations = {N_iter}",bbox=dict(facecolor='none', edgecolor='black'))
plt.xticks(fontsize = 10,rotation = 45)
plt.yticks(fontsize = 10,rotation = 45)

ax.set_xlabel("Frequency [Hz]",fontsize = 15)
ax.xaxis.set_label_position('top') 
ax.set_ylabel('Frequency [Hz]',fontsize = 15)
ax.set_title(r"Analytical noise covariance matrix : $\log_{10}\left |\Sigma_{N}(f,f')\right |$",y = -0.1,fontsize = 15, fontweight='bold')
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


