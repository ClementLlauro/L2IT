import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey


# ----- Fit parameters dictionnary gathering values for 6 months and one/two/four year(s) of observation. ----- #

fit_parameters = {"half_year": [0.133,243,482,917,0.00258], "one_year" : [0.171,292,1020,1680,0.00215],
                  "two_years" : [0.165,299,611,1340,0.00173], "four_years" : [0.138,-221,521,1680,0.00113]}


# ----- Compute PSD with one year observation fit parameters ----- #
def PowerSpectralDensity(f):
    """
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf 
    Instrumental + confusion background noises
    """        
    
    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**-3    

    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise 

    S_n = ((10/(3*L**2))*(Poms + (4*Pacc)/((2*np.pi*f))**4)*(1 + 0.6*(f/f0)**2)) # instrumental noise contribution  

    S_c = (9*10**-45)*(f**(-7/3))*np.exp(-0.171*f + 292*f*np.sin(1020*f))*(1 + np.tanh(1680*(0.00215- f))) #confusion bakcground of 
                                                                                                           #unresolved sources contribution (observation time = 1 yr)
    # possible improvements : store the parameter values in a dictionnary or table to compute PSD at different observation times

    return (S_n, S_c)


# ----- Compute PSD for any observation time ----- #
def PowerSpectralDensity_dico(f,key):
    """
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf 
    Instrumental + confusion background noises
    """        
    
    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**-3    

    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise 

    S_n = ((10/(3*L**2))*(Poms + (4*Pacc)/((2*np.pi*f))**4)*(1 + 0.6*(f/f0)**2)) # instrumental noise contribution  

    S_c = (9*10**-45)*(f**(-7/3))*np.exp(-fit_parameters[key][0]*f + fit_parameters[key][1]*f*np.sin(fit_parameters[key][2]*f))*(1 + np.tanh(fit_parameters[key][3]*(fit_parameters[key][4]- f))) 
    #confusion background of unresolved sources contribution 
    

    return (S_n,S_c)

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

def FFT(waveform):
    """
    Here we taper the signal, pad and then compute the FFT. We remove the zeroth frequency bin because 
    the PSD (for which the frequency domain waveform is used with) is undefined at f = 0.
    """
    N = len(waveform)
    taper = tukey(N,0.1)
    waveform_w_pad = zero_pad(waveform*taper)
    return np.fft.rfft(waveform_w_pad)[1:]

def freq_PSD(waveform_t,delta_t,observation_t):
    """
    Here we take in a waveform and sample the correct fourier frequencies and output the PSD. There is no 
    f = 0 frequency bin because the PSD is undefined there.
    """    
    n_t = len(zero_pad(waveform_t))
    freq = np.fft.rfftfreq(n_t,delta_t)[1:]
    S_n,S_c = PowerSpectralDensity_dico(freq,observation_t)
    return freq,S_n,S_c

def Complex_plot(data_f):
    # extract real part
    x = [ele.real for ele in data_f]
    # extract imaginary part
    y = [ele.imag for ele in data_f]

    # plot the complex numbers
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()

def Modulation(A,B,T,t):
    return(A+B*np.cos((2*np.pi*t)/T))

def rotate(matrix):
      temp_matrix = []
      column = len(matrix)-1
      for column in range(len(matrix)):
         temp = []
         for row in range(len(matrix)-1,-1,-1):
            temp.append(matrix[row][column])
         temp_matrix.append(temp)
      for i in range(len(matrix)):
         for j in range(len(matrix)):
            matrix[i][j] = temp_matrix[i][j]
      return matrix