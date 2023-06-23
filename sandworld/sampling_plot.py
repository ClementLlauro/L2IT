import numpy as np
import matplotlib.pyplot as plt

# ----- OVERSAMPLING ----- #
t_obs = 10
# --- Continuous sine plot --- #

delta_t_over = 0.20

continuous_t = np.arange(0,t_obs,0.01)

continuous_h = np.sin(2*np.pi*continuous_t)

plt.plot(continuous_t,continuous_h,color='black')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Oversampled signal', fontsize=20)


# --- Discrete sine plot --- #

delta_t_over = 0.20
discrete_t = np.arange(0,t_obs,delta_t_over)
discrete_h = np.sin(np.pi*2*discrete_t)

oversampling_frequency = 1/delta_t_over

plt.stem(discrete_t, discrete_h)
plt.show()

# --- Frequency plot --- #

f_aliased = abs(8*0.5-oversampling_frequency)

freq = np.arange(-2*f_aliased,2*f_aliased+f_aliased/4,f_aliased/4)
mag = []
for truc in freq:
    if abs(truc)-f_aliased == 0:
        mag.append(1)
    else:
        mag.append(0)

plt.stem(freq,mag)
plt.xlabel('Frequency [Hz]', fontsize=15)
plt.ylabel(r'$|\tilde{h}(f)|/|\tilde{h}(f)_{max}$', fontsize=15)
plt.title('Oversampled signal',fontsize=20)
plt.show()



# ----- UNDERSAMPLING ----- #

# --- Continuous sine plot --- #

continuous_t = np.arange(0,t_obs,0.01)

continuous_h = np.sin(2*np.pi*continuous_t)

plt.plot(continuous_t,continuous_h,color='black')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Undersampled signal', fontsize=20)


# --- Discrete sine plot --- #

delta_t_under = 0.8
discrete_t = np.arange(0,t_obs,delta_t_under)
discrete_h = np.sin(np.pi*2*discrete_t)

plt.stem(discrete_t, discrete_h)



undersampling_frequency = 1/delta_t_under

plt.plot(continuous_t,-np.sin((np.pi/2)*continuous_t),'--')
plt.show()

# --- Frequency plot --- #

f_aliased = abs(2*0.5-undersampling_frequency)

freq = np.arange(-2*f_aliased,2*f_aliased+f_aliased/2,f_aliased/2)
mag = []
for truc in freq:
    if abs(truc)-f_aliased == 0:
        mag.append(1)
    else:
        mag.append(0)

plt.stem(freq,mag)
plt.xlabel('Frequency [Hz]', fontsize=15)
plt.ylabel(r'$|\tilde{h}(f)|/|\tilde{h}(f)_{max}$', fontsize=15)
plt.title('Undersampled signal',fontsize=20)
plt.text(x=-0.14,y=0.2,s = r'$f_{a}$'+f' = {f_aliased} Hz',fontsize=15,color='red')
plt.show()

# --- Noise & Modulated noise --- #

noise = np.random.normal(0,1,len(continuous_t))

plt.plot(continuous_t,noise)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Gaussian noise',fontsize=20)
plt.show()

plt.plot(continuous_t,noise*np.sin(continuous_t))
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Modulated gaussian noise',fontsize=20)
plt.show()