import numpy as np
import matplotlib.pyplot as plt

# --- Continuous sine plot --- #


continuous_t = np.arange(0,6*np.pi,(6*np.pi)/1000)

continuous_h = np.sin(continuous_t)

plt.plot(continuous_t,continuous_h,color='black')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Oversampled signal', fontsize=20, fontweight='bold')


# --- Discrete sine plot --- #

delta_t_over = 0.25
discrete_t = np.arange(0,6*np.pi,6*0.25)
discrete_h = np.sin(discrete_t)

plt.scatter(discrete_t, discrete_h,color='red')
plt.show()
