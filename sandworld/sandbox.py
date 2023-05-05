import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
from matplotlib import pyplot, transforms

# Generate data

delta_f = 7.8125e-06
M = np.matrix([[i*j+1 for i in range (0,1025)] for j in range (0,1025)])

# Setup figures

fig= plt.figure(figsize=(8,8))
ax2 = fig.add_subplot(111)

# Plot two identical plots

im2=ax2.imshow(M)

# Change only ax2
scale_x = delta_f
scale_y = delta_f
ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format(x*scale_x))
ax2.xaxis.set_major_formatter(ticks_x)
ax2.xaxis.tick_top()
ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format(x*scale_y))
ax2.yaxis.set_major_formatter(ticks_y)

plt.xticks(fontsize = 8,rotation = 45)
plt.yticks(fontsize = 8,rotation = 45)


ax2.set_xlabel("Frequency [Hz]")
ax2.xaxis.set_label_position('top') 
ax2.set_ylabel('Frequency [Hz]')
plt.title('Noise covariance matrix(modulated)',y = -0.1,fontsize = 15)
plt.colorbar(im2,fraction = 0.04)

plt.show()

N = np.eye(1025)*6

rotated = N[::-1]

plt.imshow(rotated)
plt.show()