import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

# A low hump with a spike coming out.
# Needs to have z/colour axis on a log scale so we see both hump and spike.
# linear scale only shows the spike.
Z1 = np.exp(-(X)**2 - (Y)**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
Z = Z1 + 50 * Z2
fig= plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)



c = ax.pcolor(norm=PowerNorm(gamma=0.5,vmin=Z.min(), vmax=Z.max()), cmap='PuBu_r')
fig.colorbar(c, ax=ax)



plt.show()