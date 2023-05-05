from matplotlib import cbook
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot, transforms
delta_f = 7.8125e-06
M = np.matrix([[i*j+1 for i in range (0,1025)] for j in range (0,1025)])
N = np.eye(1025)*6e-20
rotated = N[::-1]
tinyN = rotated[20:50,20:50]
def get_demo_image():
    z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)

fig, ax = plt.subplots(figsize=[8, 8])

# make data
Z, extent = get_demo_image()
Z2 = np.zeros((150, 150))


ny, nx = tinyN.shape
rotated[900:900+ny, 900:900+nx] = tinyN

ax.imshow(rotated,extent=[0,1025,0,1025],origin="lower")
A = 0
B = 1025
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.imshow(rotated,extent=[0,1025,0,1025],origin="lower")
# subregion of the original image
#x1, x2, y1, y2 = 30,45,30,45
x1, x2, y1, y2 = 85,125,900,940
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()

plt.imshow(N)
plt.colorbar()
plt.show()