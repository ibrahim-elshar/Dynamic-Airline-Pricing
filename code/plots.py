# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  


z1 = []
z2 = []
for i in range(m.num_stages):
    for j in range(101):
        z1.append(m.policy[i,j][0])
        z2.append(m.policy[i,j][1])
        


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(12)
y = np.arange(101)
X, Y = np.meshgrid(y, x)
z1 = np.array(z1)
Z1 = z1.reshape(Y.shape)
z2 = np.array(z2)
Z2 = z2.reshape(Y.shape)

ax.plot_surface(Y, X, Z1)
ax.plot_surface(Y, X, Z2)