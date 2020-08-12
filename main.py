#author: patrick h. rupp

import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map
from pydiffmap.visualization import embedding_plot, data_plot

# include self-built library
from DataFactory import DataFactory
from DiffusionMapsKD import DiffusionMapsKD
from DiffusionMapsPat import DiffusionMapsPat
from Utility import get_pid_memory_usage

# define parameters
seed = 12345
rand_weight = 1
num_points = 800
new_dimensions = 2
alpha = 3.0

data_factory = DataFactory( seed )

# calculate the fields
X = data_factory.create_data_helix( num_points, rand_weight, t_max=20 )
print(X)

# save original data to file
np.savetxt('X.csv', X, delimiter=',')

diff_map = diffusion_map.DiffusionMap.from_sklearn(
	n_evecs = new_dimensions,
	epsilon = 1.0,
	alpha = alpha,
	k = 10, #number of neighbors to use
	neighbor_params={
		'n_jobs': -1,
		'algorithm': 'ball_tree'
	}
)

# run
Xp1_start_time = dt.now()
Xp1 = diff_map.fit_transform(X)
Xp1_end_time = dt.now()
Xp1_time_diff = Xp1_end_time - Xp1_start_time
print(Xp1_time_diff)

Xp2_start_time = dt.now()
Xp2 = DiffusionMapsKD.fit_transform(X, new_dimensions, alpha)
Xp2_end_time = dt.now()
Xp2_time_diff = Xp2_end_time - Xp2_start_time
print(Xp2_time_diff)

Xp3_start_time = dt.now()
Xp3 = DiffusionMapsPat.fit_transform(X, new_dimensions, alpha)
Xp3_end_time = dt.now()
Xp3_time_diff = Xp3_end_time - Xp3_start_time
print(Xp3_time_diff)

# save reduced data to file
np.savetxt('Xp_sklearn.csv', Xp1, delimiter=',')
np.savetxt('Xp_KDnuggets.csv', Xp2, delimiter=',')
np.savetxt('Xp_Pat.csv', Xp3, delimiter=',')


embedding_plot( diff_map, scatter_kwargs = {'c': Xp1[:,0], 'cmap':'Spectral'} )
data_plot( diff_map, dim=3, scatter_kwargs = {'cmap':'Spectral'} )
plt.show()


plt.scatter(Xp2[:,0], Xp2[:,1])
plt.show()

plt.scatter(Xp3[:,0], Xp3[:,1])
plt.show()














































