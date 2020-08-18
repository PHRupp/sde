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
num_points = 1000
new_dimensions = 2
alpha = 3.0

epoch = dt.utcfromtimestamp(0)
def unix_time_ns():
    return (dt.utcnow() - epoch).total_seconds() * 1000000000

data_factory = DataFactory( seed )

# calculate the fields
#X = data_factory.create_data_helix( num_points, rand_weight, t_max=20 )
#X = data_factory.create_sine_wave( num_points, rand_weight, x_max=20 )
X = data_factory.create_data_torus( num_points, rand_weight )

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
print( 'Start sklearn - %d' % unix_time_ns() )
Xp1 = diff_map.fit_transform(X)

print( 'Start DiffusionMapKD - %d' % unix_time_ns() )
Xp2 = DiffusionMapsKD.fit_transform(X, new_dimensions, alpha)

print( 'Start DiffusionMapPat - %d' % unix_time_ns() )
Xp3 = DiffusionMapsPat.fit_transform(X, new_dimensions, alpha)
print( 'End DiffusionMapPat - %d' % unix_time_ns() )

"""
# save reduced data to file
np.savetxt('Xp_sklearn.csv', Xp1, delimiter=',')
np.savetxt('Xp_KDnuggets.csv', Xp2, delimiter=',')
np.savetxt('Xp_Pat.csv', Xp3, delimiter=',')
"""

embedding_plot( diff_map, scatter_kwargs = {'c': Xp1[:,0], 'cmap':'Spectral'} )
data_plot( diff_map, dim=3, scatter_kwargs = {'cmap':'Spectral'} )
plt.show()


#plt.scatter(Xp2[:,0], Xp2[:,1])
#plt.show()

#plt.scatter(Xp3[:,0], Xp3[:,1])
#plt.show()














































