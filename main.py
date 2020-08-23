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
num_points = 5000
new_dimensions = 2
alpha = 3.0

pid = os.getpid()

epoch = dt.utcfromtimestamp(0)
def unix_time_ns():
    return (dt.utcnow() - epoch).total_seconds() * 1000000000

data_factory = DataFactory( seed )

# calculate the fields
X = data_factory.create_data_helix( num_points, rand_weight, t_max=20 )
#X = data_factory.create_sine_wave( num_points, rand_weight, x_max=20 )
#X = data_factory.create_data_torus( num_points, rand_weight )
#X = data_factory.create_data_spiral( num_points, rand_weight )

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
print( 'Memory (KB) before pydiffmap: %d' % get_pid_memory_usage(pid) )
print( 'Start time(ns) sklearn - %d' % unix_time_ns() )
Xp1 = diff_map.fit_transform(X)

print( 'Memory (KB) before DiffusionMapKD: %d' % get_pid_memory_usage(pid) )
print( 'Start time(ns) DiffusionMapKD - %d' % unix_time_ns() )
Xp2 = DiffusionMapsKD.fit_transform(X, new_dimensions, alpha)

print( 'Memory (KB) before DiffusionMapPat: %d' % get_pid_memory_usage(pid) )
print( 'Start time(ns) DiffusionMapPat - %d' % unix_time_ns() )
Xp3 = DiffusionMapsPat.fit_transform(X, new_dimensions, alpha)
print( 'End time(ns) DiffusionMapPat - %d' % unix_time_ns() )

"""
# save reduced data to file
np.savetxt('Xp_sklearn.csv', Xp1, delimiter=',')
np.savetxt('Xp_KDnuggets.csv', Xp2, delimiter=',')
np.savetxt('Xp_Pat.csv', Xp3, delimiter=',')
"""
exit()
embedding_plot( diff_map, scatter_kwargs = {'c': Xp1[:,0], 'cmap':'Spectral'} )
data_plot( diff_map, dim=3, scatter_kwargs = {'cmap':'Spectral'} )
plt.show()

# The plot doesn't always scale very well if one of the dimensions
# have extremely small variation. The limits on the x-axis must be
# enforced so that it doesn't show up as a verticle line and that 
# the true profile is visible. 
xmin=np.min(Xp2[:,0])
xmax=np.max(Xp2[:,0])
ymin=np.min(Xp2[:,1])
ymax=np.max(Xp2[:,1])
plt.title('KDNuggets Diffusion Map Reduction')
plt.scatter(Xp2[:,0], Xp2[:,1])
plt.xlim(left=xmin, right=xmax)
plt.ylim(bottom=ymin, top=ymax)
plt.show()














































