#author: patrick h. rupp

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map
from pydiffmap.visualization import embedding_plot, data_plot

# include self-built library
from DataFactory import DataFactory
from DiffusionMapsKD import DiffusionMapsKD
from DiffusionMapsPat import DiffusionMapsPat

# define parameters
seed = 12345
rand_weight = 3
num_points = 5
new_dimensions = 2
alpha = 10.0

data_factory = DataFactory( seed )

# calculate the fields
X = data_factory.create_data_helix( num_points, rand_weight, t_max=20 )

# save original data to file
np.savetxt('X.csv', X, delimiter=',')

diff_map = diffusion_map.DiffusionMap.from_sklearn(
	n_evecs = new_dimensions,
	epsilon = 1.0,
	alpha = alpha,
	k = 10,
	neighbor_params={
		'n_jobs': -1,
		'algorithm': 'ball_tree'
	}
)

Xp1 = diff_map.fit_transform(X)
Xp2 = DiffusionMapsKD.fit_transform(X, new_dimensions, alpha)
Xp3 = DiffusionMapsPat.fit_transform(X, new_dimensions, alpha)
"""
# save reduced data to file
np.savetxt('Xp_sklearn.csv', Xp1, delimiter=',')
np.savetxt('Xp_KDnuggets.csv', Xp2, delimiter=',')
np.savetxt('Xp_Pat.csv', Xp3, delimiter=',')

embedding_plot( diff_map, scatter_kwargs = {'c': Xp1[:,0], 'cmap':'Spectral'} )
data_plot( diff_map, dim=3, scatter_kwargs = {'cmap':'Spectral'} )
plt.show()
"""















































