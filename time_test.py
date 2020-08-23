#author: patrick h. rupp

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
new_dimensions = 2
alpha = 3.0
num_points = [int(i) for i in np.linspace(10, 2000, 10)]

data_factory = DataFactory( seed )

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

time_df = pd.DataFrame(
	index=list(range(len(num_points))), 
	columns=['sklearn', 'kdnuggets', 'kdnuggets_pat']
	)

j = 0

for n in num_points:

	# calculate the fields
	X = data_factory.create_data_helix( n, rand_weight, t_max=20 )

	# run
	Xp1_start_time = dt.now()
	Xp1 = diff_map.fit_transform(X)
	time_df.loc[j, 'sklearn'] = (dt.now() - Xp1_start_time).total_seconds()

	Xp2_start_time = dt.now()
	Xp2 = DiffusionMapsKD.fit_transform(X, new_dimensions, alpha)
	time_df.loc[j, 'kdnuggets'] = (dt.now() - Xp2_start_time).total_seconds()

	Xp3_start_time = dt.now()
	Xp3 = DiffusionMapsPat.fit_transform(X, new_dimensions, alpha)
	time_df.loc[j, 'kdnuggets_pat'] = (dt.now() - Xp3_start_time).total_seconds()
	
	j+=1 #increment

plt.plot( num_points, time_df['sklearn'], color='red', label='sklearn' )
plt.plot( num_points, time_df['kdnuggets'], color='green', label='kdnuggets' )
plt.plot( num_points, time_df['kdnuggets_pat'], color='blue', label='kdnuggets_pat' )
plt.title('Computation Time vs. Data Size')
plt.xlabel('Data Size (n)')
plt.ylabel('Computation Time (s)')
plt.legend()
plt.show()














































