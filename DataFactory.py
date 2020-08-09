
import numpy as np
from numpy.random import random

class DataFactory():

	def __init__( self, random_seed: int = None ):
		self.seed = random_seed
		if self.seed is not None:
			np.random.seed( self.seed )
		return
	
	def create_data_helix( self, num_points: int, rand_weight: int, t_max: int = 20 ):
		t = np.linspace(0, t_max, num_points)
		x = np.cos(t) + rand_weight * random( num_points )
		y = np.sin(t) + rand_weight * random( num_points )
		z = 2*t + rand_weight * random( num_points )
		X = np.hstack([
			t.reshape((num_points,1)),
			x.reshape((num_points,1)),
			y.reshape((num_points,1)),
			z.reshape((num_points,1))
		])
		return X
	
	def create_data_S( self, num_points: int, rand_weight: int ):
		t = np.linspace(0, 20, num_points)
		x = np.cos(t) + rand_weight * random( num_points )
		y = np.sin(t) + rand_weight * random( num_points )
		z = 2*t + rand_weight * random( num_points )
		X = np.hstack([
			t.reshape((num_points,1)),
			x.reshape((num_points,1)),
			y.reshape((num_points,1)),
			z.reshape((num_points,1))
		])
		return X