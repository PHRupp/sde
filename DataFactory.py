
import numpy as np
from numpy.random import random

class DataFactory():

	def __init__( self, random_seed: int = None ):
		self.seed = random_seed
		if self.seed is not None:
			np.random.seed( self.seed )
		return
	
	def create_data_helix( self, num_points: int, rand_weight: float, t_max: int = 20 ):
		n1 = int(num_points/2)
		n2 = num_points - n1
		
		t1 = np.linspace(0, t_max, n1) + 0.5 * random( n1 )
		x1 = np.cos(t1) + rand_weight * random( n1 )
		y1 = np.sin(t1) + rand_weight * random( n1 )
		z1 = 2*t1 + rand_weight * random( n1 )
		X1 = np.hstack([
			t1.reshape((n1,1)),
			x1.reshape((n1,1)),
			y1.reshape((n1,1)),
			z1.reshape((n1,1))
		])
		
		t2 = np.linspace(0, t_max, n2) + 0.5 * random( n2 )
		x2 = np.cos(t2) + rand_weight * random( n2 )
		y2 = np.sin(t2) + rand_weight * random( n2 )
		z2 = 2*t2 + rand_weight * random( n2 )
		X2 = np.hstack([
			t2.reshape((n2,1)),
			x2.reshape((n2,1)),
			y2.reshape((n2,1)),
			z2.reshape((n2,1))
		])
		
		X = np.vstack([X1,X2])
		
		return X
		
	def create_sine_wave( self, num_points: int, rand_weight: float, x_max: int = 20):
	
		n1 = int(num_points/2)
		n2 = num_points - n1
		
		x1 = np.linspace(0, x_max, n1) + 0.5 * random( n1 )
		y1 = np.linspace(0, x_max, n1) + 0.5 * random( n1 )
		z1 = np.sin( np.sqrt( np.power(x1,2) + np.power(y1,2) ))
		t1 = np.linspace(0, t_max, n1)
		X1 = np.hstack([
			x1.reshape((n1,1)),
			y1.reshape((n1,1)),
			z1.reshape((n1,1)),
			t1.reshape((n1,1))
		])
		
		x2 = np.linspace(0, x_max, n2)
		y2 = np.sin(t2) + rand_weight * random( n2 )
		z2 = 2*t2 + rand_weight * random( n2 )
		t2 = np.linspace(0, t_max, n2)
		X2 = np.hstack([
			x2.reshape((n2,1)),
			y2.reshape((n2,1)),
			z2.reshape((n2,1)),
			t2.reshape((n2,1))
		])
		
		X = np.vstack([X1,X2])
	
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