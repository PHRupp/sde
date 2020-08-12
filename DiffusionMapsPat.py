
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances

class DiffusionMapsPat():
	"""
	source: https://gist.github.com/rahulrajpl/36a5724d0c261b915292182b1d741393
	"""

	def __init__():
		return
	
	@staticmethod
	def fit_transform(X, new_dimenions: int, alpha: float = 0.15):
		P_prime, D_left = DiffusionMapsPat.find_diffusion_matrix(X, alpha)
		Xp = DiffusionMapsPat.find_diffusion_map(P_prime, D_left, new_dimenions)
		return Xp
	
	@staticmethod
	def find_diffusion_matrix(X=None, alpha=0.15):
		"""Function to find the diffusion matrix P
			
			>Parameters:
			alpha - to be used for gaussian kernel function
			X - feature matrix as numpy array
			
			>Returns:
			P_prime, P, Di, K, D_left
		"""
		n = X.shape[0]
		dist_map = np.exp(-np.power( euclidean_distances(X, X) ,2) / alpha)
		
		dist_totals = np.sum(dist_map, axis=0).reshape( (n,1) )
		D_right = np.power( dist_totals, 0.5).reshape( (n,1) )
		D_left = np.power( dist_totals, -0.5).reshape( (1,n) )
		
		P = np.array(1/dist_totals) * dist_map
		P_prime = D_right * (P * D_left)

		return P_prime, D_left.reshape( (n,1) )
		
	@staticmethod
	def find_diffusion_map(P_prime, D_left, n_eign=3):
		"""Function to find the diffusion coordinates in the diffusion space
			
			>Parameters:
			P_prime - Symmetrized version of Diffusion Matrix P
			D_left - D^{-1/2} matrix
			n_eigen - Number of eigen vectors to return. This is effectively 
						the dimensions to keep in diffusion space.
			
			>Returns:
			Diffusion_map as np.array object
		"""   
		n_eign = n_eign
		
		eigenValues, eigenVectors = eigh(P_prime)
		idx = eigenValues.argsort()[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]
		
		diffusion_coordinates = D_left * eigenVectors
		
		return diffusion_coordinates[:,:n_eign]
	
	
	
	
	
	
	
	
	
	
