
import numpy as np
import random as rand

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
    
        n1 = int(num_points/3)
        n2 = int(num_points/3)
        n3 = num_points - (n1+n2)
        
        step_rand = 0.025
    
        x1 = np.linspace(0, x_max, n1) + step_rand * random( n1 )
        x2 = np.linspace(0, x_max, n2) + step_rand * random( n2 )
        x3 = np.linspace(0, x_max, n3) + step_rand * random( n3 )
        rand.shuffle( x1 )
        rand.shuffle( x2 )
        rand.shuffle( x3 )
    
        y1 = np.linspace(0, x_max, n1) + step_rand * random( n1 )
        y2 = np.linspace(0, x_max, n2) + step_rand * random( n2 )
        y3 = np.linspace(0, x_max, n3) + step_rand * random( n3 )
        rand.shuffle( y1 )
        rand.shuffle( y2 )
        rand.shuffle( y3 )
    
        z1 = 0.0 + step_rand * random( n1 )
        z2 = 0.0 + step_rand * random( n2 )
        z3 = 0.0 + step_rand * random( n3 )
        
        t1 = np.sin( x1 + y1 + z1 ) + rand_weight * random( n1 )
        t2 = np.sin( x2 + y2 + z2 ) + rand_weight * random( n2 )
        t3 = np.sin( x3 + y3 + z3 ) + rand_weight * random( n3 )
        
        X1 = np.hstack([
            x1.reshape((n1,1)),
            y1.reshape((n1,1)),
            t1.reshape((n1,1)),
            z1.reshape((n1,1))
        ])
        
        X2 = np.hstack([
            x2.reshape((n2,1)),
            y2.reshape((n2,1)),
            t2.reshape((n2,1)),
            z2.reshape((n2,1))
        ])
        
        X3 = np.hstack([
            x3.reshape((n3,1)),
            y3.reshape((n3,1)),
            t3.reshape((n3,1)),
            z3.reshape((n3,1))
        ])
        
        X = np.vstack([X1,X2,X3])
        
        return X
    
    def create_data_torus( self, num_points: int, rand_weight: int ):
        """https://mathworld.wolfram.com/Torus.html"""
        
        ns = int(np.ceil( np.sqrt(num_points) ))
        n = int(ns**2)
        
        c = 4.0
        a = 1.0
        step_rand = 0.025
        v = np.linspace(0, 2.0*np.pi, ns) + step_rand * random(ns)
        u = np.linspace(0, 2.0*np.pi, ns) + step_rand * random(ns)
        
        q = (c + a*np.cos(v))
    
        x = np.array([q*np.cos(u) + rand_weight * random(ns) for i in range(ns)]).flatten()
        y = np.array([q*np.sin(u) + rand_weight * random(ns) for i in range(ns)]).flatten()    
        z = np.array([a*np.sin(v) + rand_weight * random(ns) for i in range(ns)]).flatten()
        
        t = np.cos( x + y + z )
        
        X = np.hstack([
            x.reshape((n,1)),
            y.reshape((n,1)),
            z.reshape((n,1)),
            t.reshape((n,1))
        ])
        
        return X
    
    def create_data_spiral( self, num_points: int, rand_weight: int ):
        """https://mathworld.wolfram.com/Torus.html"""
        
        ns = int(np.ceil( np.sqrt(num_points) ))
        n = int(ns**2)
        
        a = 1.0
        step_rand = 0.025
        v = np.linspace(0, 2.0*np.pi, ns) + step_rand * random(ns)
        
        r = a * v
    
        x = np.array([r * np.cos(v) + rand_weight * random(ns) for i in range(ns)]).flatten()
        y = np.array([r * np.sin(v) + rand_weight * random(ns) for i in range(ns)]).flatten()
        z = np.array([3 * random(ns) for i in range(ns)]).flatten()
        
        t = 2 * x**2 - 3 * x + rand_weight * random(n)
        
        X = np.hstack([
            x.reshape((n,1)),
            y.reshape((n,1)),
            z.reshape((n,1)),
            t.reshape((n,1))
        ])
        
        return X
