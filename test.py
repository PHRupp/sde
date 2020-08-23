import numpy as np

a = np.array([
	[7,4,2,6,6],
	[1,3,8,3,4],
	[7,3,0,8,1],
	[2,4,7,3,4],
	[8,5,9,3,7]
	]) * 1.0
	
diags = np.array([1,2,3,4,5]) * 1.0
	
b11 = diags[:].reshape( (len(diags),1) )
b12 = diags[:].reshape( (1,len(diags)) )
b2 = np.diag(diags)

#c1 = b1 * (a * b1)
c2 = np.matmul(b2, np.matmul(a, b2))


p = 
print(p)

c1 = b11 * (b12 * a)
print(p - np.matmul(a, b2))

print(c2-c1)
print('---')