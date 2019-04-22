a, gamma = 0.01, 0.8
q_table = [[0]*2]*3

r_table = [[0]*2]*3
print(r_table)

r_table[0][1] = 1


print(r_table)


import numpy as np 

r = np.zeros([3,2])
q = np.zeros([3,2])
print(r)
r[0][0] = 100.023
print(r)
r[:,1] = 1.234
print( r)


q = (1-a)*q + a*(r)
print(q.shape)

ind = []
for i in range(q.shape[0]):
	if i < 2:
		q[i] = q[i] + a*gamma*max(q[i+1])
	max_axis = np.where(q[i]==np.max(q[i],axis=0))
	ind.append(int(max_axis[0]))
print(q, ind)

import numpy as np
a = np.array([[1,1,1],[1,1,1],[1,1,1]])
b = np.array([[1,0,0],[0,1,0],[0,0,1]])

a = np.array([[2,0],[1,1]])
b = np.array([[], []])
c = np..array([[]])
print(a*b)








a = [4,3,2,1]


def qq3(a,k):
	b = False
	for i in range(k):
		if b:
			print(0)
			continue
		a_ = [x for x in a if x != 0]
		if len(a_) == 0:
			b = True
			print(0)
			continue
		min_ = min(a_)
		print( min_)
		a = [x-min_ for x in a if x != 0]

# qq3([0]*100000,10000)

s = 0
def qq2(a, s):
	for i in range(1,5):
		s+= abs(a[i-1])
		a[i] = a[i-1]+a[i]
		print(s,a)

# qq2([5,-4,1,-3,1],0)
import math

a = math.ceil(pow(15, 1.0/4))
print(a)

def qq1(a):
	m = max(a)
	min_ = min(a)
	a = math.ceil(pow(m, 1.0/min_))
	print(int(a+min_))

# qq1([15,4])







