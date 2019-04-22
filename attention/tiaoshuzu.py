a = [2,3,1,1,4]
b = [3,2,1,0,4]


def check(a):
	if len(a)== 0:
		return None
	l = len(a)
	reach = 0
	c = 0
	last = 0
	for i in range(l):
		if (i>reach or reach>=l-1):
			break
		reach = max(reach, i+a[i])
	return reach>=l-1

print(check(a))
print(check(b))
		

