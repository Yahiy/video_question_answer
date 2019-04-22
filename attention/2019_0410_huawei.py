# in_list = sys.stdin.readline().strip().split(" ")
in_list = []

o = []



def aa(x):
	def bb(y):
		return x+y
	return bb
a = aa(5)
print('aa', a(a(6)))


# def add_zero_string(string):
# 	t = 8 - len(string)
# 	for i in range(t):
# 		string = string + '0'
# 	return string

# for i in in_list[1:]:
# 	if 0<len(i) < 9:
# 		o.append(add_zero_string(i))
# 	else:
# 		while(len(i) > 7):
# 			o.append(i[:8])
# 			i = i[8:]
# 		if len(i):
# 			o.append(add_zero_string(i))

# o.sort()

# print(o)

# a = ['d','sddcs']
# print(''.join(a))


# a = 'fadsjk2(3(h))'

# o = ''
# nums = ["0","1","2","3","4","5","6","7","8","9"]
# l = ['[','{','(']
# r = [']', '}', ')']

# hr, hl = 0, 0
# c = []
# cc = ''
# kuo1, kuo2 = [], []

# for i in a:
# 	print(i,o,c, hr, hl)
# 	if i in l:
# 			hl +=1
# 			continue
# 	if i in r:
# 			hr +=1
# 			continue
# 	else:
# 		if i in nums:
# 			c.append(int(i))
# 			continue
# 		elif hr == hl == 0:
# 			o = o+i
# 		elif hl>0 and hr<hl:
# 			for j in range(c[-1]):
# 				o = o+i
# 			c = c[:1]
# 			cc
# 		continue


# print(o[::-1])











	
