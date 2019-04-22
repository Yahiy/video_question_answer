# import sys
# a = sys.stdin.readline().strip()
a = 'abc3(((2(h))))'

tem = []
numl = ["0","1","2","3","4","5","6","7","8","9"]
kuo = ["(",")","[","]","{","}"]
kuo1 = ["(","{","["]
kuo2 = [")","}","]"]

i = 0
l = len(a)
while(i <= l-1):
    j = i
    if a[i] in numl:
        num = []
        st = []
        num.append(a[i])
        while(j < l-1):
            j = j +1
            if a[j] in numl:
                num.append(a[j])
            else:
                break
        num = "".join(num)
        num = int(num)
        while(j < l-1):
            # j = j +1
            if a[j] in kuo1:
                j = j + 1
            else:
                if a[j] in kuo2 and a[j+1] not in kuo2:
                    j = j + 1
                    break
                else:
                    if a[j] in kuo2:
                        j = j+1
                    else:
                        st.append(a[j])
                        j = j + 1
        #st = "".join(st)
        for x in range(num):
            for z in st:
                tem.append(z)
        i = j +1
    else:
        tem.append(a[i])
        i = i + 1

tem = tem[::-1]
tem = "".join(tem)
print(tem)



