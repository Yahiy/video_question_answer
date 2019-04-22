def longestSubseq(s):
        """
        :type s: str
        :rtype: int
        """
        print(s)
        if len(s) == 0:
            return 0
        str_len = len(s)
        dp_matrix = [[0]*len(s) for i in range(str_len)]
        
        k = 0
        while k < str_len:
            for i in range(str_len-k):
                j = i + k
                if i == j:
                    dp_matrix[i][j] = 1
                elif s[i] == s[j]:
                    dp_matrix[i][j] = dp_matrix[i+1][j-1] + 2
                else:
                    dp_matrix[i][j] = max(dp_matrix[i][j-1], dp_matrix[i+1][j])
            k += 1
        h = dp_matrix[0][str_len-1]
        o = len(s) - h + 1 
        print(o)
        return o
shot = []
s = '1431567353242'
all = longestSubseq(s)
for i in range(len(s)):
    l = s[:i]
    r = s[i:]
    o = min((longestSubseq(l)+longestSubseq(r)), all)
    shot.append(o)

print(min(shot))



