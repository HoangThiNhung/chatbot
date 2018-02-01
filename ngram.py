def ngrams(tokens, n, arr=[]):
    if n == 0:
        return arr
    if len(tokens) < n - 1:
        return ngrams(tokens, n-1)
    else:
        for j in range(n-1):
            new_str = '_ '*(n-1-j)
            if j == 0:
                new_str += tokens[j]
            else:
                for i in reversed(range(n-1)):
                    if j-i >=0:
                        new_str += ' '+tokens[j-i]
            arr.append(new_str)
        for i in range(len(tokens)):
            new_str = ''
            for j in range(n):
                if j < n:
                    if (i + j) < len(tokens):
                        if j == 0:
                            new_str += tokens[i+j]
                        else:
                            new_str += ' '+tokens[i+j]
                    else:
                        new_str += ' _'
            arr.append(new_str)
    return ngrams(tokens, n-1, arr)
