import math

def entropy(p):
    total = 0
    for i in p:
        total += i * math.log2(i)

    return - total

def entropyBoolean(p):
    if p == 1 or p == 0:
        return 0
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def informationGain(b,t,data):
    summation = 0
    for d in data:
        summation += ((d[0] + d[1]) / t)*entropyBoolean(d[0] / (d[0] + d[1]))
    return b - summation