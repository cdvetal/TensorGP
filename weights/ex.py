import numpy as np

arr=np.array([12, 24, 23, 16, 13.5, 21, 20, 13.5, 17,8,19,11,1,9,2,3,5.5,4,5.5,22,18,10,7,15])
med=arr.mean()

print(med)

res = 0
for e in arr:
    res += (e-med)*(e-med)
print(res)