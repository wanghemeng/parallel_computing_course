import numpy as np
import time
from scipy.linalg import cholesky

n = 8000

start_time = time.time()
A = np.zeros((n, n))
for i in range(n):
    j = 0
    while(j <= i):
        A[i][j] = j+1
        j = j+1
    while(j < n):
        A[i][j] = i+1
        j = j+1
end_time = time.time()
print('the time of Cholesky generation is ', round(end_time - start_time, 2)*1000, 'ms')

start_time = time.time()
L1 = np.linalg.cholesky(A)
end_time = time.time()
print('the time of numpy Cholesky factorization is ', round(end_time - start_time, 2)*1000, 'ms')

start_time = time.time()
L2 = cholesky(A, lower=True)
end_time = time.time()
print('the time of scipy Cholesky factorization is ', round(end_time - start_time, 2)*1000, 'ms')

for i in range(n):
    for j in range(i+1):
        if(L1[i][j] != 1 and L1[i][j] != 0):
            print("Matrix factorization failed.")
print("Numpy matrix factorization succeeded.")

for i in range(n):
    for j in range(i+1):
        if(L2[i][j] != 1 and L2[i][j] != 0):
            print("Matrix factorization failed.")
print("Scipy matrix factorization succeeded.")
