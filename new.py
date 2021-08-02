import numpy as np
from  scipy.optimize import linprog

c = np.array([90,64,5.17,8])
a = np.array([[-3,0,1,0],[0,-4,0,1],[6,4,1,1]])
b = np.array([0,0,1200])
res = linprog(-c,a,b)
print(res)
