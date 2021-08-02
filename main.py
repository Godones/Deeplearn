from pulp import *

problem = LpProblem("max",LpMaximize)

a1 = LpVariable("a1")
a2 = LpVariable("a2")
b1 = LpVariable("b1")
b2 = LpVariable("b2")

z = 90*a1 + 64*a2 + 5.17*b1+8*b2
s1 = b2 <= 4*a2
s4 = b1 <= 3*a1
s3 = 6*a1 + 4*a2 +b1+b2 <=1200

problem +=s1
# problem +=s2
problem +=s3
problem +=s4
problem +=z
problem.solve()
print(problem)
print(a1.value(),a2.value(),b1.value(),b2.value())
print((value(problem.objective)))
