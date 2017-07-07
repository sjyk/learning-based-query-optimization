import environ

from lbqo.core import *
from lbqo.learning import *
from lbqo.baselines import *

"""
This example implements a simple 6 way join optimization. Generalization to other joins
"""


a = set([1, 2, 4, 5, 6, 7])
ra = Relation(a, 'a')

b = set([1 , 2, 3, 5, 6, 10])
rb = Relation(b,'b')

c = set([1, 3, 4, 5])
rc = Relation(c,'c')

d = set([4, 2, 5, 6])
rd = Relation(d,'d')

e = set([5, 4,  3, 8])
re = Relation(e,'e')

f = set([4, 2, 3, 6, 10])
rf = Relation(f,'f')



#LBQO
from sklearn.svm import SVR

lbqo = LBQO([ra, rb, rc, rd, re, rf], ['a','b','c','d','e','f'])
lbqo.train(SVR(epsilon=1), N=100)
print('Learning-based Query Optimization a*b*c*d*e*f', lbqo.optimize([ra, rb, rc, rd, re, rf]))
print('Brute Force a*b*c*d*e*f', bruteForceOptimizer([ra, rb, rc, rd, re, rf]))

print("")

print('Learning-based Query Optimization a*d*f', lbqo.optimize([ra, rd, rf]))
print('Brute Force a*d*f', bruteForceOptimizer([ra, rd, rf]))

print("")

print('Learning-based Query Optimization a*b*c*d*e', lbqo.optimize([ra, rb, rc, rd, re]))
print('Brute Force a*b*c*d*e', bruteForceOptimizer([ra, rb, rc, rd, re]))

print("")

print("Generalization With New Relations")

lbqo = LBQO([ra, rb, rc, rf], ['a','b','c', 'f'])
lbqo.train(SVR(epsilon=1), N=100)
print('Learning-based Query Optimization a*b*c*d*e*f', lbqo.optimize([ra, rb, rc, rd, re, rf]))
print('Brute Force a*b*c*d*e*f', bruteForceOptimizer([ra, rb, rc, rd, re, rf]))



