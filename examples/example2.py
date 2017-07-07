import environ

from lbqo.core import *
from lbqo.learning import *
from lbqo.baselines import *

"""
This example implements a simple 6 way join optimization. Estimated Sellinger costs are suboptimal
"""


a = set([1, 2, 4])
ra = Relation(a, 'a')

b = set([1 , 2, 3])
rb = Relation(b,'b')

c = set([1, 3, 4])
rc = Relation(c,'c')

d = set([4, 2, 5])
rd = Relation(d,'d')

e = set([5, 4,  3])
re = Relation(e,'e')

f = set([4, 2, 3])
rf = Relation(f,'f')



#impractical approaches

print('Sellinger^ (Full Cost)', sellingerOptimizerFullCost([ra, rb, rc, rd, re, rf]))

print('Brute Force^ ', bruteForceOptimizer([ra, rb, rc, rd, re, rf]))

print("")

#practical approaches

print('Sellinger (Estimated Cost)', sellingerOptimizerEstimatedCost([ra, rb, rc, rd, re, rf]))

print('Rewrite Optimizer', rewriteOptimizer([ra, rb, rc, rd, re, rf]))

print('No Optimization', identityOptimizer([ra, rb, rc, rd, re, rf]))

print("")


#LBQO
from sklearn.svm import SVR

lbqo = LBQO([ra, rb, rc, rd, re, rf], ['a','b','c','d','e','f'])
lbqo.train(SVR(epsilon=1), N=100)
print('Learning-based Query Optimization (100) ', lbqo.optimize([ra, rb, rc, rd, re, rf]))






