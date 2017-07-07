import environ

from lbqo.core import *
from lbqo.learning import *
from lbqo.baselines import *

"""
This example implements a more complicated N-way join
"""

num_tables = 15
min_size = 100
max_size = 1000

relations = []
rnames = []
for i in range(num_tables):
    size = np.maximum(int(np.random.rand(1)*max_size),min_size)
    r = Relation(set(np.random.choice(range(max_size), size, replace=False).tolist()), str(i))
    relations.append(r)
    rnames.append(str(i))


print('Sellinger (Estimated Cost)', sellingerOptimizerEstimatedCost(relations))

print('Rewrite Optimizer', rewriteOptimizer(relations))

print('No Optimization', identityOptimizer(relations))


#LBQO
from sklearn.svm import SVR

lbqo = LBQO(relations, rnames)
lbqo.train(SVR(epsilon=1), N=1000)
print('Learning-based Query Optimization (1000) ', lbqo.optimize(relations))
