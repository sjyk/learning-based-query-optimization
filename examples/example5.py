import environ

from lbqo.core import *
from lbqo.learning import *
from lbqo.baselines import *

"""
This example implements a more complicated N-way join

('Sellinger (Estimated Cost)', (((((((((((((((((((17 * 14) * 0) * 16) * 5) * 8) * 19) * 18) * 15) * 9) * 13) * 7) * 10) * 6) * 2) * 3) * 4) * 1) * 11) * 12) : 10500, 'Time: ', 41.332182)
('Rewrite Optimizer', (((((((((((((((((((0 * 14) * 17) * 16) * 5) * 8) * 19) * 18) * 15) * 9) * 13) * 7) * 10) * 6) * 2) * 3) * 4) * 1) * 11) * 12) : 11200, 'Time: ', 0.000159)
('Learning-based Query Optimization (1000) ', (3 * (11 * (6 * (4 * (16 * (18 * (12 * (17 * (5 * (13 * (10 * (14 * ((1 * (9 * ((0 * (2 * (7 * 8))) * 15))) * 19))))))))))))) : 471840, 'Time: ', 0.237478)
"""

num_tables = 20
min_size = 100
max_size = 1000

relations = []
rnames = []
for i in range(num_tables):
    size = np.maximum(int(np.random.rand(1)*max_size),min_size)
    r = Relation(set(np.random.choice(range(max_size), size, replace=False).tolist()), str(i))
    relations.append(r)
    rnames.append(str(i))


import datetime
now = datetime.datetime.now()

print('Sellinger (Estimated Cost)', sellingerOptimizerEstimatedCost(relations), 'Time: ', (datetime.datetime.now() - now).total_seconds() )


now = datetime.datetime.now()

print('Rewrite Optimizer', rewriteOptimizer(relations), 'Time: ', (datetime.datetime.now() - now).total_seconds() )

#LBQO
from sklearn.svm import SVR

lbqo = LBQO(relations, rnames)
lbqo.train(SVR(epsilon=1), N=10)

now = datetime.datetime.now()

print('Learning-based Query Optimization (1000) ', lbqo.optimize(relations), 'Time: ', (datetime.datetime.now() - now).total_seconds() )
