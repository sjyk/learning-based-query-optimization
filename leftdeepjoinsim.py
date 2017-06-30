import numpy as np
import itertools


"""
Crude simulator to find the best left deep join
"""


class JoinSimulator(object):

    def __init__(self, num_tables=6, max_size=1000, min_size=50):
        self.tables = {}
        for i in range(num_tables):
            size = np.maximum(int(np.random.rand(1)*max_size),min_size)
            self.tables[i] = np.random.choice(range(max_size), size, replace=False)

    def join(self, tablei, tablej):
        result = set(tablei.tolist()).intersection(set(tablej.tolist())) #equijoin simplest case
        cost = len(tablei) * len(tablej) #nested loop join cost
        return np.array(list(result)), cost


    #only use leftdeep joins which is what the selinger optimizer does
    def leftDeepJoin(self, order):

        cur = self.tables[order[0]]
        cost = 0

        for i in range(1, len(order)):
            cur, inc_cost = self.join(cur, self.tables[order[i]])
            cost += inc_cost

        return inc_cost


    #brute force search (no dynamic programming)
    def bruteForce(self, order):

        cur_cost = np.inf
        cur_join = None

        for p in itertools.permutations(order):

            cost = self.leftDeepJoin(p)

            if cost < cur_cost:

                cur_join = p

                cur_cost = cost

            print("Join cost for ", p, cost)

        return cur_join, cur_cost


#creates a simulated set of tables with a single key attribute
j = JoinSimulator()

#specify a join as a list of table indices
join = [1, 4, 3, 0]

#find the best left-deep order for the join
print(j.bruteForce(join))
