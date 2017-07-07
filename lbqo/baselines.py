from core import *

import itertools
import copy

#baseline: optimizes with a left-deep heuristic (dynamic program)
def sellingerOptimizerFullCost(relations):
    optjoin = {}
    optjoin[()] = (0, [])


    for i in range(1,len(relations)+1):

        #for all subsets S of size i
        for comb in itertools.combinations(relations, i):

            bestlcost = np.inf
            bestltuple = ()
            bestr = None

            #optimize the one-step join with S - a
            for r in comb:

                comblist = list(comb)

                comblist.remove(r)


                if len(comblist) > 1:
                    localcost = optjoin[tuple(comblist)][0] + joinAll(list(comb)).cost #assume full knowledge of join cardinality
                else:
                    localcost = optjoin[tuple(comblist)][0] + r.size


                if localcost < bestlcost:
                    bestlcost = localcost
                    bestltuple = tuple(comblist)
                    bestr = optjoin[tuple(comblist)][1]  + [r] 

            #cache for the next iteration
            optjoin[comb] = (bestlcost, bestr)

    return joinAll(optjoin[comb][1])


#baseline: optimizes with a left-deep heuristic (dynamic program)
def sellingerOptimizerEstimatedCost(relations):
    optjoin = {}
    optjoin[()] = (0, [])

    for i in range(1,len(relations)+1):

        for comb in itertools.combinations(relations, i):

            bestlcost = np.inf
            bestltuple = ()
            bestr = None

            for r in comb:

                comblist = list(comb)

                comblist.remove(r)


                if len(comblist) > 1:
                    localcost = optjoin[tuple(comblist)][0] + r.iocost(*comblist) #assume estimated join cardinality
                else:
                    localcost = optjoin[tuple(comblist)][0] + r.size


                if localcost < bestlcost:
                    bestlcost = localcost
                    bestltuple = tuple(comblist)
                    bestr = optjoin[tuple(comblist)][1]  + [r] 

            optjoin[comb] = (bestlcost, bestr)

    return joinAll(optjoin[comb][1])


#baseline no opt
def identityOptimizer(relations):
    return joinAll(relations) 


#baseline re-write, orders by smallest relation first
def rewriteOptimizer(relations):
    return joinAll(sorted(relations, key = lambda r: r.size)) 


def bruteForceOptimizer(relations, N=1000):
    results = []
    for i in range(N):
        results.append(sampleRandomJoins(relations))

    #print(results)

    return sorted(results, key = lambda r: r.cost)[0]