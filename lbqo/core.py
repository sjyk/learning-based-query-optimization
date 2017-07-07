import numpy as np
import itertools
import copy


"""
Implements a set algebra where a relation is a set of keys.
This is a simplified case of equijoins
"""
class Relation(object):

    #defines a named relation with only keys
    def __init__(self, data, name, cost=0, provenance=[]):
        """
        data is a set of keys
        name is a string name for the relation
        cost is a running total of how much it costs to generate the relation
        provenance is how the relation is generated
        """

        self.data = data
        self.name = name
        self.size = len(data)
        self.cost = cost #cost of constructing this relation

        if provenance == []:
            self.provenance = [self]
        else:
            self.provenance = provenance

    #defines a simulated join operation
    def __mul__(self, other):
        return Relation(self.data.intersection(other.data), \
                        '('+self.name + ' * ' + other.name +')', \
                        cost= self.cost + other.cost + self.iocost(other),\
                        provenance = self.provenance + other.provenance) 

    #gives a crude io cost
    def iocost(self, *argv):
        cost = self.size 

        for other in argv:
            cost *= other.size

        return cost

    def reset(self):
        self.cost = 0

    def __str__(self):
        return self.name + " : " + str(self.cost)


    __repr__ = __str__


"""
Some global helper methods
"""

#joins the iterable of relations
def joinAll(relations):
    result = copy.copy(relations)
    join = result[0]
    for i in range(1, len(result)):
        join *= result[i]

    return join 


#recursive random sampler of join orders
def sampleRandomJoins(relations):
    
    relations = copy.copy(relations)

    if len(relations) == 2:
        return joinAll(relations)

    indices = np.random.choice(range(0, len(relations)),2, replace=False)

    left = relations[indices[0]]
    right = relations[indices[1]]

    join = left * right

    relations[indices[1]] = join

    relations.remove(left)

    return sampleRandomJoins(relations)