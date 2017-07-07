import numpy as np
import itertools
import copy

"""
LBQO implements a techniqe that samples through possible query
plans and generates training data.
"""
class LBQO(object):

    def __init__(self, relations, universe):
        """
        Given a list of relations and a universe of names
        """
        self.relations = relations
        self.universe = universe

    #this function bit-encodes the relation's provenance
    def getFeatures(self, relation):
        feat = np.zeros((1,len(self.universe)))
        for r in relation.provenance:

            #if the key doesn't exist 0
            try:
                i = self.universe.index(r.name)
                feat[0,i] = 1
            except:
                pass

        return feat


    #recursive random sampler of join orders
    def generateOneSample(self, relations, traj=[]):
        
        relations = copy.copy(relations)

        if len(relations) == 1:

            N = len(traj)
            X = np.zeros((N, traj[0][0].shape[1]*2))
            Y = np.ones((N,1))*relations[0].cost

            for i, t in enumerate(traj):
                X[i,:] = np.hstack(t)

            return X,Y

        indices = np.random.choice(range(0, len(relations)),2, replace=False)

        left = relations[indices[0]]
        right = relations[indices[1]]

        join = left * right

        relations[indices[1]] = join

        relations.remove(left)

        nextStep = traj + [(self.getFeatures(left), self.getFeatures(right)),\
                             (self.getFeatures(right), self.getFeatures(left))]

        return self.generateOneSample(relations, nextStep)


    def generateTrainingData(self, N=1000):

        X, Y = self.generateOneSample(self.relations)

        for i in range(N-1):
            Xn, Yn = self.generateOneSample(self.relations)
            X = np.vstack((X,Xn))
            Y = np.vstack((Y, Yn)) 

        return X,Y


    #given a function approximator generates a policy from N samples
    def train(self, fapprox, N=1000):
        X, Y = self.generateTrainingData(N)
        fapprox.fit(X,np.ravel(Y))
        self.fapprox = fapprox


    #applies the policy to select the joins
    def optimize(self, relations):
        
        relations = copy.copy(relations)
        policy = self.fapprox


        if len(relations) == 1:
            return relations[0]


        scores = []

        for i in range(0, len(relations)):
            for j in range(0, len(relations)):

                if i != j:

                    left = relations[i]
                    right = relations[j]

                    feature = np.hstack((self.getFeatures(left), self.getFeatures(right)))

                    scores.append((policy.predict(feature)[0],i,j))


        scores.sort()

        indices = scores[0][1:3]

        left = relations[indices[0]]
        right = relations[indices[1]]

        join = left * right

        relations[indices[1]] = join

        relations.remove(left)

        return self.optimize(relations)













