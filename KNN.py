__authors__ = [1600959, 1551813, 1603542, 1544112]
__group__ = 'GrupZZ'

import numpy as np
import math
import operator
from copy import deepcopy
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        P=train_data.shape[0]
        M=train_data.shape[1]
        N=train_data.shape[2]
        D=train_data.shape[3]

        self.train_data = train_data.reshape((P,N*M*D))


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #Matriu de distàncies creuades
        P=test_data.shape[0]
        M=test_data.shape[1]
        N=test_data.shape[2]
        D=test_data.shape[3]

        test_data = test_data.reshape((P,N*M*D))

        distMatrix = cdist(test_data, self.train_data)
        neighborsMatrix = np.empty([P,k], dtype = object)
        dist = np.argsort(distMatrix,axis=1)

        for i in range(P):
            fila=dist[i][0:k]
            ar = [self.labels[j] for j in fila]
            neighborsMatrix[i]=ar

        self.neighbors = neighborsMatrix


    def get_class(self,voting_info=False):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        r1 = []
        r2 = []
        for i in range(self.neighbors.shape[0]):
            listaUnics,listaIndex,listaCompte = np.unique(self.neighbors[i],return_inverse=True,return_counts=True)
            #for debug purposes only
            if(voting_info):
                print(listaUnics,listaIndex,listaCompte)
            compteBo = listaCompte[listaIndex]
            maxim=np.max(compteBo)
            index = compteBo.argmax(0)
            r1.append(self.neighbors[i][index])
            r2.append(compteBo[index]/self.neighbors.shape[1]*100) #GumerNota: Percentatge?
        return r1

    def predict(self, test_data, K, voting_info):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:           the number of neighbors to look at
        :param voting_info: option to return info about the voting percentages
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, K)
        r1 = self.get_class(voting_info)
        return r1
