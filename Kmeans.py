__authors__ = [1600959, 1551813, 1603542, 1544112]
__group__ = 'GrupZZ'

import numpy as np
import utils
from copy import deepcopy

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self.TOLWCD = 20
        self._init_X(X)
        self._init_options(options)
        self._init_centroids()

    def _init_X(self, X):
        """
        Initialization of all pixels, sets X as an array of data in vector form (PxD)
        Args:
            X (list or np.array): list(matrix) of all pixel values
                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                the last dimension
        """
        newX = np.empty([X.shape[0]*X.shape[1],3])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                newX[i*X.shape[1]+j]=X[i][j]
        self.X = newX

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'wcd'  # within class distance.

        self.options = options


    def _init_centroids(self): #bé
        """
        Initialization of centroids
        """

        centr = np.zeros([self.K,3])
        if self.options['km_init'].lower() == 'first': #K primers diferents
            myList = [] #llista de centres apart
            amp = 0
            for i in range(self.K): #fem aixo K vegades
                flag=False #flag = True (el nou valor no està repe)
                while flag==False:
                    flag=True #reset de flag
                    newcentroid = self.X[i+amp] #agafem un nou centre
                    for j in range(len(myList)): #per cada valor de myList
                        if (newcentroid[0]== myList[j][0] and
                            newcentroid[1]== myList[j][1] and
                            newcentroid[2]== myList[j][2]):
                            flag=False #si es troba un valor igual
                    if flag==False:
                        amp+=1 #augmentem l'amp (mirem el seguent de self.X)
                myList.append(newcentroid) #si no està repe, l'afegim a myList
                centr[i] = newcentroid #l'afegim a centr
            self.centroids = centr
            self.old_centroids = centr #revisar

        elif self.options['km_init'].lower() == 'random':
            indexes = []
            for i in range(self.K):
                r = np.random.randint(self.X.shape[0])
                while r in indexes:
                    r = np.random.randint(self.X.shape[0])
                centr[i] = self.X[r]
                indexes.append(r)
            self.centroids = centr
            self.old_centroids = centr #revisar
            #revisar si és obligatori fer que els centroides valguin diferent
            #(no només index diferent sino valor diferent)

        elif self.options['km_init'].lower() == 'custom': #done
            a = []
            for i in range(self.K):
                z = 255*(i+1)/(self.K+1)
                a.append([z,z,z])
            self.centroids = np.array(a)
            self.old_centroids = np.array(a)
            #cal repartir els punts de forma equidistant per la diagonal
            #del cub de colors [0,0,0] a [255,255,255]

        elif self.options['km_init'].lower() == 'custom2':
            a = []
            k = self.K
            pi = np.pi
            phi = 0
            r = 80
            for i in range(k):
                x = np.round(r*np.cos(i*2*pi/k+phi)+122.5,2)
                y = np.round(r*np.sin(i*2*pi/k+phi)+122.5,2)
                z = np.round(-x-y+255+122.5,2)
                a.append([x,y,z])
            self.centroids = np.array(a)
            self.old_centroids = np.array(a)
            pass

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.labels = np.zeros(self.X.shape[0])
        dist = distance(self.X, self.centroids)
        for i in range(dist.shape[0]):
            point = dist[i]
            minimum_dist = point[0]
            closest_centroid = 0
            for j in range(dist.shape[1]):
                if point[j]<minimum_dist:
                    minimum_dist=point[j]
                    closest_centroid = j
            self.labels[i]=closest_centroid
        return self.labels


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = deepcopy(self.centroids)
        for index in range(self.K):
            MC = [0,0,0]
            count = 0
            for point,label in zip(self.X,self.labels):
                if label==index:
                    MC[0] += point[0]
                    MC[1] += point[1]
                    MC[2] += point[2]
                    count += 1
            if count>0:
                MC[0] /= count
                MC[1] /= count
                MC[2] /= count
                self.centroids[index]=np.round(MC,3)
            else:
                self.centroids[index]=np.round(self.old_centroids[index],3)
        return self.centroids

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        for centroid,old_centroid in zip(self.centroids,self.old_centroids):
            dist = 0
            for w1,w2 in zip(centroid,old_centroid):
                dist+=(w1-w2)*(w1-w2)
            dist=np.sqrt(dist)
            if dist>self.options['tolerance']:
                return False
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        iter = 0
        converged = False
        while iter<self.options['max_iter'] and not converged:
            self.get_labels()
            self.get_centroids()
            iter+=1
            converged = self.converges()
        pass

    def withinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        WCD = 0
        dist = distance(self.X,self.centroids)
        if(self.K == 2):
            pass
        for index in range(self.K):
            distclass=0
            N=0
            for i in range(self.X.shape[0]):
                N+=1
                if self.labels[i]==index:
                    dt = np.array([dist[i,index]])
                    distclass +=dist[i,index]*dt.T
            WCD+=distclass/N
        return WCD

    def externClassDistance(self):
        """
         returns the distance between classes of the current clustering
        """
        ECD = 0
        count = 1
        dist = distance(self.centroids,self.centroids)
        for i in range(self.K-1):
            for j in range(i+1,self.K):
                count+=1
                ECD+=dist[i][j]
        ECD/=count
        return ECD

    def WCD_ECD(self):
        WCD = self.withinClassDistance()
        ECD = self.externClassDistance()
        return np.sqrt(self.K)*WCD/(ECD)

    def silhouette(self):
        WCD = []
        dist = distance(self.X,self.centroids)
        for index in range(self.K):
            distclass=0
            N=0
            for i in range(self.X.shape[0]):
                N+=1
                if self.labels[i]==index:
                    dt = np.array([dist[i,index]])
                    distclass +=dist[i,index]*dt.T
            WCD.append(distclass/N)
        for i in range(self.K):
            dist[i][i]=100000
        s=0
        for centroid in range(self.K):
            centroid_wcd = WCD[centroid]
            closest_centroid_index = np.where(dist[centroid]==np.min(dist[centroid]))
            closest_centroid = self.centroids[closest_centroid_index[0]][0]
            centroid_ecd = 0
            for w in range(len(self.centroids[0])):
                centroid_ecd=np.power(self.centroids[centroid][w]-closest_centroid[w],2)
            s += (centroid_ecd-centroid_wcd)/max(centroid_ecd,centroid_wcd)
        s_mean = s/self.K
        return s_mean

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        self.K = 1
        self.fit()
        if(max_K<2):
            return
        if (self.options['fitting'].lower()=="wcd"):
            print("Option is wcd")
            old_value = self.withinClassDistance()
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.withinClassDistance()
                print("K: "+str(K)+". Old: "+str(old_value)+". New: "+str(new_value))
                if(new_value<old_value*0.30):
                    print("Best K is "+str(K))
                    return
                if(new_value>old_value*0.85):
                    print("Best K is "+str(K-1))
                    self.K=K-1
                    return
                old_value=new_value
        elif (self.options['fitting'].lower()=="ecd"):
            print("Option is ecd")
            old_value = self.externClassDistance()
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.externClassDistance()
                print("K: "+str(K)+". Old: "+str(old_value)+". New: "+str(new_value))
                if((new_value-old_value)/old_value<0.2):
                    print("Best K is "+str(K))
                    return
                if(K>2 and (new_value-old_value)/old_value>1):
                    print("Best K is "+str(K-1))
                    self.K=K-1
                    return
                old_value=new_value
        elif (self.options['fitting'].lower()=="wcd_ecd"):
            print("Option is wcd_ecd")
            old_value = 1
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.WCD_ECD()[0]
                if(new_value<old_value*0.5):
                    return
                if(K>2 and new_value>old_value*0.7):
                    self.K=K-1
                    return
                old_value=new_value
        elif (self.options['fitting'].lower()=="silhouette"):
            print("Option is silhouette")
            old_value = 0.86
            print(old_value)
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.silhouette()[0]
                if(new_value<old_value):
                    self.K=K-1
                    return
                if(new_value>0.965 or (new_value-old_value)<0.01):
                    return
                old_value=new_value
        elif (self.options['fitting'].lower()=="wcd_ecd_2"):
            print("Option is wcd_ecd")
            old_value = 1
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.WCD_ECD()/(np.sqrt(2)*K*K)
                if(new_value<1):
                    return
                old_value=new_value
        self.K=max_K
        return max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.zeros(shape=(len(X),len(C)))
    for i in range(len(X)):
        for j in range(len(C)):
            dist[i][j] = np.linalg.norm(X[i]-C[j])
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """
    return [utils.colors[np.argmax(k)] for k in utils.get_color_prob(centroids)]
