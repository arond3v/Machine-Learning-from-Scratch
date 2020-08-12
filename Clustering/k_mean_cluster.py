import numpy as np 

class KMeanClustering:
    def __init__(self,n_cluster = 8,init = "k-mean++",max_iter = 300):
        self.n_cluster = n_cluster
        self.init = init
        self.max_iter = max_iter
    def _big(self,X):
        big = 0
        for i in range(X.shape[0]):
            if X[big]<X[i]:
                big = i
        return big
    def _probability(self,X,C):
        par = []
        a = np.array([])
        for i in range(len(C)):
            temp = (X - C[i])**2
            temp = temp.sum(axis = 1)
            par.append(temp)
        for i in range(X.shape[0]):
            j = 0
            for k in range(len(par)):
                if par[j][i]>par[k][i]:
                    j = k
            a = np.append(a,par[j][i])
        return self._big(a)
    def _kplusplus(self,X,k):
        centroid = []
        for _ in range(k):
            if len(centroid)>0:
                centroid.append(X[self._probability(X,centroid)])
            else:
                centroid.append(X[np.random.randint(X.shape[0])])
        return centroid
    def _split_cluster(self,X,Y):
        split = []
        class_type = []
        for i in range(X.shape[0]):
            added = False
            k = 0
            for j in class_type:
                if Y[i] == j:
                    split[k] = np.append(split[k],X[i])
                    added = True
                k += 1
            if added == False:
                class_type.append(Y[i])
                split.append(X[i])  
        for i in range(len(split)):
            split[i] = split[i].reshape(-1,X.shape[1])
        return split
    def _classify(self,X,C):
        temp = []
        a = np.array([])
        for i in range(self.n_cluster):
            temp.append(((X - C[i])**2).sum(axis = 1))
        for i in range(X.shape[0]):
            small = 0
            for j in range(len(temp)):
                if(temp[small][i]>temp[j][i]):
                    small = j
            a = np.append(a,small)
        return a
    def fit_predict(self,X):
        if self.init == 'k-mean++':
            centroid = self._kplusplus(X,self.n_cluster)
        elif self.init =='random':
            centroid = []
            for _ in range(self.n_cluster):
                centroid.append(np.random.rand(X.shape[1]))
        for _ in range(self.max_iter):
            cluster = self._split_cluster(X,self._classify(X,centroid))
            for j in range(self.n_cluster):
                centroid[j] = cluster[j].sum(axis = 0)*(1/cluster[j].shape[0])
        return cluster,centroid