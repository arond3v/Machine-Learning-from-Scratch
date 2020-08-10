import numpy as np

class KNN :
    def _split_class(self,X,Y):
        no_of_par = X.shape[1]
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
            split[i] = split[i].reshape(-1,no_of_par)
        return split,class_type
    def _manhattan(self,X):
        result = []
        for i in range(X.shape[0]):
            cal = []
            k = 0
            for j in range(len(self.par)):
                cal.append(np.sort(np.abs(self.par[j] - X[i])))
                cal[j] = cal[j][:self.neighbor]
                cal[j] = cal[j].sum()
            for j in range(len(cal)):
                if cal[k]>cal[j]:
                    k = j
            result.append(self.class_type[k])
        return np.array(result)
    def _euclidean(self,X):       
        result = []
        for i in range(X.shape[0]):
            cal = []
            k = 0
            for j in range(len(self.par)):
                cal.append(np.sort(((self.par[j]-X[i])**2).sum(axis = 1)**0.5))
                cal[j] = cal[j][:self.neighbor]
                cal[j] = cal[j].sum()
            for j in range(len(cal)):
                if cal[k]>cal[j]:
                    k = j
            result.append(self.class_type[k])
        return np.array(result)
    def _minkowski(self,X):
        result = []
        for i in range(X.shape[0]):
            cal = []           
            k = 0
            for j in range(len(self.par)):
                cal.append(np.sort((np.abs(self.par[j]-X[i])**self.p).sum(axis = 1)**(1/self.p)))
                cal[j] = cal[j][:self.neighbor]
                cal[j] = cal[j].sum()
            for j in range(len(cal)):
                if cal[k]>cal[j]:
                    k = j
            result.append(self.class_type[k])
        return np.array(result)
    def fit(self,X,Y,neighbor = 5,metric = 'manhattan',p = None):
        self.par,self.class_type = self._split_class(X,Y)
        self.metric = metric
        self.p = p
        self.neighbor = neighbor
    def predict(self,X):
        if self.metric == "manhattan":
            pred = self._manhattan(X)
        elif self.metric == "euclidean":
            pred = self._euclidean(X)
        elif self.metric == "minkowski":
            pred = self._minkowski(X)
        return pred