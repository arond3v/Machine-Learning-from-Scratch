import numpy as np

class LogisticRegression:
    def _modify(self,X):
        ones = np.ones(X.shape[0]) 
        ones = ones.reshape(-1,1) 
        return np.append(ones,X,axis = 1)
    def _sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def fit(self,X,Y,no_of_iter = 10000,alpha = 0.001):
        X = self._modify(X)
        self.theta = np.random.randn(X.shape[1])
        for _ in range(no_of_iter):
            temp = self._sigmoid(np.matmul(X,self.theta))
            temp = temp - Y
            self.theta = self.theta-(alpha/X.shape[0])*(X.T.dot(temp))

    def predict(self,X):
        X = self._modify(X)
        X = self._sigmoid(np.matmul(X,self.theta))
        for i in range(X.shape[0]):
            if X[i] < 0.5:
                X[i] = 0
            else:
                X[i] = 1
        return X
    