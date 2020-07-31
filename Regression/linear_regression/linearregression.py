import numpy as np
class LinearRegression:
  
    def modify(self,X):
        # y = m*x + c is y = m*x +c*1 so a column of ones is needed 
        ones = np.ones(X.shape[0]) # just X.shape could be used and 
        ones = ones.reshape(-1,1) # remove the next line  if it's for linear regression but it's gonna be used for other regression
        return np.append(ones,X,axis = 1)

    def fit(self,X,Y,no_of_iter = 1000,alpha = 0.001,):
        X = self.modify(X)
        self.theta = np.random.randn(X.shape[1])
        # gradient descent 
        # check .txt for the formula its simplified here 
        for _ in range(no_of_iter):  
            temp = np.matmul(X,self.theta)
            temp = temp - Y
            self.theta = self.theta-(alpha/X.shape[0])*(X.T.dot(temp))

    def predict(self,X):
        X = self.modify(X)
        return np.matmul(X,self.theta)
