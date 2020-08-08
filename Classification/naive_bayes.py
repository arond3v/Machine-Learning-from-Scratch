import numpy as np

class NB:
    
    def split_class(self,X,Y):
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
    def fit(self,X,Y):
        data,self.class_type = self.split_class(X,Y)
        self.standard_deviation = []
        self.probability = []
        deviation = 0
        for i in range(len(data)):
            self.probability.append(data[i].shape[0]/X.shape[0])
            temp = np.mean(data[i])
            for j in range(data[i].shape[0]):
                deviation += (data[i][j]-temp)**2
            self.standard_deviation.append(((1/data[i].shape[0])*(deviation))**0.5)
            data[i] = temp  
        self.mean = data
    def predict(self,X):
        temp = []
        pred = []
        for i in range(len(self.class_type)):
            temp.append((1/((2**0.5)*self.standard_deviation[i]))*np.exp((-1/2)*(((X-self.mean[i])/self.standard_deviation[i])**2)))
            temp[i] = temp[i].prod(axis = 1)*self.probability[i]
        for i in range(X.shape[0]):
            k = 0
            for j in range(len(self.class_type)):
                if (temp[k][i]<temp[j][i]):
                    k = j
            pred.append(self.class_type[k])
        return np.array(pred)