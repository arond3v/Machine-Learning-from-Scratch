For Linear Multiple and Polynomial Regression

Cost_Funcion = (1/N)*sum(Y_pred - Y_i) //for i in N
partial derivation of Cost_funtion with coefficient = (2/N)*sum((Y_pred - Y_i)*coefficient)


1 x1	transpose	x1 x2 x3 x4 	dot	y1_pred - y1	x1*(y1_pred - y1)+x2*(y2_pred - y)+x3*(y3pred - y3)+x4*(y4_pred - y4)
1 x2	    		1  1  1  1	    	y2_pred - y2	 1*(y1_pred - y1)+1 *(y2_pred - y)+1 *(y3pred - y3)+1 *(y4_pred - y4)
1 x3	         				    	y3_pred - y3
1 x4		        			    	y4_pred - y4
