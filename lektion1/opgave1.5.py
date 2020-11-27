# Opgave 1.3 - plots etc
#1.4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor

# Du kan finde dokumentation for pyplot her: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html

X = 2 * np.random.rand(100, 1)  # hvad betyder de to parametre 100 og 1?
# Gernere et array som er 1*100 med random tal mellem 0 og 2

y = 4 + 3 * X + np.random.randn(100, 1) # så hvilke parametre ville vi forvente modellen har?
# Y = a*x+b
# Y = 3*x+4
# hvad er forskellen mellem rand og randn? (du må se om du kan finde dokumentationen selv....)
# randn giver en normalt fordelt distribution rand giver en uniform distriubtion
# print(X)

#X = np.append(X,[[ 10 ]], axis = 0 )
#y = np.append(y,[[ 5 ]], axis = 0 )
#print(X)

plt.plot(X,y, "b+")  # hvad betyder "b." ? Se dokumentationen for plot i pyplot linket ovenover
# b er farven . er figuren.
# og prøv at skifte til noget andet....
# xmin, xmax, ymin, ymax : float
plt.axis([0,2,0,15])  # betyder parameterne her?
plt.plot()



# max_iter 1000 eta 0.1
# score 0.6871200884899488
# b[4.01085192]
# a[2.97814832]

# max_iter 100 eta 0.1
# score 0.8138206526488347
# b[3.86377972]
# a[3.13194638]

# max_iter 10 eta 0.1
# score 0.7193300049551374
# b[3.78363797]
# a[3.09306132]

# max_iter 100 eta 1
# score 0.4946541860947906
# b[3.52928606]
# a[2.52573331]

# max_iter 100 eta 5
# score -0.1172425307236995
# b[5.26696758]
# a[3.22587283]


#training the model
sgd_reg = SGDRegressor(max_iter=100, tol=1e-3, penalty=None, eta0=5)
sgd_reg.fit(X, y.ravel())



#calculating the score
score = sgd_reg.score(X,y)  # NOTICE THAT THE CLOSER TO 1 THE BETTER.
print("score "+str(score))
print("b"+str(sgd_reg.intercept_))
print("a"+str(sgd_reg.coef_))
#using the model on new data
new_x = np.array([[0],[2]])
y_predict = sgd_reg.predict(new_x)

#plot the new data
plt.plot(new_x,y_predict,"r")

plt.show()

#how to get a and b parameters - i.e. the trained parameters for the model...?
#see documentation for help...




# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

#How does this compare to the expected values for a and b? (Hint: Look at how the test data is defined)





