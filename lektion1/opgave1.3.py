# Opgave 1.3 - plots etc
#1.4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
print(X)

plt.plot(X,y, "b+")  # hvad betyder "b." ? Se dokumentationen for plot i pyplot linket ovenover
# b er farven . er figuren.
# og prøv at skifte til noget andet....
# xmin, xmax, ymin, ymax : float
plt.axis([0,2,0,15])  # betyder parameterne her?
plt.plot()

#training the model
lin_reg = LinearRegression()
lin_reg.fit(X,y)   # train the model on the data



#calculating the score
score = lin_reg.score(X,y)  # NOTICE THAT THE CLOSER TO 1 THE BETTER.
print("score "+str(score))
print("b"+str(lin_reg.intercept_))
print("a"+str(lin_reg.coef_))
#using the model on new data
new_x = np.array([[0],[2]])
y_predict = lin_reg.predict(new_x)

#plot the new data
plt.plot(new_x,y_predict,"r")

plt.show()

#how to get a and b parameters - i.e. the trained parameters for the model...?
#see documentation for help...




# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

#How does this compare to the expected values for a and b? (Hint: Look at how the test data is defined)





