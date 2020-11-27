# Opgave 1.6
# Cat growth curve: https://i.pinimg.com/originals/5c/49/a0/5c49a0eaa79b077c210c52a419232c4f.jpg
#X = np.array([0, 7, 14, 21, 28, 35, 42, 49, 56, 73, 80, 87, 94]).reshape(-1, 1)
#X = np.array([28, 35, 42, 49, 56, 63, 70, 77, 84]).reshape(-1, 1) # my cleanup
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from dateutil.parser import parse

from sklearn.linear_model import LinearRegression

# Du kan finde dokumentation for pyplot her: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html


#X = np.arange(0, 85, 7).reshape(-1, 1) # all
#X = np.arange(62, 85, 7).reshape(-1, 1) # 4
#X = np.arange(55, 85, 7).reshape(-1, 1) # 5
X = np.arange(27, 85, 7).reshape(-1, 1) # 9

print(X)


#y = np.array([550, 710, 730, 710, 685, 740, 880, 1155, 1207, 1425, 1575, 1750, 1890]) # all
#y = np.array([1425, 1575, 1750, 1890]) # 4
#y = np.array([1207, 1425, 1575, 1750, 1890]) # 5
y = np.array([685, 740, 880, 1155, 1207, 1425, 1575, 1750, 1890]) # 9

plt.plot(X,y, "b+")
# xmin, xmax, ymin, ymax : float
plt.axis([0, 200, 0, 3500])
plt.plot()

#training the model
lin_reg = LinearRegression()
lin_reg.fit(X,y)   # train the model on the data



predictDate = np.array([84+14]).reshape(-1, 1)
print("Cat weight 28/10: " + str(lin_reg.predict(predictDate)))

# y=a*x+b
# x=(y-b)/a
# https://www.symbolab.com/solver/solve-for-equation-calculator/solve%20for%20x%2C%20y%20%3D%20a%5Ccdot%20x%2Bb

days_estimated = (3500 - lin_reg.intercept_)/lin_reg.coef_

datetime = parse('2020-07-22 22:22:22')
estimated_date = datetime + timedelta(days=int(days_estimated))

print("Days estimated: " + str(days_estimated))
print("Date for 3.5kg: " + str(estimated_date.date()))

#calculating the score
score = lin_reg.score(X,y)
print("score "+str(score))
print("b "+str(lin_reg.intercept_))
print("a "+str(lin_reg.coef_))

new_x = np.array([[0], [200]])
y_predict = lin_reg.predict(new_x)

#print(new_x)

#plot the new data
plt.plot(new_x, y_predict, "r")

plt.show()

#how to get a and b parameters - i.e. the trained parameters for the model...?
#see documentation for help...


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

#How does this compare to the expected values for a and b? (Hint: Look at how the test data is defined)





