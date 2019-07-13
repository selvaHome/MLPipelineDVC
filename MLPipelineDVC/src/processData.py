import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt(open('./data/data.csv', "rb"), delimiter=",",skiprows=0)
xTrain, xTest, yTrain, yTest = train_test_split(np.hstack((data[:,[0]],data[:,[5]], data[:,[6]])), data[:,[1]], test_size = 1/3, random_state = 0)


np.savetxt("./data/dataTrain.csv", np.column_stack((xTrain,yTrain)), delimiter=",")
np.savetxt("./data/dataTest.csv", np.column_stack((xTest,yTest)), delimiter=",")
