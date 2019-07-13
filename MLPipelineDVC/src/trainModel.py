import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm

dataTrain = np.loadtxt(open('./data/dataTrain.csv',"rb"), delimiter=",")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(dataTrain[:,:-1], dataTrain[:,-1])

with open('./outputs/model.pkl', 'wb') as file:  
    pickle.dump(model, file)
