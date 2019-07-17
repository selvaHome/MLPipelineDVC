import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# load training data
dataTrain = np.loadtxt(open('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/data/data_processed/dataTrain.csv',"rb"), delimiter=",")

# build model
#model = LinearRegression().fit(dataTrain[:,:-1], dataTrain[:,-1])
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(dataTrain[:,:-1], dataTrain[:,-1])

# save model artifact
os.makedirs("/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/outputs", exist_ok = True)
with open('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/outputs/model.pkl', 'wb') as file:  
    pickle.dump(model, file)