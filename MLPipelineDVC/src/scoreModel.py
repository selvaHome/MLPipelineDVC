import pickle 
import numpy as np
import json
import matplotlib.pyplot as plt

# Load from file
with open('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/outputs/model.pkl', 'rb') as file:  
    model = pickle.load(file)

# score model on test dataset
dataTest = np.loadtxt(open('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/data/data_processed/dataTest.csv',"rb"), delimiter=",")
score = model.score(dataTest[:,:-1], dataTest[:,-1])  
score_json = {"score": score}
with open('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/outputs/score.json', 'w') as fp:
    json.dump(score_json, fp)

# correlation plot
plt.scatter(dataTest[:,-1], model.predict(dataTest[:,:-1]), color='blue', linewidth=3, label='co-efficient of lift')
plt.legend(loc='upper left')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig('/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/outputs/cl.png')


