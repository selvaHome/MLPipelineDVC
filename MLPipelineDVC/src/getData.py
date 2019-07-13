import os
import numpy as np
import urllib.request

# get data from original source
urllib.request.urlretrieve(url = "http://airfoiltools.com/polar/csv?polar=xf-rae2822-il-50000", 
                           filename = "./data/data.csv")

# skip the initial info. in the data file
data_actual = np.loadtxt(open("./data/data.csv", "rb"), delimiter=",", skiprows=11)
np.savetxt("./data/data.csv", data_actual, delimiter=",")