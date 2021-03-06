#
#  getData.py
#  Created by Selvakumar Ulaganathan, PhD on 18/07/19.
#

import os
import numpy as np

# skip the initial info. in the raw data file
data_actual = np.loadtxt(open("/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/data/data_raw.csv", "rb"), delimiter=",", skiprows=11)
os.makedirs("/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/data/data_actual", exist_ok = True)
np.savetxt("/Users/selva/Documents/linkedinPosts/rae2822Airfoil/MLPipelineDVC/data/data_actual/data_actual.csv", data_actual, delimiter=",")