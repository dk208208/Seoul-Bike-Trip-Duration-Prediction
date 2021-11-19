import joblib
import matplotlib as plt
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# Captures the path of current folder
curr_path = os.path.dirname(os.path.realpath(__file__))
print(curr_path)

feat_cols = ['Distance', 'Haversine', 'Pmin', 
'Dhour', 'Dmin', 'Temp', 'Humid', 'Solar', 'GroundTemp','Dust']

rf_final = joblib.load(curr_path + "/For_Modelling.joblib")


print(rf_final)
def predict_duration(attributes: np.ndarray):
    """ Returns Bike Trip Duration value"""

    pred = rf_final.predict(attributes)
    print("Duration predicted")

    return int(pred[0])