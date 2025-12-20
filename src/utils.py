import os
import sys

import pandas as pd
import numpy as np

import dill

from src.exception import CustomeException

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models:dict):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]        #pull the model from models dict. eg LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = score      # add the name of the model as key and score as value in report dict

            return report
    except Exception as e:
        raise CustomeException(e, sys)