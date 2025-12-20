import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import r2_score

from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from src.components.data_transformation import DataTransformationConfig

import pickle

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Split train and test data')
            X_train, y_train, X_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]

            models = {
                'Linear Regression': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forrest': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose = True),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'Gradient Boost': GradientBoostingRegressor()
            }

            model_report:dict = evaluate_model(X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            #best score
            best_score = max(list(model_report.values()))

            #best model name
            best_performing_model = list(model_report.keys())[
                list(model_report.values()).index(best_score)
                ]
            
            #best model as object
            final_model = models[best_performing_model]     
            
            if best_score < 0.6:
                raise CustomeException("No best Model Found")
            
            logging.info("Best model found on both train and test data")

            # open preprocessor.pkl file from file folder
            # file_path = os.path.join('artifact', 'preprocessor.pkl')
            # with open(file_path, 'r') as f:
            #     preprocessor = pickle.load(f)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = final_model
            )

            predicted = final_model.predict(X_test)

            score = r2_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomeException(e, sys)