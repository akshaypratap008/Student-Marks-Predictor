import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        function returns a preprocessor with data transformation pipelines
        '''
        try:
            num_feature = ['writing_score', 'reading_score']
            cat_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical Columns encoding completed')
            logging.info('Categorical Columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, num_feature),
                ('cat_pipeline', cat_pipeline, cat_feature)]
            )

            return preprocessor
        except Exception as e:
            raise CustomeException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'math_score'
            num_col = ['writing_score', 'reading_score']

            # train df
            input_feature_train_df = train_df.drop(columns = [target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            #test_df
            input_feature_test_df = test_df.drop(columns = [target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            #apply transformation on input features on train and test df
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            #create test_arr and train_arr by concatinating input features and target features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # save preprocessing object as pickle 
            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomeException(e, sys)
            