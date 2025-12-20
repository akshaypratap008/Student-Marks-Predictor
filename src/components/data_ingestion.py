import os
import sys
from src.exception import CustomeException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from sqlalchemy import create_engine

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact', 'train.csv')
    test_data_path:str = os.path.join('artifact', 'test.csv')
    raw_data_path:str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self, host, user, password, database):
        self.ingestion_config = DataIngestionConfig()
        self.host = host
        self.user = user
        self.password = password
        self.database = database

        #create sql server connection url
        self.conn_url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    def load_data(self, query):
        # create MySQL connection
        try:
            engine = create_engine(self.conn_url)
            logging.info('Sql server connection created ')

            df = pd.read_sql(query, engine)
            logging.info('Data loaded succesfully from mysql')

            return df
        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            query = 'SELECT * FROM stud'
            df = self.load_data(query)
            logging.info('Data succesfully read from server and stored as dataframe')

            # save raw data            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            #train test split
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header=True)

            logging.info('Data Ingestion completed')

            return(
                # self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomeException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion(
        host = '127.0.0.1',
        user = 'root',
        password = '',
        database= 'mlproject1'
    )
    obj.initiate_data_ingestion()

