import os
import sys
from src.exception import CustomeException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

import mysql.connector

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

    def _connect(self):
        # create MySQL connection
        return mysql.connector.connect(
            host = self.host,
            user = self.user,
            password = self.password,
            database = self.database
        )
    
    def load_data(self, query):
        conn = self._connect()  #activate connection
        logging.info('Connection established with sql server')
        try:
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            query = 'SELECT * FROM stud'
            df = self.load_data(query)
            logging.info('Data succesfully read from server and stored as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

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

