from src.exception import Airlines_Exeption
from src.logger import logging
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformaton
from src.components.model_train import ModelTraining

@dataclass
class DataIngestionConfig:
    raw_data_config_path: str = os.path.join("artifacts","raw_data.csv")
    train_data_config_path: str = os.path.join("artifacts","train.csv")
    test_data_config_path: str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def InititateDataIngestion(self):
        try:
            logging.info("Data ingestion Initiated")
            logging.info("Reading the dataset")

            data = pd.read_csv('notebooks/data/airlines_flights_data.csv')
            logging.info("Creating artifact  directory to store datasets")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_config_path),exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_config_path,header=True,index=False)

            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_config_path,header=True,index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_config_path,header=True,index=False)

            logging.info("Data ingestion Completed")

            return (self.data_ingestion_config.train_data_config_path,self.data_ingestion_config.test_data_config_path)

        except Exception as e:
            raise Airlines_Exeption(e,sys)


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.InititateDataIngestion()
    data_transformer = DataTransformaton()
    x_train,y_train,x_test,y_test = data_transformer.InitiateDataTransformation(train_data,test_data)
    model_train = ModelTraining()
    best_model_score,best_model = model_train.InitiateModelTrain(x_train,y_train,x_test,y_test)
    print("Best Model: ",best_model)
    print("Model score: ",best_model_score)