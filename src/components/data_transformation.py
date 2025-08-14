from src.exception import Airlines_Exeption
from src.logger import logging
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_obj,transform_duration

@dataclass
class DataTransformConfig:
    data_preprocessor_file_path: str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformaton:
    def __init__(self):
        self.data_transform = DataTransformConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Entered Data Transformaion")
            logging.info("Separating the numerical and categorical features")
            num_features = ['days_left', 'duration_hour','duration_minute']
            cat_features = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

            logging.info("Creating pipelines to perform transfromation")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one hot",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer([
                ("Numerical features",num_pipeline,num_features),
                ("Categorical Features",cat_pipeline,cat_features)
            ])

            return  preprocessor
        except Exception as e:
            raise Airlines_Exeption(e,sys)

    def InitiateDataTransformation(self,train_data,test_data):
        try:
            logging.info("Initiated Data Transformaion")
            self.data_transformer = self.get_data_transformation_object()

            logging.info("Reading the train and test data")
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            target_feature = 'price'

            train_input_data = train_df.drop(columns=target_feature,axis=1)
            print(train_input_data.columns)
            print("Before transform")
            print(train_input_data.shape)
            train_target_data = train_df[target_feature]
            print(train_target_data.shape)

            test_input_data = test_df.drop(columns=target_feature,axis=1)
            print(test_input_data.shape)
            test_target_data = test_df[target_feature]
            print(test_target_data.shape)

            train_input = transform_duration(train_input_data)
            test_input = transform_duration(test_input_data)

            logging.info("Data Transformaion started")
            train_input_arr = self.data_transformer.fit_transform(train_input)
            test_input_arr = self.data_transformer.transform(test_input)
            print("After transform")
            print(train_input_arr.shape)
            print(train_target_data.shape)
            print(test_input_arr.shape)
            print(test_target_data.shape)

            train_target_arr = np.array(train_target_data)
            test_target_arr = np.array(test_target_data)

           # train_arr = np.c_[train_input_arr,train_target_data]
            #test_arr = np.c_[test_input_arr,test_target_data]

            logging.info("Data Transformaion Completed")

            save_obj(self.data_transform.data_preprocessor_file_path,self.data_transformer)

            return (train_input_arr,train_target_arr,test_input_arr,test_target_arr)
        except Exception as e:
            raise Airlines_Exeption(e,sys)
