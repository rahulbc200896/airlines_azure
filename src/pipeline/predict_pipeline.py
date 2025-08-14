from src.exception import Airlines_Exeption
from src.logger import logging
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.utils import load_data
import pandas as pd
from src.utils import transform_duration_for_pipeline

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict_data(self,data):
        try:

            scaler_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"
            preprocessor = load_data(scaler_path)
            model = load_data(model_path)
            transformed_data = transform_duration_for_pipeline(data)
            scaled_data = preprocessor.transform(transformed_data)
            pred = model.predict(scaled_data)

            return pred
        except Exception as e:
            raise Airlines_Exeption(e,sys)

class Custom_data:
    def __init__(self,days_left:int, 
    duration:str,
    airline:str, 
    source_city:str, 
    departure_time:str, 
    stops:str, 
    arrival_time:str, 
    destination_city:str, 
    classs:str):
        self.days_left = days_left
        self.duration = duration
        self.airline = airline
        self.source_city = source_city
        self.departure_time = departure_time
        self.stops = stops
        self.arrival_time = arrival_time
        self.destination_city = destination_city
        self.classs = classs

    def get_data_as_dataframe(self):
        try:
            data = {
                "days_left":[self.days_left],
                "duration":[self.duration],
                "airline":[self.airline],
                "source_city":[self.source_city],
                "departure_time":[self.departure_time],
                "stops":[self.stops],
                "arrival_time":[self.arrival_time],
                "destination_city":[self.destination_city],
                "class":[self.classs]
            }

            return pd.DataFrame(data)
        except Exception as e:
            raise Airlines_Exeption(e,sys)


