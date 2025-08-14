from src.exception import Airlines_Exeption
from src.logger import logging
import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(file_path,file_obj):
    try:
        logging.info("pkl generation started")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open (file_path, "wb") as file:
            pickle.dump(file_obj,file)
        logging.info("pkl generation completed")
    except Exception as e:
        raise Airlines_Exeption(e,sys)

def transform_duration(data):
    try:
        data['duration'] = data['duration'].fillna(data['duration'].median()).astype(str)
        data['duration_hour'] = data['duration'].str.split(".").str[0].astype(int)
        data['duration_minute'] = data['duration'].str.split(".").str[1].astype(int)
        cleaned_data = data.drop('duration',axis=1)
        return cleaned_data
    except Exception as e:
        raise Airlines_Exeption(e,sys)

def Best_model_fittng(x_train,y_train,x_test,y_test,models):
    try:
        models_list = {}
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            ##param = params[list(models.keys())[i]]
            #grid = GridSearchCV(estimator=model,param_grid=param,cv=3)
            #grid.fit(x_train,y_train)
            #model.set_params(**grid.best_params_)
            model.fit(x_train,y_train)
            logging.info("predicting the model output")
            y_pred = model.predict(x_test)

            r2 = r2_score(y_test,y_pred)
            logging.info("Storing r2 score")
            models_list[list(models.keys())[i]] = r2

        return models_list
    except Exception as e:
        raise Airlines_Exeption(e,sys)

def load_data(file_path):
    try:
        with open(file_path,"rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Airlines_Exeption(e,sys)

def transform_duration_for_pipeline(data):
    try:
        data['duration_hour'] = data['duration'].str.split(".").str[0].astype(int)
        data['duration_minute'] = data['duration'].str.split(".").str[1].astype(int)
        cleaned_data = data.drop('duration',axis=1)
        return cleaned_data
    except Exception as e:
        raise Airlines_Exeption(e,sys)