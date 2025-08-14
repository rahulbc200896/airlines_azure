import sys
import os
from src.logger import logging
from src.exception import Airlines_Exeption
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from dataclasses import dataclass
from src.utils import save_obj,Best_model_fittng

@dataclass
class ModelTrainConfig:
    model_config_file_path: str = os.path.join("artifacts","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_train = ModelTrainConfig()

    def InitiateModelTrain(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("Assigning x_train,y_train,x_test,y_test values")
            
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test

            logging.info("Listing the models to perform model training")
            models = {
                    #"LinearRegression": LinearRegression(),
                    #"Lasso": Lasso(),
                    #"Ridge": Ridge(),
                    #"KNeighborsRegressor": KNeighborsRegressor(),
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    #"RandomForestRegressor": RandomForestRegressor(),
                    #"AdaBoostRegressor": AdaBoostRegressor(),
                    #"GradientBoostingRegressor": GradientBoostingRegressor(),
                    #"XGBRegressor": XGBRegressor()
                    }

            logging.info("Initialising params to perform Hyper parameter tuning")
            '''params = {
                "KNeighborsRegressor":{},
                "DecisionTreeRegressor":{
                    "criterion": ["squared_error","frieman_mse","absolute_eror","poisson"]
                },
                "RandomForestRegressor":{
                    #"criterion": ["squared_error","frieman_mse","absolute_eror","poisson"],
                    "n_estimators": [8,16,32]
                },
                "AdaBoostRegressor":{},
                "LinearRegression":{},
                "Lasso":{},
                "Ridge":{},
                "GradientBoostingRegressor":{
                    #"loss": ["squared_error","huber","absolute_eror","quantile"],
                    "n_estimators": [8,16,32],
                    "learning_rate": [0.1,0.01,0.5]
                },
                "XGBRegressor":{}
            }'''
            logging.info("Evaluation of model started")

            Model_Report:dict = Best_model_fittng(
                self.x_train,
                self.y_train,
                self.x_test,
                self.y_test,
                models
            )

            logging.info("checking and storing the best model and score")

            best_model_score = max(sorted(Model_Report.values()))
            best_model_name = list(Model_Report.keys())[list(Model_Report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Airlines_Exeption("Best model Not found")
            else:
                save_obj(self.model_train.model_config_file_path,best_model)
            logging.info("model training completed")
            return (best_model_score,best_model)

        except Exception as e:
            raise Airlines_Exeption(e,sys)