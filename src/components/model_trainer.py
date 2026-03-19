import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Spliting training and test data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Nearest Neighbors":KNeighborsRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "AdaBoost":AdaBoostRegressor()
            }

            params={
                "Random Forest":{
                    "n_estimators":[100,200,300],
                    "max_depth":[10,20,30],
                    "min_samples_split":[2,5,10],
                    "min_samples_leaf":[1,2,4]
                },
                "Decision Tree":{
                    "max_depth":[10,20,30],
                    "min_samples_split":[2,5,10],
                    "min_samples_leaf":[1,2,4]
                },
                "Gradient Boosting":{
                    "n_estimators":[100,200,300],
                    "max_depth":[10,20,30],
                    "min_samples_split":[2,5,10],
                    "min_samples_leaf":[1,2,4]
                },
                "K-Nearest Neighbors":{
                    "n_neighbors":[3,5,7,9],
                    "weights":["uniform","distance"],
                    "metric":["minkowski","euclidean","manhattan"]
                },
                "Linear Regression":{},
                "XGBoost":{
                    "n_estimators":[100,200,300],
                    "max_depth":[3,5,10],
                    "learning_rate":[0.01,0.1,0.3],
                    "min_child_weight":[1,3,5]
                },
                "CatBoost":{
                    "iterations":[100,200,300],
                    "depth":[4,6,10],
                    "learning_rate":[0.01,0.1,0.3]
                },
                "AdaBoost":{
                    "n_estimators":[100,200,300],
                    "learning_rate":[0.01,0.1,1]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_sq=r2_score(y_test,predicted)
            return r2_sq

        except Exception as e:
            raise CustomException(e,sys)