import os
import sys
from dataclasses import dataclass
from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
from src.customerFeedbackAnalyzer.utils import save_object, evaluate_models
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        """
        Initializes a new instance of the ModelTrainer class.

        This constructor initializes the `model_trainer_config` attribute with a new instance of the `ModelTrainerConfig` class.

        Parameters:
            None

        Returns:
            None
        """
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        """
        This function trains a model and evaluates it on the test data
        
        Parameters:
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The testing features.
        y_test (np.ndarray): The testing labels.
        
        Returns:
        float: The accuracy score of the best model on the test data.
        """
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            # x_train, y_train, x_test, y_test = (
            #     train_array[:,:-1],
            #     train_array[:,-1],
            #     test_array[:,:-1],
            #     test_array[:,-1]
            # )
            models = {
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                # "XGBClassifier": XGBClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                # "AdaBoostClassifier": AdaBoostClassifier(),
                # "GradientBoostingClassifier": GradientBoostingClassifier()
            }
            
            params = {
                "DecisionTreeClassifier": {"criterion": ["gini", "entropy"], "splitter": ["best", "random"]},
                # "XGBClassifier": {
                #     "learning_rate": [0.5, 0.1, 0.01, 0.001],
                #     "max_depth": [3, 5, 10, 20],
                #     "n_estimators": [10, 50, 100, 200],
                # },
                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [3, 5, 10, 20],
                    "criterion": ["gini", "entropy"],
                },
                # "AdaBoostClassifier": {
                #     "learning_rate": [0.5, 0.1, 0.01, 0.001],
                #     "n_estimators": [10, 50, 100, 200],
                # },
                # "GradientBoostingClassifier": {
                #     "learning_rate": [0.5, 0.1, 0.01, 0.001],
                #     "n_estimators": [10, 50, 100, 200],
                # }   
            }
            
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, params)
            # Get best model score
            best_model_score = max(sorted(model_report.values()))
            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found!')
            logging.info('Best model found on both training and testing dataset')
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            return acc_score
        
        except Exception as e:
            raise CustomException(e, sys)
            