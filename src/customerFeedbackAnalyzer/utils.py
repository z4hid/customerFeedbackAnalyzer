import os
import sys
from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score

# Function to save an object to a file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Serialize and save the object to the specified file

    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate different machine learning models
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary to store model evaluation reports

        # Loop through each model and its corresponding parameters
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the parameters for the model

            gs = GridSearchCV(model, para, cv=3)  # Perform grid search cross-validation

            gs.fit(X_train, y_train)  # Fit the model

            model.set_params(**gs.best_params_)  # Set the best parameters found by grid search
            model.fit(X_train, y_train)  # Train the model with the best parameters

            y_train_pred = model.predict(X_train)  # Predict on the training data

            y_test_pred = model.predict(X_test)  # Predict on the test data

            train_model_score = accuracy_score(y_train, y_train_pred)  # Calculate training accuracy
            test_model_score = accuracy_score(y_test, y_test_pred)  # Calculate test accuracy

            report[list(models.keys())[i]] = test_model_score  # Store the test accuracy in the report

        return report  # Return the model evaluation report

    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs during evaluation
        raise CustomException(e, sys)
