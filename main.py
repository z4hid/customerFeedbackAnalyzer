from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
from src.customerFeedbackAnalyzer.components.data_ingestion import DataIngestion
from src.customerFeedbackAnalyzer.components.data_transformation import DataTransformation, DataTransformationConfig
from src.customerFeedbackAnalyzer.components.model_trainer import ModelTrainer, ModelTrainerConfig
import sys
import numpy as np

if __name__ == '__main__':

    logging.info('Data Ingestion Started')
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info('Data Ingestion Finished')

        logging.info('Data Transformation Started')
        data_transform = DataTransformation()

        # Correctly handle the four returned values
        X_train, X_test, y_train, y_test = data_transform.initiate_data_transformation(train_data, test_data)
        print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
        logging.info('Data Transformation Finished')

        # Model Training
        logging.info('Model Training Started')
        model_trainer = ModelTrainer()

        # Train the model with the correct arguments
        accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        print(f'Accuracy Score is: {accuracy}')
        logging.info('Model Training Finished')

    except Exception as e:
        logging.info('Custom Exception occurred')
        raise CustomException(e, sys)