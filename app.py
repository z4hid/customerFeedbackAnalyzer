from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
from src.customerFeedbackAnalyzer.components.data_ingestion import DataIngestion
import sys


        
if __name__=='__main__':
    
    logging.info('Data Ingestion Started')
    try:
        
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
    except Exception as e:
        logging.info('Custom Exception occured at Data Ingestion stage')
        raise CustomException(e, sys)
    
