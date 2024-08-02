import os
import sys
from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        """
        Initializes a new instance of the DataIngestion class.

        This constructor initializes the `ingestion_config` attribute with a new instance of the `DataIngestionConfig` class.

        Parameters:
            None

        Returns:
            None
        """
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process by reading a CSV file, dropping rows with missing values, and splitting the data into training and testing sets.

        Returns:
            tuple: A tuple containing the paths to the training and testing data files.

        Raises:
            CustomException: If an error occurs during the data ingestion process.
        """
        logging.info('Data ingestion method started')
        try:
            df = pd.read_csv(r'artifacts/data.tsv', delimiter = '\t', quoting = 3)
            logging.info('Read dataset as pandas DataFrame')
            
            df[df['verified_reviews'].isna() == True]
            df.dropna(inplace=True)
        
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Initiating Train Test Split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=='__main__':
    
    logging.info('Data Ingestion Started')
    try:
        
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
    except Exception as e:
        logging.info('Custom Exception occured at Data Ingestion stage')
        raise CustomException(e, sys)
    
