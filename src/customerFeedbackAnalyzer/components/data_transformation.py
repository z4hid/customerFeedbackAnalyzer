import os
import re
import sys
import pickle
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
from src.customerFeedbackAnalyzer.utils import save_object
import numpy as np
import nltk

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

@dataclass
class DataTransformationConfig:
    count_vectorizer_path: str = os.path.join('artifacts', 'Vectorizer.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.stopwords_set = set(stopwords.words('english'))

    def stemming(self, text):
        port_stem = PorterStemmer()
        stemmed_text = re.sub('[^a-zA-Z]', ' ', text)
        stemmed_text = stemmed_text.lower()
        stemmed_text = stemmed_text.split()
        stemmed_text = [port_stem.stem(word) for word in stemmed_text if word not in self.stopwords_set]
        stemmed_text = ' '.join(stemmed_text)
        return stemmed_text

    def initiate_data_transformation(self, train_data, test_data):
        try:
            logging.info("Entered initiate_data_transformation method of DataTransformation class")
            
            # Read the data files into DataFrames
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            # Preprocess text data
            train_df['verified_reviews'] = train_df['verified_reviews'].apply(self.stemming)
            test_df['verified_reviews'] = test_df['verified_reviews'].apply(self.stemming)
            
            X_train = train_df['verified_reviews'].values
            X_test = test_df['verified_reviews'].values
            y_train = train_df['feedback'].values
            y_test = test_df['feedback'].values

            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            # Save vectorizer
            save_object(self.data_transformation_config.count_vectorizer_path, vectorizer)
            logging.info('TfidfVectorizer saved successfully')
            
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
