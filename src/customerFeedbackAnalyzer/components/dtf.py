import os
import re
import sys
import pickle
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.customerFeedbackAnalyzer.exception import CustomException
from src.customerFeedbackAnalyzer.logger import logging
from src.customerFeedbackAnalyzer.utils import save_object

@dataclass
class TextDataTransformationConfig:
    count_vectorizer_path: str = os.path.join('artifacts', 'countVectorizer.pkl')
    scaler_path: str = os.path.join('artifacts', 'scaler.pkl')

class TextDataTransformation:
    def __init__(self):
        self.config = TextDataTransformationConfig()
        self.stemmer = PorterStemmer()
        self.stopwords_set = set(stopwords.words('english'))

    def preprocess_text(self, data: pd.DataFrame, text_column='verified_reviews') -> list:
        """
        Preprocess text data by cleaning, stemming, and removing stopwords.
        
        Parameters:
        data (pd.DataFrame): The dataframe containing the text data.
        text_column (str): The name of the column containing the text to preprocess.
        
        Returns:
        list: A list of preprocessed text data.
        """
        corpus = []
        for i in range(data.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', data.iloc[i][text_column])
            review = review.lower().split()
            review = [self.stemmer.stem(word) for word in review if word not in self.stopwords_set]
            review = ' '.join(review)
            corpus.append(review)
        return corpus

    def vectorize_text(self, corpus_train: list, corpus_test: list, max_features: int = 2500):
        """
        Vectorize the preprocessed text data using CountVectorizer.
        
        Parameters:
        corpus_train (list): The list of preprocessed training text data.
        corpus_test (list): The list of preprocessed testing text data.
        max_features (int): The maximum number of features for the vectorizer.
        
        Returns:
        np.ndarray: Array of vectorized training text data.
        np.ndarray: Array of vectorized testing text data.
        CountVectorizer: The fitted CountVectorizer object.
        """
        cv = CountVectorizer(max_features=max_features)
        X_train = cv.fit_transform(corpus_train).toarray()
        X_test = cv.transform(corpus_test).toarray()
        
        # Save the vectorizer
        save_object(self.config.count_vectorizer_path, cv)
        
        return X_train, X_test, cv

    def scale_features(self, X_train, X_test):
        """
        Scale the features using MinMaxScaler.
        
        Parameters:
        X_train (np.ndarray): The training features.
        X_test (np.ndarray): The testing features.
        
        Returns:
        np.ndarray: The scaled training features.
        np.ndarray: The scaled testing features.
        MinMaxScaler: The fitted scaler object.
        """
        scaler = MinMaxScaler()
        X_train_scl = scaler.fit_transform(X_train)
        X_test_scl = scaler.transform(X_test)
        
        # Save the scaler
        save_object(self.config.scaler_path, scaler)
        
        return X_train_scl, X_test_scl, scaler

    def initiate_data_transformation(self, train_path: str, test_path: str, text_column ='verified_reviews', target_column='feedback'):
        """
        Orchestrates the entire data transformation process.
        
        Parameters:
        train_path (str): Path to the training CSV file containing the data.
        test_path (str): Path to the testing CSV file containing the data.
        text_column (str): Name of the column containing the text data.
        target_column (str): Name of the column containing the target variable.
        
        Returns:
        tuple: The processed training and testing data arrays and the paths of saved vectorizer and scaler.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            # Preprocess text
            corpus_train = self.preprocess_text(train_df, text_column)
            corpus_test = self.preprocess_text(test_df, text_column)
            
            # Vectorize text
            X_train, X_test, cv = self.vectorize_text(corpus_train, corpus_test)
            
            # Extract target variable
            y_train = train_df[target_column].values
            y_test = test_df[target_column].values
            
            # Scale features
            X_train_scl, X_test_scl, scaler = self.scale_features(X_train, X_test)
            
            logging.info("Data transformation completed")

            return X_train_scl, X_test_scl, y_train, y_test, self.config.count_vectorizer_path, self.config.scaler_path

        except Exception as e:
            raise CustomException(e, sys)


# # Example usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     data_transformation = TextDataTransformation()
#     train_path = "path/to/your/train.csv"
#     test_path = "path/to/your/test.csv"
#     text_column = "verified_reviews"
#     target_column = "feedback"
#     X_train_scl, X_test_scl, y_train, y_test, cv_path, scaler_path = data_transformation.initiate_data_transformation(train_path, test_path, text_column, target_column)
