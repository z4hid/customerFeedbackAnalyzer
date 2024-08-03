# Import necessary libraries
import os  # For file path operations
import re  # For regular expression operations
import sys  # For error handling
import pickle  # For saving objects as files
import pandas as pd  # For data manipulation
from dataclasses import dataclass  # For defining data classes
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from nltk.corpus import stopwords  # For stopwords
from nltk.stem import PorterStemmer  # For stemming

# Import custom modules
from src.customerFeedbackAnalyzer.exception import CustomException  # For custom exception handling
from src.customerFeedbackAnalyzer.logger import logging  # For logging
from src.customerFeedbackAnalyzer.utils import save_object  # For saving objects as files

# Import numpy and nltk
import numpy as np
import nltk

# Download stopwords for nltk
nltk.download('stopwords')

# Define stopwords set
STOPWORDS = set(stopwords.words('english'))

# Define data class for data transformation configuration
@dataclass
class DataTransformationConfig:
    count_vectorizer_path: str = os.path.join('artifacts', 'Vectorizer.pkl')  # Path to save vectorizer

# Define data transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Initialize data transformation configuration
        self.stopwords_set = set(stopwords.words('english'))  # Initialize stopwords set

    def stemming(self, text):
        """
        Stemming function to remove suffixes from words.

        Parameters:
        text (str): The text to be stemmed.

        Returns:
        str: The stemmed text.
        """
        port_stem = PorterStemmer()  # Initialize Porter stemmer
        stemmed_text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
        stemmed_text = stemmed_text.lower()  # Convert text to lowercase
        stemmed_text = stemmed_text.split()  # Split text into words
        stemmed_text = [port_stem.stem(word) for word in stemmed_text if word not in self.stopwords_set]  # Stem each word
        stemmed_text = ' '.join(stemmed_text)  # Join stemmed words into a string
        return stemmed_text

    def initiate_data_transformation(self, train_data, test_data):
        """
        Function to initiate the data transformation process.

        Parameters:
        train_data (str): Path to the training data file.
        test_data (str): Path to the testing data file.

        Returns:
        tuple: The transformed training and testing data, as well as the labels.
        """
        try:
            logging.info("Entered initiate_data_transformation method of DataTransformation class")
            
            # Read the data files into DataFrames
            train_df = pd.read_csv(train_data)  # Read training data
            test_df = pd.read_csv(test_data)  # Read testing data

            # Preprocess text data
            train_df['verified_reviews'] = train_df['verified_reviews'].apply(self.stemming)  # Apply stemming to training data
            test_df['verified_reviews'] = test_df['verified_reviews'].apply(self.stemming)  # Apply stemming to testing data
            
            X_train = train_df['verified_reviews'].values  # Extract training data
            X_test = test_df['verified_reviews'].values  # Extract testing data
            y_train = train_df['feedback'].values  # Extract training labels
            y_test = test_df['feedback'].values  # Extract testing labels

            vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer
            X_train = vectorizer.fit_transform(X_train)  # Transform training data
            X_test = vectorizer.transform(X_test)  # Transform testing data

            # Save vectorizer
            save_object(self.data_transformation_config.count_vectorizer_path, vectorizer)  # Save vectorizer
            logging.info('TfidfVectorizer saved successfully')  # Log successful saving of vectorizer
            
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)  # Raise custom exception if an error occurs

