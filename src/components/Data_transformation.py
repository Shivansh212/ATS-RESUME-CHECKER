import sys
import os
import re
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import customException
from src.logger import logging
from src.utils import save_object
import spacy

# Load the spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("Loaded spaCy 'en_core_web_sm' model")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found.")
    logging.info("Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)


def spacy_tokenizer(text):
    """
    Custom tokenizer using spaCy for lemmatization, 
    stop-word removal, and punctuation removal.
    Returns a list of clean lemma tokens.
    """
    try:
        doc = nlp(str(text).lower())
        lemmas = []
        for token in doc:
            if (not token.is_stop and
                not token.is_punct and
                token.text.strip() and
                not token.is_space):
                
                lemmas.append(token.lemma_)
        
        
        return " ".join(lemmas) 
    except Exception as e:
        logging.error(f"Error in spacy_tokenizer: {e}")
        return ""


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        logging.info("DataTransformation component initialized")

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation object.
        """
        try:
            logging.info("Creating TF-IDF Vectorizer object")
            
            tfidf_vectorizer = TfidfVectorizer(
                
                preprocessor=spacy_tokenizer, 
                
                
                
                
                ngram_range=(1, 3) 
            )
            
            logging.info("TF-IDF Vectorizer object created with n-grams (1, 3)")
            return tfidf_vectorizer

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_transformation(self, processed_data_path):
        """
        Applies the transformation to the data.
        """
        try:
            logging.info("Data transformation process started")
            
            df = pd.read_csv(processed_data_path)
            
            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Fitting vectorizer on all text data...")
            all_text_data = df['text'].astype(str)
            preprocessor_obj.fit(all_text_data)
            logging.info("Vectorizer fitting complete.")

            logging.info(f"Saving preprocessor object to {self.transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Data transformation process completed")
            
            return self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error during data transformation")
            raise customException(e, sys)

