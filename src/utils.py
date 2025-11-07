import os
import sys
import pickle
from src.exception import customException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        raise customException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        raise customException(e, sys)