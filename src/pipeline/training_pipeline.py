import sys
from src.exception import customException
from src.logger import logging
from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        logging.info("TrainingPipeline initialized")
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        """
        Executes the full training pipeline step-by-step.
        """
        try:
            logging.info("Training pipeline started...")

            # Step 1: Data Ingestion
           
            logging.info("Starting Data Ingestion...")
            processed_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Processed data at: {processed_data_path}")

            # Step 2: Data Transformation
            
            logging.info("Starting Data Transformation...")
            preprocessor_obj_path = self.data_transformation.initiate_data_transformation(processed_data_path)
            logging.info(f"Data Transformation completed. Preprocessor at: {preprocessor_obj_path}")

            # Step 3: Model Training (Scoring)
            
            logging.info("Starting Model Training (Scoring)...")
            self.model_trainer.initiate_model_training(processed_data_path, preprocessor_obj_path)
            logging.info("Model Training (Scoring) completed.")

            logging.info("Training pipeline finished successfully.")

        except Exception as e:
            logging.error("Training pipeline failed.")
            raise customException(e, sys)


if __name__ == "__main__":
    
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()