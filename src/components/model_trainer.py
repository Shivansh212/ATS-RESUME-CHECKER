import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import customException
from src.logger import logging
from src.utils import load_object

@dataclass
class ModelTrainerConfig:
    
    scores_file_path: str = os.path.join('artifacts', 'ats_scores.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("ModelTrainer component initialized")

    def initiate_model_training(self, processed_data_path, preprocessor_obj_path):
        """
        "Training" here means loading the vectorizer and calculating
        the similarity scores between all resumes and jobs.
        """
        try:
            logging.info("Model training (scoring) process started")

            # 1. Load the processed data and the fitted vectorizer
            df = pd.read_csv(processed_data_path)
            vectorizer = load_object(file_path=preprocessor_obj_path)
            logging.info("Loaded processed data and preprocessor object")

            # 2. Separate jobs and resumes
            jobs_df = df[df['type'] == 'job_description'].reset_index(drop=True)
            resumes_df = df[df['type'] == 'resume'].reset_index(drop=True)

            if jobs_df.empty or resumes_df.empty:
                raise customException("No jobs or resumes to compare.", sys)

            # 3. Transform the text data using the *loaded* vectorizer
            
            logging.info("Transforming job and resume text into TF-IDF vectors...")
            job_vectors = vectorizer.transform(jobs_df['text'])
            resume_vectors = vectorizer.transform(resumes_df['text'])
            
            

            # 4. Calculate Cosine Similarity
            
            logging.info("Calculating cosine similarity matrix...")
            similarity_matrix = cosine_similarity(resume_vectors, job_vectors)
            
            

            # 5. Format the results into a readable DataFrame
            
            resume_ids = resumes_df['id']
            job_ids = jobs_df['id']

            scores_df = pd.DataFrame(similarity_matrix, index=resume_ids, columns=job_ids)

            
            scores_df = scores_df.apply(lambda x: round(x * 100, 2))

            logging.info(f"Calculated Scores:\n{scores_df}")

            # 6. Save the scores CSV
            scores_df.to_csv(self.model_trainer_config.scores_file_path)
            logging.info(f"Scores saved to {self.model_trainer_config.scores_file_path}")

            logging.info("Model training (scoring) process completed")

        except Exception as e:
            logging.error("Error during model training/scoring")
            raise customException(e, sys)

if __name__ == "__main__":
    
    from src.components.Data_ingestion import DataIngestion
    from src.components.Data_transformation import DataTransformation

    # Step 1: Ingestion
    ingestion = DataIngestion()
    processed_path = ingestion.initiate_data_ingestion()
    
    # Step 2: Transformation
    transformer = DataTransformation()
    preprocessor_path = transformer.initiate_data_transformation(processed_path)
    
    # Step 3: Training (Scoring)
    trainer = ModelTrainer()
    trainer.initiate_model_training(processed_path, preprocessor_path)
    print(f"Scoring complete. Scores saved to {trainer.model_trainer_config.scores_file_path}")