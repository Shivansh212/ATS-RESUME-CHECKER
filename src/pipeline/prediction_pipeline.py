import sys
import os
from src.exception import customException
from src.logger import logging
from src.utils import load_object
from sklearn.metrics.pairwise import cosine_similarity


from src.components.Data_ingestion import extract_text

class PredictionPipeline:
    def __init__(self):
        
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        logging.info("PredictionPipeline initialized")

    def predict_score(self, resume_file_bytes, resume_filename, jd_file_bytes, jd_filename):
        """
        Predicts the similarity score between a single new resume and a single new job description.
        
        Args:
            resume_file_bytes (bytes): The byte content of the uploaded resume file.
            resume_filename (str): The original filename of the resume (e.g., "my_resume.pdf").
            jd_file_bytes (bytes): The byte content of the uploaded JD file.
            jd_filename (str): The original filename of the JD (e.g., "job_desc.docx").
        
        Returns:
            float: A similarity score formatted as a percentage (e.g., 85.25).
        """
        try:
            logging.info("Prediction process started")

            # 1. Load the saved TF-IDF vectorizer (the "brain")
            logging.info(f"Loading preprocessor from: {self.preprocessor_path}")
            vectorizer = load_object(file_path=self.preprocessor_path)
            if not vectorizer:
                raise customException("Could not load preprocessor model. Has the training pipeline been run?", sys)
            
            logging.info("Preprocessor model loaded successfully.")

            # 2. Parse text from the uploaded files (in-memory)
            logging.info(f"Parsing resume text from: {resume_filename}")
            resume_text = extract_text(resume_file_bytes, resume_filename)
            if not resume_text:
                raise customException(f"Could not extract text from resume: {resume_filename}", sys)

            logging.info(f"Parsing job description text from: {jd_filename}")
            jd_text = extract_text(jd_file_bytes, jd_filename)
            if not jd_text:
                raise customException(f"Could not extract text from job description: {jd_filename}", sys)
            
            logging.info("Text extraction complete.")

            # 3. Transform the two new text documents
            
            documents = [resume_text, jd_text]
            
            logging.info("Transforming new text into TF-IDF vectors...")
            vectors = vectorizer.transform(documents)
            
            # 4. Separate the vectors
            resume_vector = vectors[0]
            jd_vector = vectors[1]

            # 5. Calculate Cosine Similarity between just these two vectors
            logging.info("Calculating cosine similarity...")
            score = cosine_similarity(resume_vector, jd_vector)
            
            
            similarity_score = score[0][0]
            
            

            # 6. Format as percentage
            final_score = round(similarity_score * 100, 2)
            
            logging.info(f"Prediction complete. Score: {final_score}%")
            
            return final_score

        except Exception as e:
            logging.error("Error during prediction")
            raise customException(e, sys)

if __name__ == "__main__":
    
    pass