import os
import sys
import glob
import pandas as pd
from dataclasses import dataclass
from src.exception import customException
from src.logger import logging

import io
import pdfplumber
from docx import Document

def extract_text_from_pdf(file_stream: io.BytesIO):
    text = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_stream: io.BytesIO):
    doc = Document(file_stream)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def extract_text(file_bytes: bytes, filename: str):
    f = io.BytesIO(file_bytes)
    if filename.lower().endswith(".pdf"):
        logging.info(f"Extracting text from PDF: {filename}")
        return extract_text_from_pdf(f)
    elif filename.lower().endswith(".docx"):
        logging.info(f"Extracting text from DOCX: {filename}")
        return extract_text_from_docx(f)
    elif filename.lower().endswith(".txt"):
         logging.info(f"Extracting text from TXT: {filename}")
         # fallback: decode as utf-8 plain text
         try:
             return file_bytes.decode("utf-8")
         except:
             logging.error(f"Failed to decode TXT file: {filename}")
             return ""
    else:
        logging.warning(f"Unsupported file type: {filename}. Skipping.")
        return ""



@dataclass
class DataIngestionConfig:
    raw_data_dir: str = os.path.join('data', 'raw')
    processed_data_path: str = os.path.join('data', 'processed', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("DataIngestion component initialized")

    def initiate_data_ingestion(self):
        logging.info("Data ingestion process started")
        try:
            # 1. Find all files in the raw data directory
            all_files_pattern = os.path.join(self.ingestion_config.raw_data_dir, "*.*")
            all_files = glob.glob(all_files_pattern)
            
            if not all_files:
                raise customException(f"No files found in {self.ingestion_config.raw_data_dir}", sys)

            job_data = []
            resume_data = []

            # 2. Loop, read, parse, and categorize files
            for file_path in all_files:
                filename = os.path.basename(file_path)
                try:
                    # Read file as bytes
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    # Extract text using the provided parsers
                    text = extract_text(file_bytes, filename)
                    
                    if text: # Only add if text was successfully extracted
                        file_id = filename.rsplit('.', 1)[0] # 'resume1.txt' -> 'resume1'

                        # Categorize based on filename
                        if filename.lower().startswith('job'):
                            job_data.append({'id': file_id, 'text': text})
                        elif filename.lower().startswith('resume'):
                            resume_data.append({'id': file_id, 'text': text})
                        else:
                            logging.warning(f"Skipping file: {filename} (does not start with 'job' or 'resume')")

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

            # 3. Create DataFrames
            jobs_df = pd.DataFrame(job_data)
            resumes_df = pd.DataFrame(resume_data)

            if jobs_df.empty:
                raise customException("No job descriptions were successfully parsed from 'data/raw'.", sys)
            if resumes_df.empty:
                 raise customException("No resumes were successfully parsed from 'data/raw'.", sys)
            
            jobs_df['type'] = 'job_description'
            resumes_df['type'] = 'resume'

            # 4. Combine into one DataFrame
            all_data_df = pd.concat([jobs_df, resumes_df], ignore_index=True)
            logging.info(f"Loaded and parsed {len(jobs_df)} jobs and {len(resumes_df)} resumes.")

            # 5. Save processed data
            os.makedirs(os.path.dirname(self.ingestion_config.processed_data_path), exist_ok=True)
            all_data_df.to_csv(self.ingestion_config.processed_data_path, index=False, header=True)
            logging.info(f"Processed data saved to {self.ingestion_config.processed_data_path}")

            return self.ingestion_config.processed_data_path

        except Exception as e:
            logging.error("Error during data ingestion")
            raise customException(e, sys)

