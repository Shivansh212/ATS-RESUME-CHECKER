import sys
from flask import Flask, request, render_template, redirect, url_for, flash
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import customException
from src.logger import logging
import os

application = Flask(__name__)
app = application

# This is needed for flashing messages (e.g., error messages)
app.secret_key = "ats_project_secret_key" 

# Route for the main welcome page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the prediction form (home.html)
# This route handles both GET (showing the form) and POST (submitting the form)
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # Just show the form page
        return render_template('home.html')
    else:
        # This is a POST request, so we process the uploaded files
        try:
            # 1. Check if files are present in the request
            if 'resume' not in request.files or 'jd' not in request.files:
                flash('Both resume and job description files are required.', 'error')
                return redirect(request.url)

            resume_file = request.files['resume']
            jd_file = request.files['jd']

            # 2. Check if files are selected
            if resume_file.filename == '' or jd_file.filename == '':
                flash('Both files must be selected.', 'error')
                return redirect(request.url)

            # 3. Read file bytes and get filenames
            resume_bytes = resume_file.read()
            resume_filename = resume_file.filename
            
            jd_bytes = jd_file.read()
            jd_filename = jd_file.filename
            
            logging.info(f"Received files: {resume_filename}, {jd_filename}")

            # 4. Call your Prediction Pipeline
            logging.info("Initializing Prediction Pipeline...")
            pipeline = PredictionPipeline()
            score = pipeline.predict_score(
                resume_file_bytes=resume_bytes,
                resume_filename=resume_filename,
                jd_file_bytes=jd_bytes,
                jd_filename=jd_filename
            )
            logging.info(f"Prediction successful. Score: {score}")

            # 5. Format the result text
            result_text = f"Your ATS Match Score is: {score:.2f}%"

            # 6. Render the result page
            return render_template('result.html', prediction_text=result_text)

        except Exception as e:
            logging.error("Error occurred in /home POST route")
            flash(f"An error occurred: {str(e)}", 'error')
            # Go back to the form page if an error happens
            return redirect(url_for('home'))


if __name__ == "__main__":
    # To run:
    # 1. Make sure your 'artifacts/preprocessor.pkl' exists (by running training_pipeline.py once)
    # 2. Run this file: python app.py
    # 3. Open http://127.0.0.1:5000 in your browser
    app.run(host="0.0.0.0", port=5006, debug=True)