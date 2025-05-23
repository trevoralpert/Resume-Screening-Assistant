# Resume Screening Assistance App

A Streamlit application designed to assist HR professionals in the resume screening process. This tool leverages Pinecone and Large Language Models (LLMs) to enable users to upload resumes, analyze their relevance to a given job description, and return the most suitable candidates based on similarity scores.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Example Scenarios](#example-scenarios)
- [Contributing](#contributing)
- [License](#license)

## Overview

This app streamlines the resume screening process using a Streamlit interface, Pinecone for vector similarity search, and LLMs for text analysis. Users can:
- Upload multiple resumes in PDF format.
- Provide a job description.
- Retrieve resumes most relevant to the job description, ranked by a similarity score.

## Features

- Upload multiple PDF resumes.
- Input a job description to define desired candidate qualifications.
- Uses sentence embeddings (specifically, `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Transformers) for similarity calculations.
- Stores and queries embeddings using Pinecone's vector database.
- Summarizes resumes for quick insights using an LLM (OpenAI).
- Displays match scores for each resume, indicating relevance to the job description.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For creating the interactive web application.
- **LangChain**: For managing LLM interactions, document processing, and embedding workflows.
- **Pinecone**: For efficient vector similarity search and storage of resume embeddings.
- **OpenAI**: Used via LangChain for its LLM capabilities, specifically for summarizing resumes.
- **HuggingFace Transformers**: For generating sentence embeddings from resume text and job descriptions (`sentence-transformers/all-MiniLM-L6-v2` model).
- **PyPDF**: For extracting text content from uploaded PDF resumes.
- **python-dotenv**: For managing environment variables (e.g., API keys) securely.

## Setup and Installation

### 1. Prerequisites
- Ensure you have Python 3.7+ installed on your system.

### 2. Clone the repository
   ```bash
   git clone https://github.com/trevoralpert/resume-screening-assistance.git
   cd resume-screening-assistance
   ```

### 3. Create and activate a virtual environment
   - **For Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **For Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

### 4. Install dependencies
   Once the virtual environment is activated, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### 5. Set up environment variables
   This application requires API keys for Pinecone and OpenAI services.
   - A file named `.env.example` is included in the repository. This file lists all the necessary environment variables.
   - Create a copy of this file and name it `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open the `.env` file and add your specific API keys and configuration:
     ```
     PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
     PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., "gcp-starter" or "us-west1-gcp" etc.
     PINECONE_INDEX_NAME="your-pinecone-index-name" # Choose a name for your Pinecone index
     OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
     ```
   **Note:** Ensure that the Pinecone index name you choose here matches the one used in your Pinecone account setup if the index already exists, or it will be created with this name. The `PINECONE_ENVIRONMENT` should also match your Pinecone project's environment.

## Usage

Follow these steps to run and use the Resume Screening Assistance application:

### 1. Running the Application
   - Ensure that you have completed all steps in the [Setup and Installation](#setup-and-installation) section, including activating your virtual environment and setting up environment variables.
   - Navigate to the root directory of the project (where `app4.py` is located).
   - Run the following command in your terminal:
     ```bash
     streamlit run app4.py
     ```
   - This will open the application in your default web browser.

### 2. Steps to Use the Application
   Once the application is running, follow these steps:

   1.  **Paste Job Description**:
       *   In the text area labeled "Please paste the 'JOB DESCRIPTION' here...", paste the complete job description for the role you are hiring for.

   2.  **Specify Number of Resumes**:
       *   In the input field labeled "No. of 'RESUMES' to return", enter the number of top matching resumes you wish to see (e.g., 3, 5, 10).

   3.  **Upload Resumes**:
       *   Click on the "Browse files" button under "Upload resumes here (PDF only):" or drag and drop PDF files into the uploader.
       *   You can upload multiple PDF resume files at once.

   4.  **Start Analysis**:
       *   After providing the job description, the number of resumes to return, and uploading the resume files, click the "Help me with the analysis" button.
       *   A spinner will appear indicating that the analysis is in progress.

   5.  **Review Results**:
       *   Once the analysis is complete, the application will display the following:
           *   **Total Resumes Uploaded**: A count of the PDF files processed.
           *   **Ranked Resumes**: A list of the top resumes that match the job description, based on the number you specified. Each result will include:
               *   **Resume Number**: An identifier for the resume in the list (e.g., 👉 Resume 1).
               *   **File Name**: The original name of the uploaded PDF file for the recommended resume.
               *   **Match Score**: A similarity score (e.g., 0.85) indicating how well the resume matches the job description. Higher scores indicate a better match.
               *   **Summary (Expandable)**: An expandable section labeled "View Summary". Clicking on this will show an AI-generated summary of the resume content, allowing for a quick review.
           *   A success message: "Analysis complete! Hope I saved you some time."

   **Note on Errors:** If an error occurs (e.g., missing API keys, issues with PDF parsing), an error message will be displayed on the screen. Please check your setup, especially the `.env` file and the PDF files, if this happens.

## Example Scenarios

Here are a couple of ways you might use the Resume Screening Assistance:

### Scenario 1: Hiring for a Technical Role

*   **Job Description**: You paste a detailed job description for a "Senior Python Developer" requiring experience with Django, REST APIs, and PostgreSQL.
*   **Resumes**: You upload a folder containing 75 resumes from various applicants.
*   **Number to Return**: You specify you want to see the top 5 matches.
*   **Outcome**: The application processes the resumes and presents the 5 candidates whose resumes most closely match the Python developer job description, along with their match scores and summaries, allowing you to quickly focus on the most promising applicants.

### Scenario 2: Quick Screening for a Junior Role

*   **Job Description**: You input a simpler job description for a "Marketing Assistant" focusing on social media skills and content creation.
*   **Resumes**: You have a batch of 30 resumes for an internship program.
*   **Number to Return**: You set it to 3.
*   **Outcome**: The tool helps you identify the top 3 candidates who best fit the junior role, saving time in initial screening. You can then review their summaries before deciding on interviews.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to:

1.  **Open an Issue**: If you want to discuss a new feature, a change, or report a bug, please open an issue first. This allows for discussion before any work is done.
2.  **Fork the Repository**: Create your own fork of the project.
3.  **Create a Feature Branch**: Make your changes in a dedicated branch (e.g., `git checkout -b feature/AmazingFeature` or `git checkout -b bugfix/IssueName`).
4.  **Commit Your Changes**: Write clear and concise commit messages (e.g., `git commit -m 'Add some AmazingFeature'`).
5.  **Push to the Branch**: Push your changes to your forked repository (e.g., `git push origin feature/AmazingFeature`).
6.  **Open a Pull Request**: Submit a pull request from your feature branch to the main repository's `main` or `master` branch.

Please make sure your code adheres to any existing coding standards and include tests if applicable.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
