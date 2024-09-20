# INSIGHT BOT:
This repository contains a chatbot application built using Streamlit, LangChain, and Google Generative AI. The chatbot is designed to process PDF documents, summarize content, and answer questions based on the documents provided.

# 1. Clone the repository:

git clone https://github.com/sandhya3055/Insight_Bot.git

# 2. Set up environment variables:

Create a .env file in the root directory and add the GOOGLE API KEY
GOOGLE_API_KEY='your_GOOGLE_api_key'

# 3. Prerequisites

Python 3
Google API Key for Generative AI
 

# 4. Install the required packages:

pip install -r requirements.txt

# 5 PostgreSQL Setup
Install PostgreSQL and pgAdmin if not already present in your system. In pgAdmin, after registering your server, create the following table using the following command in the query tool:


        CREATE TABLE IF NOT EXISTS insight_bot(

        description TEXT NOT NULL,

        description_embedding FLOAT8[] NOT NULL

        );
            
    

# 6. Run the Streamlit app:

streamlit run app.py

# 7. Usage

Upload a PDF: Use the file uploader in the sidebar to select and upload a PDF document.
Process Document: Click the "Upload" button to process the document. The document will be split, and its content will be stored in a database.
Ask Questions: Type your questions in the chat input. The chatbot will respond with answers based on the uploaded document

# 8. Project Structure

app.py: The main Streamlit application file.
conn.py: Contains database connection and query functions.
Image/: Contains images used in the app, such as the logo.
README.md: This file, describing the project.
