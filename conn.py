from sqlalchemy import create_engine,text
import psycopg2
from sqlalchemy import Column, Integer, String, DateTime
import os
 
from dotenv import load_dotenv
load_dotenv()
 
# Access the variables
REAL_DB_USER = os.getenv("REAL_DB_USER")
REAL_DB_PASSWORD = os.getenv("REAL_DB_PASSWORD")
REAL_DB_HOST = os.getenv("REAL_DB_HOST")
REAL_DB_NAME = os.getenv("REAL_DB_NAME")
REAL_DB_PORT = os.getenv("REAL_DB_PORT")
 
engine =create_engine(
    # f"postgresql://{REAL_DB_USER}@{REAL_DB_HOST}:{REAL_DB_PORT}/{REAL_DB_NAME}"
    f"postgresql+psycopg2://{REAL_DB_USER}@{REAL_DB_HOST}:{REAL_DB_PORT}/{REAL_DB_NAME}"
)
def execute_query(query, params=None):
    try:
        with engine.connect() as connection:
            transaction=connection.begin()
            result = connection.execute(text(query), params)
            transaction.commit()

            return result
    except Exception as e:
        print(f"An error occurred: {e}")


def create_table():
    try:
        query=f"""
                CREATE TABLE IF NOT EXISTS insight_bot(
                description TEXT NOT NULL,
                description_embedding FLOAT8[] NOT NULL
            );
        """
        result=execute_query(query)
    except Exception as e:
        print("Create table query error")
 
def insert_query(doc, embedding):
        try:
            query = """
                INSERT INTO insight_bot (description, description_embedding)
                VALUES (:doc, :embedding)
            """
            params = {'doc': doc, 'embedding': embedding}  # Use dictionary to pass params
            result = execute_query(query, params)
            # if result :
            #     print('data inserted successfully')
            # else :
            #     print('insertion failed')  
        except Exception as e:
            print(f"Insert query error: {e}")
            return None

def fetch_chunks_and_embedding():
    try:
        query="""
            SELECT description,description_embedding
            FROM insight_bot
        """
        result= execute_query(query)
        rows=result.mappings().all()
        # Separate descriptions (chunk) and embeddings into two lists
        descriptions =[row['description'] for row in rows]
        embeddings =[row['description_embedding'] for row in rows]

        return descriptions,embeddings
    except Exception as e:
        print(f'An error occur while fetching the data :{e}')
        return None,None
# create_table()
# insert_query('sbsbd', '{729.28,849.90}')
