import os
import tempfile
import streamlit as st
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import faiss
from langchain.docstore import InMemoryDocstore
from dotenv import load_dotenv
from conn import *

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# Streamlit configuration
st.set_page_config(
    page_title="Document Summarizer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load documents
def load_documents(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error loading PDF file: {str(e)}")
        return []

# Function to create vector store
def create_vector_store(docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        save_chunks_and_embeddings(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to save chunks and embeddings to the database
def save_chunks_and_embeddings(docs, embeddings):
    try:
        for i, doc in enumerate(docs):
            doc_embedding = embeddings.embed_query(doc.page_content)
            res = insert_query(doc.page_content, doc_embedding)
            if res:
                print('Data inserted successfully')
    except Exception as e:
        print(f"Error: {e}")

# Function to insert document into the database
def insert_document_into_db(doc):
    # Use a temporary file to save the uploaded PDF and pass its path to the loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(doc.read())
        temp_file_path = temp_file.name
    
    # Load and process the document using the temporary file path
    docs = load_documents(temp_file_path)
    if docs:
        create_vector_store(docs)
        st.success("Document successfully processed and inserted into the database.")
    else:
        st.error("Failed to process the document.")

    # Remove the temporary file after processing
    os.remove(temp_file_path)

# Function to create FAISS retriever from the database
def create_faiss_retriever_from_db():
    try:
        docs, embeddings = fetch_chunks_and_embedding()

        # Check if the documents and embeddings are correctly fetched
        if not docs:
            raise ValueError("No documents found.")
        if not embeddings:
            raise ValueError("No embeddings found.")
        if len(docs) != len(embeddings):
            raise ValueError(f"Mismatch between the number of documents ({len(docs)}) and embeddings ({len(embeddings)}).")

        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Check if embeddings are 2D
        if len(embeddings_np.shape) != 2:
            raise ValueError("Embeddings array is not 2-dimensional.")
        
        embedding_dim = embeddings_np.shape[1]
        
        # Initialize FAISS index with L2 distance and embedding dimension
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(embeddings_np)

        # Create documents in LangChain format
        documents = [Document(page_content=doc) for doc in docs]
        
        # Create a mapping from FAISS index to document store IDs
        index_to_docstore_id = {i: str(i) for i in range(len(embeddings))}
        
        # Create document store using LangChain's InMemoryDocstore
        docstore = InMemoryDocstore(dict(zip(index_to_docstore_id.values(), documents)))

        # Use the Google Generative AI Embeddings model for embedding (ensure it's properly configured)
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # Create FAISS vector store
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        # Return the retriever
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"FAISS retriever creation failed: {str(e)}")
        return None

# Function to generate an answer using the model
def generate_answer(query, retriever):
    try:
        if retriever is None:
            raise ValueError("Retriever is not initialized")

        relevant_docs = retriever.get_relevant_documents(query)
        
        # Check if there are any relevant documents
        if not relevant_docs:
            raise ValueError("No relevant documents found.")
        
        context = "\n".join([doc.page_content for doc in relevant_docs])

        prompt_template = """
            Consider yourself as an AI chatbot. A document is provided to you. Your task is to generate a clear and concise answer from the provided document. If you are not able to find the answer in the provided document, generate a response related to the word present in the question. Provide a response in a more detailed manner.

            Context: {context}
            Question: {question}
            Instructions:
            1. If a single word is provided instead of a query, consider that word as a query and generate a response related to that query.
            2. Analyze and understand the question and generate a response. If the question is not appropriate, create an appropriate question yourself and generate output related to that question.
            3. Ensure your response is informative and relevant to the user's query.
            4. If any question asks about harassment, generate an appropriate answer for it.
            5. If the user greets you, generate a response in a greeting manner. Acknowledge the greeting and offer assistance.
            6. You are a robust and flexible assistant designed to understand and answer questions, even if the user makes spelling mistakes or uses incorrect wording. Interpret the user's intent and provide a relevant, accurate answer based on the provided document. If unsure of the user's intent due to ambiguous wording, make an educated guess. Always prioritize providing a helpful response.
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )

        formatted_prompt = prompt.format(context=context, question=query)
        result = qa_chain({"query": formatted_prompt})

        answer = result['result']
        source_docs = result['source_documents']

        return answer, source_docs
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {str(e)}")
        return None, None

# Sidebar UI setup
st.markdown("""
<style>
    .stSidebar {
        background-color: #f2a354e0 !important;
    }
    """, unsafe_allow_html=True)

st.sidebar.image('./Image/logo.png', width=300)

# Upload and process document
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if st.button("Upload"):
    if uploaded_file:
        insert_document_into_db(uploaded_file)
    else:
        st.error('Please select a file to upload')

# Initialize chat messages session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input and response handling
query = st.chat_input("Message InsightBot")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Create retriever only once if not already created
    retriever = create_faiss_retriever_from_db()
    if retriever:
        answer, source_docs = generate_answer(query, retriever)
        if answer:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
