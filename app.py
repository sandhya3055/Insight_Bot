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
import os
from conn import *
load_dotenv()
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

st.set_page_config(
    page_title="Document Summarizer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
#load the document
# def load_documents(file_path):
#     try:
#         # Use PyPDFLoader to load and split PDF documents
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#         #  Split the extracted text into documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         docs = text_splitter.split_documents(documents)  # Updated method to split raw text
#         return docs
#     except Exception as e:
#         # st.error(f"Error loading PDF file: ")
#         return []
# def create_vector_store(docs):
#     try:
#         # Use Google Generative AI embeddings
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         # Create the FAISS vector store from the documents and embeddings
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         save_chunks_and_embeddings(docs, embeddings)
#         # st.write(data)
#         return vectorstore
    
#     except ModuleNotFoundError as e:
#         # st.error(f"Module not found: ")
#         raise
    
#     except Exception as e:
#         # st.error(f"Error creating vector store: ")
#         raise
# def save_chunks_and_embeddings(docs, embeddings):
#     try:
#         data = []
#         for i, doc in enumerate(docs):
#             doc_embedding = embeddings.embed_query(doc.page_content)
#             data.append({
#                 "chunk": doc.page_content,
#                 "embedding": doc_embedding
#             })
#             res = insert_query(doc.page_content,doc_embedding)
            # if res:
                # print('data inserted sucessfully')
#         # return data
#     except Exception as e:
#         print("error : ",e)


# def get_retriever(vectorstore):
#     if vectorstore:
#         return vectorstore.as_retriever(search_kwargs={"k":5})
#     else:
#         return None
# docs = load_documents("./policy_workplace_concerns.pdf")
# # st.write(docs)
# vectorstore = create_vector_store(docs)
# # st.write(vectorstore)
# retriever = get_retriever(vectorstore)
# # st.write(retriever)

def create_faiss_retriever_from_db():
    try:
        # Fetch documents and embeddings from the database
        docs, embeddings = fetch_chunks_and_embedding()
 
        # Check if docs and embeddings are present and their lengths match
        if not docs or not embeddings or len(docs) != len(embeddings):
            raise ValueError("The number of documents does not match the number of embeddings.")
 
        # Convert embeddings to a NumPy array and ensure it's in the correct data type (float32)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Initialize FAISS index with L2 distance and embedding dimension
        embedding_dim = embeddings_np.shape[1]  # Assuming 2D array
        # print(embedding_dim)
        faiss_index = faiss.IndexFlatL2(embedding_dim)  # Use L2 distance for similarity search
        
        # Add embeddings to FAISS index
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
            google_api_key=os.getenv("GOOGLE_API_KEY")  # Ensure GOOGLE_API_KEY is set
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
        print(f"FAISS retriever creation failed: {e}")
        return None


def generate_answer(query, retriever):
    try:
        # Assuming retriever is the one created from the database embeddings
        if retriever is None:
            raise ValueError("Retriever is not initialized")
        
        relevant_docs = retriever.get_relevant_documents(query)
        # print(relevant_docs)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create a prompt template
        prompt_template = """
            Consider yourself as an AI chatbot. A document is provided to you Your task is to generate a clear and concise answer from the  provided document .If you will not able to find the answer in provided document then generate response related to the word which is present in question.provide response in  more detailed manner.

            context : {context}
            question: {question}
            Instructions:

            1.If is providing any word instead of a query then consider that word as query and generate response realted to that query.
            2. Analyze and understand the question and generate response, If question is not appropriate the create a appropriate question by yourself and generate output related to that question.  
            3. Ensure your response is informative and relevant to the user's query.
            4. If any question ask regarding harrasment generate appropriate answer for it.
            5. If user greets you generate response in greeting manner.The user has greeted you (e.g., "Hello," "Hi," "Hey"). Generate a friendly and engaging response that acknowledges the greeting and offers assistance.
            6. You are a robust and flexible assistant designed to understand and answer questions, even if the user makes spelling mistakes or uses incorrect wording. When given a question, interpret the intent of the question and provide a relevant, accurate answer based on the provided document. If you are unsure of the user's intent due to ambiguous wording, make an educated guess. Always prioritize providing a helpful response.
 
        """
 
 
        # Use PromptTemplate to structure the input
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        # Initialize the model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # Create a QA chain with the retriever and the model
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        # Retrieve relevant documents for the context
        # context_docs = retriever.get_relevant_documents(query)
        # context = "\n".join([doc.page_content for doc in context_docs])
        
        # Format the prompt using the retrieved context and user's question
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Run the chain with the formatted prompt
        result = qa_chain({"query": formatted_prompt})
        
        # Separate the answer and source documents
        answer = result['result']
        source_docs = result['source_documents']
        
        return answer, source_docs
    
    except Exception as e:
        st.error(f"An error occurred while generating the answer: ")
        return None, None


st.markdown("""
<style>
    .stSidebar {
        background-color: #f2a354e0 !important;
        
    }

    """, unsafe_allow_html=True)

with st.sidebar.markdown('', unsafe_allow_html=True):
    st.image('logo.png',width=300)

query = st.chat_input("Message Insighbot")
retriever = create_faiss_retriever_from_db()


 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
 

 
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})
    answer, source_docs = generate_answer(query, retriever)

    with st.chat_message('assistant'):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
 
 




# def initialize_session_state():
