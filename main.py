import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os
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
def load_documents(file_path):
    try:
        # Use PyPDFLoader to load and split PDF documents
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        #  Split the extracted text into documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)  # Updated method to split raw text
        return docs
    except Exception as e:
        st.error(f"Error loading PDF file: {str(e)}")
        return []
def create_vector_store(docs):
    try:
        # Use Google Generative AI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create the FAISS vector store from the documents and embeddings
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        return vectorstore
    
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {e}")
        raise
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

def get_retriever(vectorstore):
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k":5})
    else:
        return None
st.title("LangChain RAG Chatbot")
# docs = load_documents("./policy_workplace_concerns.pdf")
docs = load_documents("./Handbook.pdf")
# st.write(docs)
vectorstore = create_vector_store(docs)
# st.write(vectorstore)
retriever = get_retriever(vectorstore)
# st.write(retriever)
def generate_answer(query, retriever):
    try:
        # Create a prompt template
        prompt_template = """
            Consider you as a HR handbook assistant. The user has a question related to a document, but instead of providing the exact information from the document, your task is to summarize the relevant sections.
 
            Context: You have to generate response from given document. In case if some question does not have seems to be a question then generate response related to words, which is present into question.
 
            Question: {question}
 
            Instructions:
                1. Analyze and understand the question and generate response, If question is not appropriate the create a appropriate question by yourself and generate output related to that question.
                2. If question is not from the document then also generate response and say that you are only trained for that document.
                3. Summarize your response.
        """
 
 
        # Use PromptTemplate to structure the input
        prompt = PromptTemplate(
            input_variables=[ "question"],
            template=prompt_template
        )
        
        # Initialize the model
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
                        temperature=0.5,
            convert_system_message_to_human=True
        )
        
        # Create a QA chain with the retriever and the model
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        # Retrieve relevant documents for the context
        context_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Format the prompt using the retrieved context and user's question
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Run the chain with the formatted prompt
        result = qa_chain({"query": formatted_prompt})
        
        # Separate the answer and source documents
        answer = result['result']
        source_docs = result['source_documents']
        
        return answer, source_docs
    
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {str(e)}")
        return None, None


st.markdown("""
<style>
    .stSidebar {
        background-color: #87ceeb !important;
        
    }
    .sidebarh1{
        border: 2px solid grey;
        border-radius : 12px;
        padding : 12px;
    }
    </style>

    """, unsafe_allow_html=True)
st.title(" ")
st.sidebar.markdown('<h1 class="sidebarh1">HR Handbook Chatbot </h1>', unsafe_allow_html=True)

# st.markdown("<h2>Hi How can I help you?</h2>", unsafe_allow_html=True)
query = st.chat_input()

# if st.button("Submit") and query:
message = st.chat_message("user")
message.write(query)


answer, source_docs = generate_answer(query, retriever)
message = st.chat_message("assistant")
message.write(answer)
