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
        data = save_chunks_and_embeddings(docs, embeddings)
        st.write(data)
        return vectorstore
    
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {e}")
        raise
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise
def save_chunks_and_embeddings(docs, embeddings):
    data = []
    for i, doc in enumerate(docs):
        doc_embedding = embeddings.embed_query(doc.page_content)
        data.append({
            "chunk": doc.page_content,
            "embedding": doc_embedding
        })
        break
    return data


def get_retriever(vectorstore):
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k":5})
    else:
        return None
#st.title("LangChain RAG Chatbot")
# docs = load_documents("./policy_workplace_concerns.pdf")
docs = load_documents("./policy_workplace_concerns.pdf")
# st.write(docs)
vectorstore = create_vector_store(docs)
# st.write(vectorstore)
retriever = get_retriever(vectorstore)
# st.write(retriever)
def generate_answer(query, retriever):
    try:
        # Create a prompt template
        prompt_template = """
            You are policy chatbot design to answer the questions related to policy.
            
            Consider yourself as an AI chatbot. The user has a query  instead of providing the exact information, your task is to summarize the relevant sections.Sandhya has trained you.

            question: {question}
            Instructions:

            1.If is providing any word instead of a query then consider that word as query and generate response realted to that query.
            2. Analyze and understand the question and generate response, If question is not appropriate the create a appropriate question by yourself and generate output related to that question.
            3. Provide a clear and concise answer.
            4. Ensure your response is informative and relevant to the user's query.
            5. If any question ask regarding  the harrasment or Sexual harassment generate appropriate answer for
            6.If the user greets you with any common greeting like "hello", "hi", "hey", "good morning", "good evening", or similar phrases, respond with a friendly and natural greeting in return. For example, if the user says "hello", respond with something like "Hello! How can I assist you today?" or "Hi there! How's it going?" Ensure the response is warm and welcoming, matching the tone of the user's greeting.
            
            If user question is about greeting respond with greeting :
            # Example 1:
            question: Hi 
            assistant: Hello. How can i help you.
 
            Provide a concise, summarized answer based on the matching sections from the document, highlighting key points without repeating text verbatim. If specific keywords match, ensure the answer reflects that but in a summarized form.
        """
 
 
        # Use PromptTemplate to structure the input
        prompt = PromptTemplate(
            input_variables=[ "question"],
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
        background-color: #f2a354e0 !important;
        
    }

    """, unsafe_allow_html=True)

with st.sidebar.markdown('', unsafe_allow_html=True):
    st.image('logo.png',width=300)

# st.markdown("<h2>Hi How can I help you?</h2>", unsafe_allow_html=True)
query = st.chat_input("Message Insighbot")

# if st.button("Submit") and query:
# message = st.chat_message("user", avatar="🙍🏻‍♀️")
##message.write(query)
 
# answer, source_docs = generate_answer(query, retriever)
#message = st.chat_message("assistant",avatar="🤖")
#message.write(answer)

 
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
