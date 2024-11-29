import streamlit as st
import base64
from byaldi import RAGMultiModalModel
from PIL import Image
from io import BytesIO
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(layout="wide", page_title="Multimodal RAG Assistant")

# Constants
UPLOAD_DIR = "./doc"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state for RAG model
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None

def initialize_rag_model(model_name):
    """Initialize the RAG model with caching"""
    @st.cache_resource
    def load_model(name):
        return RAGMultiModalModel.from_pretrained(name, verbose=10)
    return load_model(model_name)

def create_rag_index(rag_model, file_path):
    """Create RAG index for the document"""
    @st.cache_data
    def index_document(_file_path):
        rag_model.index(
            input_path=_file_path,
            index_name="document_index",
            store_collection_with_index=True,
            overwrite=True,
        )
    index_document(file_path)

def get_llm_instance(llm_choice):
    """Get LLM instance based on user choice"""
    if llm_choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif llm_choice == "gpt4":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return None

def generate_response(llm, query, base64_image=None):
    """Generate response from LLM"""
    if base64_image:
        query_with_image = f"{query}\n\nImage data (base64): {base64_image}"
        messages = [HumanMessage(content=query_with_image)]
    else:
        messages = [HumanMessage(content=query)]
    
    response = llm.predict_messages(messages)
    return response.content

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    colpali_model = st.selectbox(
        "Select RAG Model",
        options=["vidore/colpali", "vidore/colpali-v1.2"],
        key="rag_model_choice"
    )
    
    llm_choice = st.selectbox(
        "Select LLM Model",
        options=["gemini", "gpt4"],
        key="llm_choice"
    )
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf"])

# Main content
st.title("Multimodal RAG Assistant")

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Document Processing")
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Save uploaded file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved: {uploaded_file.name}")
        
        # Initialize and index
        if st.session_state.rag_model is None:
            with st.spinner("Initializing RAG model..."):
                st.session_state.rag_model = initialize_rag_model(colpali_model)
        
        with st.spinner("Indexing document..."):
            create_rag_index(st.session_state.rag_model, save_path)
    
    with col2:
        st.write("### Query and Results")
        query = st.text_input("Enter your query")
        
        if st.button("Process Query"):
            if query:
                with st.spinner("Processing query..."):
                    # Perform RAG search
                    results = st.session_state.rag_model.search(
                        query, 
                        k=1, 
                        return_base64_results=True
                    )
                    
                    if results:
                        # Display image result
                        image_data = base64.b64decode(results[0].base64)
                        image = Image.open(BytesIO(image_data))
                        st.image(image, caption="Retrieved Image", use_column_width=True)
                        
                        # Generate LLM response
                        llm = get_llm_instance(llm_choice)
                        if llm:
                            response = generate_response(llm, query, results[0].base64)
                            st.markdown("### LLM Response")
                            st.markdown(response)
                        else:
                            st.error("Invalid LLM choice")
                    else:
                        st.warning("No results found for your query")
            else:
                st.warning("Please enter a query")
else:
    st.info("Please upload a document to begin")