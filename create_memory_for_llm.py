import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. DEFINE CONSTANTS ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
# File to track which documents have already been processed
PROCESSED_FILES_LOG = "vectorstore/processed_files.log"

# --- 2. SETUP FUNCTIONS ---

# Function to read the list of already processed files
def get_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            # Use a set for efficient lookups
            return set(line.strip() for line in f)
    return set()

# Function to update the list of processed files
def update_processed_files(new_files):
    with open(PROCESSED_FILES_LOG, "a") as f:
        for file in new_files:
            f.write(f"{file}\n")

# Function to create chunks from documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- 3. MAIN EXECUTION LOGIC ---
def main():
    print("Starting the vector store creation/update process...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- A) LOGIC FOR UPDATING AN EXISTING DATABASE ---
    if os.path.exists(DB_FAISS_PATH):
        print("Existing vector store found. Checking for new documents...")
        
        # Loading the existing database
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        # Getting the sets of processed and current files
        processed_files = get_processed_files()
        current_files = set(os.listdir(DATA_PATH))
        
        # Finding which files are new
        new_files = [f for f in current_files if f.endswith('.pdf') and f not in processed_files]

        if not new_files:
            print("No new documents to add. Exiting.")
            return

        print(f"Found {len(new_files)} new document(s) to process: {', '.join(new_files)}")
        
        # Processing and adding only the new files
        new_docs = []
        for file in new_files:
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            new_docs.extend(loader.load())
        
        new_chunks = create_chunks(new_docs)
        db.add_documents(new_chunks)
        db.save_local(DB_FAISS_PATH)
        update_processed_files(new_files)
        print("Successfully updated the vector store with new documents.")

    # --- B) LOGIC FOR CREATING A NEW DATABASE ---
    else:
        print("No existing vector store found. Building a new one from scratch...")
        
        # Load all documents from the data directory
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            print("No PDF documents found in the 'data' folder. Exiting.")
            return
            
        text_chunks = create_chunks(documents)
        
        # Create the database from scratch
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        # Log all the files that were just processed
        initial_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        update_processed_files(initial_files)
        print("Successfully created and saved a new vector store.")

if __name__ == "__main__":
    main()