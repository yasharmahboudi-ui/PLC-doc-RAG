import os
import time

def ingest_documents():
    # Target paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_path = os.path.join(base_dir, "data", "s71200_easy_book.pdf")
    vector_store_dir = os.path.join(base_dir, "vector_store")
    
    if not os.path.exists(pdf_path):
        print(f"Error: S7-1200 PDF not found at {pdf_path}.")
        print("Please run 'python src/download_manual.py' first.")
        return

    print("Importing LangChain components (this may take a few seconds)...")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please run 'pip install -r requirements.txt' inside your virtual environment.")
        return

    print(f"Loading PDF from: {pdf_path}...")
    start_time = time.time()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages in {time.time() - start_time:.2f} seconds.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} text chunks.")

    print("Initializing HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    # This downloads and loads the model locally on CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("Creating FAISS vector store index (embedding chunks)...")
    embed_start = time.time()
    db = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created in {time.time() - embed_start:.2f} seconds.")

    # Save to disk
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)
        
    db.save_local(os.path.join(vector_store_dir, "faiss_index"))
    print(f"Vector store index saved successfully to: {os.path.join(vector_store_dir, 'faiss_index')}")

if __name__ == "__main__":
    ingest_documents()
