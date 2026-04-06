import os
import sys
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# รองรับ Thai ใน Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

DB_DIR = "./chroma_db_v2"
DOCS_DIR = "./Docs"

def build_db():
    print("Finding PDF files in Docs/...")
    all_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + glob.glob(os.path.join(DOCS_DIR, "*.PDF"))
    pdf_files = list(set(all_files))
    
    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}.")
        return

    documents = []
    print(f"Loading {len(pdf_files)} files...")
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {os.path.basename(pdf_file)}")
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    print(f"Total chunks generated: {len(splits)}")

    print("Initializing embedding model (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    print(f"Creating Chroma vectorstore at {DB_DIR}...")
    if os.path.exists(DB_DIR):
        import shutil
        print("Cleaning up old DB...")
        shutil.rmtree(DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )

    print("Successfully built vector DB!")

if __name__ == "__main__":
    build_db()
