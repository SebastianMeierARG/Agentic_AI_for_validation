import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import CONFIG

class RagEngine:
    def __init__(self):
        self.documents_path = CONFIG['paths']['documents_folder']
        self.vector_store = None
        # Initialize embeddings. Expects OPENAI_API_KEY in env.
        # If API key is missing, this might fail at instantiation or usage.
        # We handle it gracefully or let it fail if keys are strictly required.
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"Warning: Could not initialize OpenAIEmbeddings: {e}")
            self.embeddings = None

    def load_documents(self):
        docs = []
        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path)
            return docs

        print(f"Loading documents from {self.documents_path}...")
        for filename in os.listdir(self.documents_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.documents_path, filename)
                print(f"Loading {filename}...")
                try:
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return docs

    def build_index(self):
        if not self.embeddings:
            print("Embeddings not initialized. Cannot build index.")
            return

        docs = self.load_documents()
        if not docs:
            print("No documents found to index.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        if splits:
            print(f"Creating vector store with {len(splits)} chunks...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            print("Index built successfully.")
        else:
            print("No text chunks created.")

    def retrieve(self, query, k=5):
        if not self.vector_store:
            # Attempt to build if not exists
            print("Vector store not found. Building index...")
            self.build_index()

        if not self.vector_store:
            return []

        return self.vector_store.similarity_search(query, k=k)
