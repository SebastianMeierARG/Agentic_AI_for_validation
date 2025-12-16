import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from config import CONFIG
import shutil

class RagEngine:
    def __init__(self):
        self.documents_path = CONFIG['paths']['documents_folder']
        self.vector_store = None
        
        # Load document language from config
        self.doc_language = CONFIG['rag_settings'].get('document_language', 'English')

        # Initialize LLM for translation/HyDE
        self.llm = ChatOpenAI(
            model=CONFIG['llm_settings']['model'],
            temperature=CONFIG['llm_settings']['temperature']
        )
        
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
        
        # Updated chunk settings from config
        chunk_size = CONFIG.get('rag_settings', {}).get('chunk_size', 1000)
        chunk_overlap = CONFIG.get('rag_settings', {}).get('chunk_overlap', 100)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)

        if splits:
            print(f"Creating vector store with {len(splits)} chunks (Size: {chunk_size}, Overlap: {chunk_overlap})...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            print("Index built successfully.")
        else:
            print("No text chunks created.")

    def generate_search_query(self, query):
        """Generates a hypothetical answer (HyDE) in the target document language."""
        system_prompt = (
            f"You are an expert Auditor. The user is asking: '{query}'.\n"
            f"Your task: Write a HYPOTHETICAL text snippet in {self.doc_language} that answers this question using technical banking vocabulary.\n"
            "Output ONLY the hypothetical statement."
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=system_prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content

    def retrieve(self, query, k=10):
        if not self.vector_store:
            # Attempt to build if not exists
            print("Vector store not found. Building index...")
            self.build_index()

        if not self.vector_store:
            return []
        
        # Use HyDE to generate search query
        print(f"DEBUG: Generating HyDE query in {self.doc_language}...")
        search_query = self.generate_search_query(query)
        print(f"Original Query: {query[:50]}...")
        print(f"HyDE Search Query: {search_query[:50]}...")

        return self.vector_store.similarity_search(search_query, k=k)
