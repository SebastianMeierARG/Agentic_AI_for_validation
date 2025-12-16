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
        
        # Initialize LLM for translation
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

    def translate_query(self, query):
        """Translates an English query to Spanish using the LLM."""
        system_prompt = "You are a helpful translator. Translate the following audit query from English to Spanish to match technical banking documentation language."
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
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
        
        # Translate query if needed (assuming input is English, docs are Spanish)
        # We'll just always translate to ensure better matching if the query is English-like
        # A simple check could be added, but the instruction implies "if the query is in English"
        # We will assume the audit queries are English as per the CSV content
        translated_query = self.translate_query(query)
        print(f"Original Query: {query[:50]}...")
        print(f"Translated Query: {translated_query[:50]}...")

        return self.vector_store.similarity_search(translated_query, k=k)
