import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from config import CONFIG
from llm_factory import get_llm, get_embeddings
import shutil

class RagEngine:
    def __init__(self):
        self.documents_path = CONFIG['paths']['documents_folder']
        self.vector_store = None
        
        # Load document language from config
        self.doc_language = CONFIG['rag_settings'].get('document_language', 'English')

        # Initialize LLM for translation/HyDE
        self.llm = get_llm()
        
        try:
            self.embeddings = get_embeddings()
        except Exception as e:
            print(f"Warning: Could not initialize Embeddings: {e}")
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
            
            import time
            from langchain_core.documents import Document

            batch_size = 10  # Process 10 chunks at a time to avoid rate limits
            delay_seconds = 5 # Wait 5 seconds between batches

            self.vector_store = None
            
            total_batches = (len(splits) + batch_size - 1) // batch_size
            for i in range(0, len(splits), batch_size):
                batch = splits[i : i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} chunks)...")
                
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    self.vector_store.add_documents(batch)
                
                if i + batch_size < len(splits):
                    time.sleep(delay_seconds)

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
        response = None
        # Retry logic for generation
        import time
        from google.api_core.exceptions import ResourceExhausted

        max_retries = 5
        base_delay = 20 

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as e:
                # Check for ResourceExhausted or similar 429
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt) 
                        print(f"Rate limit hit during HyDE. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise e 
                else:
                    raise e
        
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
