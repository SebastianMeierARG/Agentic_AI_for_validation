import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from config import CONFIG
from llm_factory import get_llm, get_embeddings
import shutil
import time

class RagEngine:
    def __init__(self):
        self.documents_path = CONFIG['paths']['documents_folder']
        self.index_path_client = "faiss_index_client"
        self.vector_store = None
        
        # Phase 2: Regulations Paths
        self.regulations_path = "regulations"
        self.index_path_regs = "faiss_index_regs"
        self.vector_store_regs = None

        # Load document language from config
        self.doc_language = CONFIG['rag_settings'].get('document_language', 'English')

        # Initialize LLM for translation/HyDE
        self.llm = get_llm()
        
        try:
            self.embeddings = get_embeddings()
        except Exception as e:
            print(f"Warning: Could not initialize Embeddings: {e}")
            self.embeddings = None

    def load_documents_from_folder(self, folder_path):
        docs = []
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return docs

        print(f"Loading documents from {folder_path}...")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                print(f"Loading {filename}...")
                try:
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return docs

    def _build_or_load_index(self, index_name, folder_path):
        """Helper to build or load an index."""
        if not self.embeddings:
            print("Embeddings not initialized.")
            return None

        # Check if index exists on disk
        if os.path.exists(index_name):
            print(f"Loading existing index from {index_name}...")
            try:
                vector_store = FAISS.load_local(index_name, self.embeddings, allow_dangerous_deserialization=True)
                print(f"Index {index_name} loaded successfully.")
                return vector_store
            except Exception as e:
                print(f"Error loading index {index_name}: {e}. Rebuilding...")

        # Build from scratch
        docs = self.load_documents_from_folder(folder_path)
        if not docs:
            print(f"No documents found in {folder_path} to index.")
            return None
        
        chunk_size = CONFIG.get('rag_settings', {}).get('chunk_size', 1000)
        chunk_overlap = CONFIG.get('rag_settings', {}).get('chunk_overlap', 100)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)

        if not splits:
            print("No text chunks created.")
            return None

        print(f"Creating vector store for {index_name} with {len(splits)} chunks...")
        
        batch_size = 10
        delay_seconds = 5
        vector_store = None
        
        total_batches = (len(splits) + batch_size - 1) // batch_size
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} chunks)...")
            
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                vector_store.add_documents(batch)
            
            if i + batch_size < len(splits):
                time.sleep(delay_seconds)

        print(f"Saving index to {index_name}...")
        vector_store.save_local(index_name)
        print(f"Index {index_name} built and saved successfully.")
        return vector_store

    def build_index(self):
        """Builds/Loads the Client Index."""
        self.vector_store = self._build_or_load_index(self.index_path_client, self.documents_path)

    def ingest_regulations(self):
        """Builds/Loads the Regulations Index."""
        self.vector_store_regs = self._build_or_load_index(self.index_path_regs, self.regulations_path)

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
        from google.api_core.exceptions import ResourceExhausted

        max_retries = 5
        base_delay = 20 

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as e:
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
        # Ensure indices are ready
        if not self.vector_store:
            print("Client Vector store not found. Building...")
            self.build_index()
        
        # Optional: Load regulations if not loaded, but only if they exist or we want to force it.
        # For now, let's try to load/build it if it's not ready, effectively making it part of the default init or on-demand.
        if not self.vector_store_regs:
             print("Regulations Vector store not found. Checking/Building...")
             self.ingest_regulations()

        # Use HyDE to generate search query
        print(f"DEBUG: Generating HyDE query in {self.doc_language}...")
        search_query = self.generate_search_query(query)
        print(f"Original Query: {query[:50]}...")
        print(f"HyDE Search Query: {search_query[:50]}...")

        results = []
        
        # Retrieve from Client Docs
        if self.vector_store:
            results.extend(self.vector_store.similarity_search(search_query, k=k))

        # Retrieve from Regulations (if any)
        if self.vector_store_regs:
            # We might want to distinguish sources or limit total K?
            # Let's add top k from regs too.
            regs_results = self.vector_store_regs.similarity_search(search_query, k=k)
            # Mark them as from regulations if possible (metadata update? FAISS docs are copies)
            for doc in regs_results:
                doc.metadata['source_type'] = 'regulation'
            results.extend(regs_results)

        return results
