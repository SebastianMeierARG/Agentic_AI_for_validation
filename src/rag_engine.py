import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from config import CONFIG, PROJECT_ROOT
from llm_factory import get_llm, get_embeddings
import shutil
import time


class RagEngine:
    def __init__(self):
        self.documents_path = CONFIG['paths']['documents_folder']
        self.index_path_client = CONFIG['paths'].get(
            'faiss_index_client', str(PROJECT_ROOT / "faiss_index_client")
        )
        self.vector_store = None

        # Phase 2: Regulations Paths
        self.regulations_path = str(PROJECT_ROOT / "regulations")
        self.index_path_regs = str(PROJECT_ROOT / "faiss_index_regs")
        self.vector_store_regs = None

        self.doc_language = CONFIG['rag_settings'].get('document_language', 'English')
        self.llm = get_llm()

        try:
            self.embeddings = get_embeddings()
        except Exception as e:
            print(f"Warning: Could not initialize Embeddings: {e}")
            self.embeddings = None

        # Lazy-loaded CrossEncoder reranker
        self._reranker = None
        self._reranker_loaded = False

    def _get_reranker(self):
        """Lazy-load the CrossEncoder reranker once and cache it."""
        if not self._reranker_loaded:
            self._reranker_loaded = True
            try:
                from sentence_transformers import CrossEncoder
                model_name = CONFIG.get('rag_settings', {}).get(
                    'reranker_model', 'cross-encoder/stsb-roberta-base'
                )
                self._reranker = CrossEncoder(model_name)
                print(f"Reranker loaded: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load reranker ({e}). Skipping reranking.")
                self._reranker = None
        return self._reranker

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

        if os.path.exists(index_name):
            print(f"Loading existing index from {index_name}...")
            try:
                vector_store = FAISS.load_local(
                    index_name, self.embeddings, allow_dangerous_deserialization=True
                )
                print(f"Index {index_name} loaded successfully.")
                return vector_store
            except Exception as e:
                print(f"Error loading index {index_name}: {e}. Rebuilding...")

        docs = self.load_documents_from_folder(folder_path)
        if not docs:
            print(f"No documents found in {folder_path} to index.")
            return None

        chunk_size = CONFIG.get('rag_settings', {}).get('chunk_size', 1000)
        chunk_overlap = CONFIG.get('rag_settings', {}).get('chunk_overlap', 100)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
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
            batch = splits[i: i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch)} chunks)...")

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
        self.vector_store = self._build_or_load_index(
            self.index_path_client, self.documents_path
        )

    def ingest_regulations(self):
        self.vector_store_regs = self._build_or_load_index(
            self.index_path_regs, self.regulations_path
        )

    def _invoke_with_retry(self, messages, max_retries=5, base_delay=20):
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(
                            f"Rate limit hit during HyDE. Waiting {wait_time}s "
                            f"(retry {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    def generate_search_query(self, query):
        """Generates a hypothetical answer (HyDE) in the target document language."""
        system_prompt = (
            f"You are an expert Bank Auditor. The user is asking the following query about "
            f"interpreting a bank's policy:\n\n'{query}'\n\n"
            f"Your task: Write a detailed HYPOTHETICAL paragraph in {self.doc_language} that "
            "answers this question, exactly as it might appear in a bank's official model "
            "governance documentation or risk policy.\n"
            "Include technical IFRS 9 banking vocabulary, potential methodological synonyms "
            "(e.g., if asked about backtesting, mention PSI, stability, out-of-time validation), "
            "and specific examples of what compliance looks like.\n"
            "Output ONLY the hypothetical text."
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=system_prompt),
        ]
        response = self._invoke_with_retry(messages)
        return response.content

    def _filter_by_threshold(self, results_with_scores, threshold, label=""):
        """
        Discard chunks with L2 distance > threshold.
        Falls back to all results (sorted by score) if every chunk fails the threshold,
        so the LLM always receives something rather than an empty context.
        """
        filtered = [(doc, s) for doc, s in results_with_scores if s <= threshold]
        if not filtered:
            print(
                f"WARNING: No {label} chunks passed threshold ({threshold}). "
                "Using unfiltered fallback."
            )
            filtered = sorted(results_with_scores, key=lambda x: x[1])
        return filtered

    def retrieve(self, query, k=15, threshold_override=None, client_top_k_override=None):
        """
        Retrieve relevant document chunks using HyDE + source-balanced retrieval
        + score threshold + multilingual CrossEncoder reranking.

        Key design decisions:
          - Client docs and regulation docs are queried separately with independent
            per-source caps (client_top_k / regs_top_k). This prevents regulation
            documents from crowding out client policy pages when both are scored
            together in a single pool.
          - Threshold filter uses a permissive default (1.8) so valid chunks are not
            discarded due to the naturally higher L2 distances that arise from
            cross-lingual embeddings (non-English docs vs. HyDE query in the target
            language embedded alongside English training data).
          - The CrossEncoder reranker uses a multilingual MS MARCO model that handles
            Spanish, German, French, and other languages natively.
        """
        use_regs = CONFIG.get('rag_settings', {}).get('use_regulations', True)

        if not self.vector_store:
            print("Client vector store not found. Building...")
            self.build_index()
        if use_regs and not self.vector_store_regs:
            print("Regulations vector store not found. Checking/Building...")
            self.ingest_regulations()

        print(f"DEBUG: Generating HyDE query in {self.doc_language}...")
        search_query = self.generate_search_query(query)
        print(f"Original Query: {query[:80]}...")
        print(f"HyDE Search Query: {search_query[:80]}...")

        rag_cfg = CONFIG.get('rag_settings', {})
        threshold = threshold_override if threshold_override is not None else rag_cfg.get('retrieval_score_threshold', 1.8)
        client_top_k = client_top_k_override if client_top_k_override is not None else rag_cfg.get('client_top_k', 8)
        regs_top_k = rag_cfg.get('regs_top_k', 4)
        rerank_top_k = rag_cfg.get('rerank_top_k', 10)

        # --- Step 1: Source-balanced retrieval ---
        # Query each index independently with its own cap so neither source dominates.
        client_candidates = []
        if self.vector_store:
            raw = self.vector_store.similarity_search_with_score(search_query, k=client_top_k * 2)
            client_candidates = self._filter_by_threshold(raw, threshold, label="client")
            client_candidates = client_candidates[:client_top_k]

        regs_candidates = []
        if use_regs and self.vector_store_regs:
            raw_regs = self.vector_store_regs.similarity_search_with_score(search_query, k=regs_top_k * 2)
            for doc, score in raw_regs:
                doc.metadata['source_type'] = 'regulation'
            regs_candidates = self._filter_by_threshold(raw_regs, threshold, label="regulation")
            regs_candidates = regs_candidates[:regs_top_k]

        combined = [doc for doc, _ in client_candidates] + [doc for doc, _ in regs_candidates]
        print(
            f"Source-balanced pool: {len([d for d, _ in client_candidates])} client + "
            f"{len([d for d, _ in regs_candidates])} regulation chunks."
            + ("" if use_regs else " (regulations disabled)")
        )

        if not combined:
            print("WARNING: No chunks retrieved from any source.")
            return []

        # --- Step 2: Multilingual CrossEncoder reranking ---
        reranker = self._get_reranker()

        if reranker and len(combined) > 1:
            try:
                # HyDE query (target document language) paired with each chunk.
                pairs = [[search_query, doc.page_content] for doc in combined]
                rerank_scores = reranker.predict(pairs)
                ranked = sorted(
                    zip(combined, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                final_docs = [doc for doc, _ in ranked[:rerank_top_k]]
                print(
                    f"Reranking: {len(combined)} chunks -> top {len(final_docs)} selected."
                )
            except Exception as e:
                print(f"Reranking failed ({e}). Using source-balanced unranked results.")
                final_docs = combined[:rerank_top_k]
        else:
            final_docs = combined[:rerank_top_k]

        return final_docs
