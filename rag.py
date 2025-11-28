"""
Retrieval-Augmented Generation (RAG) Module
Handles medical knowledge base creation, web crawling, and document retrieval
"""

import os
import requests
import pickle
import faiss
import numpy as np
from queue import Queue
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict
from config import (
    KB_PATH, INDEX_PATH, MAX_PAGES_TO_CRAWL, REQUEST_TIMEOUT,
    EMBEDDING_MODEL_NAME, FALLBACK_MEDICAL_KNOWLEDGE, TOP_K_RETRIEVAL,
    MIN_CONTENT_LENGTH, MAX_CONTENT_LENGTH, EMBEDDING_DIMENSION_FALLBACK
)


class WebBasedMedicalKnowledgeRAG:
    """
    Retrieval-Augmented Generation system for medical knowledge

    Handles:
    - Web crawling for medical content
    - Document indexing with FAISS
    - Semantic search using embeddings
    - Fallback knowledge base
    """

    def __init__(self,
                 knowledge_base_path: str = KB_PATH,
                 index_path: str = INDEX_PATH,
                 seed_urls: List[str] = None):
        """
        Initialize RAG system

        Args:
            knowledge_base_path: Path to save/load pickled knowledge base
            index_path: Path to save/load FAISS index
            seed_urls: List of URLs to crawl for medical information
        """
        self.knowledge_base_path = knowledge_base_path
        self.index_path = index_path
        self.index = None
        self.documents = None
        self.embeddings_model = None
        self.seed_urls = seed_urls or []
        self.visited_urls = set()
        self.max_pages = MAX_PAGES_TO_CRAWL

        # Initialize embedding model
        self._init_embeddings()

        # Load or create knowledge base
        if os.path.exists(knowledge_base_path) and os.path.exists(index_path):
            self._load_knowledge_base()
        else:
            self._initialize_knowledge_base()

    def _init_embeddings(self):
        """Initialize embedding model - tries SentenceTransformer, falls back to simple embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✓ Sentence-transformers loaded successfully")
        except ImportError as e:
            print(f"Warning: sentence-transformers import error: {e}")
            self.embeddings_model = None
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            self.embeddings_model = None

    def _initialize_knowledge_base(self):
        """
        Initialize knowledge base from websites
        Creates new FAISS index if not cached
        """
        if not self.seed_urls:
            print("No seed URLs provided. Using default medical URLs.")
            self.seed_urls = [
                "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
                "https://medlineplus.gov/pneumonia.html"
                "https://www.cdc.gov/pneumonia/index.html",
            ]

        print(f"Starting to crawl {len(self.seed_urls)} websites...")
        self.documents = []
        self._crawl_websites()

        if not self.documents:
            print("No documents crawled from websites. Using fallback knowledge base.")
            self._use_fallback_knowledge_base()

        self._build_faiss_index()
        self._save_knowledge_base()

    def _crawl_websites(self):
        """
        Crawl websites and extract medical content
        Uses breadth-first search limited to same domain
        """
        queue = Queue()
        for url in self.seed_urls:
            queue.put(url)

        doc_id = 0

        while not queue.empty() and len(self.documents) < self.max_pages:
            current_url = queue.get()

            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)

            try:
                print(f"Crawling: {current_url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(current_url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract title
                title = soup.find('h1') or soup.find('title')
                title_text = title.get_text(strip=True) if title else "Untitled"

                # Extract main content from common content containers
                content_tags = soup.find_all(
                    ['p', 'li', 'div'],
                    class_=lambda x: x and any(keyword in x.lower() for keyword in ['content', 'text', 'body'])
                )
                if not content_tags:
                    content_tags = soup.find_all(['p', 'li'])

                content_text = ' '.join([tag.get_text(strip=True) for tag in content_tags[:10]])

                if content_text and len(content_text) > MIN_CONTENT_LENGTH:
                    doc_id += 1
                    self.documents.append({
                        "id": doc_id,
                        "title": title_text,
                        "content": content_text[:MAX_CONTENT_LENGTH],
                        "url": current_url
                    })
                    print(f"✓ Extracted document {doc_id} from {current_url}")

                # Find links to crawl next (same domain only)
                links = soup.find_all('a', href=True)
                current_domain = urlparse(current_url).netloc

                for link in links[:5]:  # Limit links per page
                    href = link['href']
                    next_url = urljoin(current_url, href)
                    next_domain = urlparse(next_url).netloc

                    if next_domain == current_domain and next_url not in self.visited_urls:
                        queue.put(next_url)

            except requests.exceptions.RequestException as e:
                print(f"Error crawling {current_url}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {current_url}: {e}")

    def _use_fallback_knowledge_base(self):
        """Use fallback knowledge base if web crawling fails"""
        self.documents = FALLBACK_MEDICAL_KNOWLEDGE.copy()
        print(f"Using fallback knowledge base with {len(self.documents)} documents")

    def _build_faiss_index(self):
        """
        Build FAISS index for efficient similarity search
        Uses SentenceTransformer embeddings or fallback simple embeddings
        """
        if not self.documents:
            return

        # Prepare document texts for embedding
        contents = [f"{doc['title']}. {doc['content']}" for doc in self.documents]

        # Generate embeddings
        if self.embeddings_model:
            print(f"Generating embeddings for {len(contents)} documents using SentenceTransformer...")
            embeddings = self.embeddings_model.encode(contents)
        else:
            print(f"Generating embeddings for {len(contents)} documents using simple embeddings...")
            embeddings = self._simple_embeddings(contents)

        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✓ FAISS index built with {len(embeddings)} embeddings")

    def _simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Simple bag-of-words embedding fallback
        Creates vector representations of text using word frequency

        Args:
            texts: List of text documents

        Returns:
            numpy array of embeddings
        """
        # Build vocabulary
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())
        vocab = sorted(list(vocab))

        # Create embeddings
        embeddings = []
        for text in texts:
            vec = np.zeros(min(len(vocab), EMBEDDING_DIMENSION_FALLBACK))
            words = text.lower().split()
            for word in words:
                if word in vocab:
                    idx = vocab.index(word) % EMBEDDING_DIMENSION_FALLBACK
                    vec[idx] += 1
            embeddings.append(vec)

        return np.array(embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query string
            top_k: Number of top results to return

        Returns:
            List of retrieved documents with scores
        """
        if not self.index or not self.documents:
            return []

        # Generate query embedding
        if self.embeddings_model:
            query_embedding = self.embeddings_model.encode([query])
        else:
            query_embedding = self._simple_embeddings([query])

        query_embedding = np.array(query_embedding).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx].copy()
                results.append(doc)

        return results

    def _save_knowledge_base(self):
        """Save knowledge base and FAISS index to disk"""
        try:
            # Save documents
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"✓ Knowledge base saved to {self.knowledge_base_path}")

            # Save index
            if self.index:
                faiss.write_index(self.index, self.index_path)
                print(f"✓ FAISS index saved to {self.index_path}")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def _load_knowledge_base(self):
        """Load knowledge base and FAISS index from disk"""
        try:
            # Load documents
            with open(self.knowledge_base_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"✓ Knowledge base loaded from {self.knowledge_base_path} ({len(self.documents)} documents)")

            # Load index
            self.index = faiss.read_index(self.index_path)
            print(f"✓ FAISS index loaded from {self.index_path}")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.documents = None
            self.index = None

    def cleanup(self):
        """Delete local crawled data files"""
        files_to_remove = [self.knowledge_base_path, self.index_path]

        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ Cleanup: Deleted {file_path}")
            except Exception as e:
                print(f"Cleanup warning: Could not delete {file_path}: {e}")

    def get_knowledge_base_info(self) -> Dict:
        """Get information about current knowledge base"""
        return {
            "num_documents": len(self.documents) if self.documents else 0,
            "has_embeddings_model": self.embeddings_model is not None,
            "index_size": self.index.ntotal if self.index else 0,
            "knowledge_base_path": self.knowledge_base_path,
            "index_path": self.index_path
        }

    def __repr__(self):
        """String representation of RAG system"""
        info = self.get_knowledge_base_info()
        return (
            f"WebBasedMedicalKnowledgeRAG("
            f"documents={info['num_documents']}, "
            f"embeddings={'SentenceTransformer' if info['has_embeddings_model'] else 'Simple'}"
            f")"
        )

