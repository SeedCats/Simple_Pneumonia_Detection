"""
RAG Module - Medical knowledge base and document retrieval
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
    """RAG system for medical knowledge retrieval"""

    def __init__(self, knowledge_base_path: str = KB_PATH, index_path: str = INDEX_PATH, seed_urls: List[str] = None):
        self.knowledge_base_path = knowledge_base_path
        self.index_path = index_path
        self.index = None
        self.documents = None
        self.seed_urls = seed_urls or []
        self.visited_urls = set()
        self.max_pages = MAX_PAGES_TO_CRAWL

        self._init_embeddings()

        if os.path.exists(knowledge_base_path) and os.path.exists(index_path):
            self._load_knowledge_base()
        else:
            self._initialize_knowledge_base()

    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✓ Sentence-transformers loaded")
        except Exception as e:
            print(f"Warning: Embedding model failed: {e}")
            self.embeddings_model = None

    def _initialize_knowledge_base(self):
        """Initialize knowledge base from websites"""
        print(f"Crawling {len(self.seed_urls)} websites...")
        self.documents = []
        self._crawl_websites()

        if not self.documents:
            print("Using fallback knowledge base.")
            self.documents = FALLBACK_MEDICAL_KNOWLEDGE.copy()

        self._build_faiss_index()
        self._save_knowledge_base()

    def _crawl_websites(self):
        """Crawl websites and extract medical content"""
        queue = Queue()
        for url in self.seed_urls:
            queue.put(url)

        doc_id = 0
        while not queue.empty() and len(self.documents) < self.max_pages:
            url = queue.get()
            if url in self.visited_urls:
                continue
            self.visited_urls.add(url)

            try:
                print(f"Crawling: {url}")
                resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, 'html.parser')

                title = (soup.find('h1') or soup.find('title'))
                title_text = title.get_text(strip=True) if title else "Untitled"

                content_tags = soup.find_all(['p', 'li'], limit=10)
                content = ' '.join(tag.get_text(strip=True) for tag in content_tags)

                if len(content) > MIN_CONTENT_LENGTH:
                    doc_id += 1
                    self.documents.append({
                        "id": doc_id, "title": title_text,
                        "content": content[:MAX_CONTENT_LENGTH], "url": url
                    })
                    print(f"✓ Extracted doc {doc_id}")

                # Queue same-domain links
                domain = urlparse(url).netloc
                for link in soup.find_all('a', href=True)[:5]:
                    next_url = urljoin(url, link['href'])
                    if urlparse(next_url).netloc == domain and next_url not in self.visited_urls:
                        queue.put(next_url)
            except Exception as e:
                print(f"Error crawling {url}: {e}")

    def _build_faiss_index(self):
        """Build FAISS index for similarity search"""
        if not self.documents:
            return

        contents = [f"{d['title']}. {d['content']}" for d in self.documents]

        if self.embeddings_model:
            embeddings = self.embeddings_model.encode(contents)
        else:
            embeddings = self._simple_embeddings(contents)

        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✓ FAISS index built with {len(embeddings)} embeddings")

    def _simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple bag-of-words embedding fallback"""
        vocab = sorted(set(word for text in texts for word in text.lower().split()))
        embeddings = []
        for text in texts:
            vec = np.zeros(min(len(vocab), EMBEDDING_DIMENSION_FALLBACK))
            for word in text.lower().split():
                if word in vocab:
                    vec[vocab.index(word) % EMBEDDING_DIMENSION_FALLBACK] += 1
            embeddings.append(vec)
        return np.array(embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.index or not self.documents:
            return []

        embedding = self.embeddings_model.encode([query]) if self.embeddings_model else self._simple_embeddings([query])
        embedding = np.array(embedding).astype('float32')

        _, indices = self.index.search(embedding, top_k)
        return [self.documents[i].copy() for i in indices[0] if 0 <= i < len(self.documents)]

    def _save_knowledge_base(self):
        """Save knowledge base and index to disk"""
        try:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.documents, f)
            if self.index:
                faiss.write_index(self.index, self.index_path)
            print(f"✓ Knowledge base saved")
        except Exception as e:
            print(f"Error saving: {e}")

    def _load_knowledge_base(self):
        """Load knowledge base and index from disk"""
        try:
            with open(self.knowledge_base_path, 'rb') as f:
                self.documents = pickle.load(f)
            self.index = faiss.read_index(self.index_path)
            print(f"✓ Loaded {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading: {e}")
            self.documents, self.index = None, None

    def cleanup(self):
        """Delete local data files"""
        for path in [self.knowledge_base_path, self.index_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"✓ Deleted {path}")
            except Exception as e:
                print(f"Cleanup warning: {e}")

    def __repr__(self):
        return f"RAG(docs={len(self.documents) if self.documents else 0})"
