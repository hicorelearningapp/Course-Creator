import time
import json
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from openai_client import AIClient

# ============================================================
# 🕸️ CONFIGURATION
# ============================================================
@dataclass
class WebFetcherConfig:
    max_results: int = 8
    max_snippet_length: int = 600
    request_delay: float = 1.0
    timeout: int = 15

# ============================================================
# 🕸️ WEB FETCHER (uses DuckDuckGo Search + BeautifulSoup)
# ============================================================
class WebContentFetcher:
    def __init__(self, config: Optional[WebFetcherConfig] = None):
        self.config = config or WebFetcherConfig()

    def extract_text(self, url: str) -> str:
        """Extract readable text from a given URL."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=self.config.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")

            for tag in ["script", "style", "nav", "footer", "header", "aside", "form"]:
                [t.decompose() for t in soup.find_all(tag)]

            selectors = [
                "article", "main", ".content", ".post-content",
                ".entry-content", "#content", "#main-content",
                ".story-content", ".article-content", ".text-content"
            ]
            text = None
            for sel in selectors:
                elem = soup.select_one(sel)
                if elem and elem.get_text(strip=True):
                    text = elem.get_text(" ", strip=True)
                    break

            if not text:
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = " ".join(p for p in paragraphs if len(p) > 30)

            text = re.sub(r"\s+", " ", text).strip()
            return text[:8000] if len(text) > 8000 else text

        except Exception as e:
            return f"⚠️ Error extracting {url}: {e}"

    def search(self, query: str) -> List[Dict[str, str]]:
        """Perform DuckDuckGo search and extract main content from each result."""
        results = []
        try:
            print(f"🔍 Searching for: '{query}'")
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=self.config.max_results, region='wt-wt'))
            print(f"📄 Found {len(hits)} results")

            for i, h in enumerate(hits):
                url = h.get("href") or h.get("url")
                title = h.get("title", "Untitled")
                snippet = h.get("body", "") or h.get("description", "")
                if not url:
                    continue

                print(f"🔗 [{i+1}] Fetching content from: {url[:60]}")
                content = self.extract_text(url)
                if not content.startswith("⚠️"):
                    results.append({
                        "rank": i + 1,
                        "title": title,
                        "url": url,
                        "snippet": snippet[:200],
                        "content": content
                    })
                    print(f"   ✅ Extracted {len(content)} characters")
                time.sleep(self.config.request_delay)
        except Exception as e:
            print(f"❌ Search error: {e}")
        return results

# ============================================================
# 🧩 VECTOR BUILDER
# ============================================================
class VectorBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        self.embed_model = SentenceTransformer(model_name)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.docs_store = []
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_text(self, text: str):
        chunks = self.splitter.create_documents([text])
        return [c.page_content.strip() for c in chunks if c.page_content.strip()]

    def add_text(self, text: str):
        chunks = self.chunk_text(text)
        for chunk in chunks:
            vec = self.embed_model.encode(chunk)
            self.index.add(np.array([vec], dtype=np.float32))
            self.docs_store.append(chunk)
        return len(chunks)

    def query(self, question: str, top_k: int = 3):
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add text before querying.")
        qvec = self.embed_model.encode(question)
        D, I = self.index.search(np.array([qvec], dtype=np.float32), top_k)
        return [self.docs_store[i] for i in I[0] if i < len(self.docs_store)]

# ============================================================
# 🔗 WEB → VECTOR → AI RESPONSE (RAG PIPELINE)
# ============================================================
class WebSearchRAG:
    def __init__(self, ai_client: AIClient, search_results: int = 8, top_chunks: int = 5):
        self.ai_client = ai_client
        self.fetcher = WebContentFetcher(WebFetcherConfig(max_results=search_results))
        self.vb = VectorBuilder()
        self.top_chunks = top_chunks
        self.sources = []

    def build_index(self, query: str):
        """Fetch and index web content for a topic."""
        results = self.fetcher.search(query)
        self.sources = [{"title": r["title"], "url": r["url"]} for r in results[:5]]
        total_chunks = 0
        for r in results:
            if not r.get("content", ""):
                continue
            total_chunks += self.vb.add_text(r['content'])
        print(f"✅ Indexed {total_chunks} text chunks.")
        return total_chunks

    def answer_with_web(self, query: str) -> str:
        """Generate an AI answer grounded in real-time web data."""
        print(f"\n🌐 Starting web-assisted answer for: {query}")
        if self.vb.index.ntotal == 0:
            self.build_index(query)

        top_contexts = self.vb.query(query, top_k=self.top_chunks)
        context = "\n\n".join(top_contexts)

        messages = [
            {"role": "system", "content": "You are an expert assistant that summarizes and explains using verified, web-fetched content."},
            {"role": "user", "content": f"Context from web:\n{context}\n\nQuestion: {query}\n\nProvide a clear, visual, and up-to-date answer with examples."}
        ]

        response = self.ai_client.get_completion(messages, temperature=0.7, max_tokens=900)
        answer = response.choices[0].message.content
        print("\n💬 Final Answer Generated.")
        return answer
