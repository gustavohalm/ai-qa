import os
import time
from typing import List, Tuple, Set, Dict, Optional
from urllib.parse import urlparse, urljoin
import glob
from pathlib import Path
import threading

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class AIService:
    """
    AI Service class for handling iPhone 17 related questions
    Uses Langchain for LLM orchestration and Qdrant for vector storage
    """
    
    def __init__(self):
        """Initialize the AI service with Langchain and Qdrant"""
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        
        # Initialize Qdrant client
        qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=os.environ.get('QDRANT_API_KEY'))
        
        # Collection name for iPhone 17 knowledge base
        self.collection_name = "iphone17_knowledge"
        # Seed URLs (Apple Support)
        self.seed_urls: List[str] = [
            os.environ.get('SEED_URL_IPHONE_17', 'https://support.apple.com/en-us/125089'),  # iPhone 17 - Tech Specs
            os.environ.get('SEED_URL_IPHONE_USER_GUIDE', 'https://support.apple.com/guide/iphone/welcome/26/ios/26'),  # iPhone User Guide
            os.environ.get('SEED_URL_IPHONE_17_PRO_MAX', 'https://support.apple.com/en-us/125091'),  # iPhone 17 Pro Max - Tech Specs
            os.environ.get('SEED_URL_IPHONE_PRO_DOCS', 'https://support.apple.com/en-us/docs/iphone/301245'),  # iPhone Pro docs hub
        ]
        # Crawl settings
        self.crawl_max_depth: int = int(os.environ.get('CRAWL_MAX_DEPTH', '1'))
        self.crawl_max_pages: int = int(os.environ.get('CRAWL_MAX_PAGES', '30'))
        self.request_timeout_sec: int = int(os.environ.get('CRAWL_REQUEST_TIMEOUT_SEC', '15'))
        self.request_delay_sec: float = float(os.environ.get('CRAWL_REQUEST_DELAY_SEC', '0.5'))
        self.enable_web_fallback: bool = os.environ.get('ENABLE_WEB_FALLBACK', 'true').lower() == 'true'
        # Ingestion concurrency control
        self._ingest_lock = threading.Lock()
        self._ingest_running = False
        
        # Initialize or connect to Qdrant collection
        self._initialize_qdrant_collection()
        
        # Agent setup (tools + policy) replaces the previous LLMChain flow
        self.agent = self._build_agent()
        
        # Load or build iPhone 17 knowledge base from Apple Support via crawling
        self._load_iphone17_knowledge()
    
    def _initialize_qdrant_collection(self):
        """Initialize Qdrant collection for storing iPhone 17 knowledge"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection with appropriate vector size (1536 for OpenAI embeddings)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant collection: {e}")
            print("The service will work with limited functionality")
    
    def _load_iphone17_knowledge(self):
        """Load iPhone 17 knowledge into the vector store from Apple Support pages"""
        try:
            # Initialize Langchain Qdrant vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            # Add documents to vector store if collection is empty
            try:
                # Check if collection has any points
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info.points_count == 0:
                    print("Collection empty. Starting crawl and ingestion from Apple Support seed URLs...")
                    pages = self._crawl_seed_urls(self.seed_urls)
                    self._ingest_pages(pages)
                    print(f"Ingested {len(pages)} pages from Apple Support into Qdrant")
                    # Also ingest local PDF Product Environmental Report(s) if present
                    self._ingest_pdf_reports()
                else:
                    print(f"Collection already contains {collection_info.points_count} documents")
            except Exception as e:
                print(f"Could not check collection: {e}")
                
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            self.vector_store = None
    
    def _build_agent(self):
        """Construct a LangChain agent with tools for retrieval and web search."""
        # Tools defined with @tool decorator per LangChain docs
        ddg = DuckDuckGoSearchRun()
        
        @tool("retrieve_context")
        def retrieve_context_tool(query: str) -> str:
            """Retrieve relevant context from Qdrant about iPhone 17 and related Apple Support docs. Always call this first."""
            return self.retrieve_context(query)
        
        @tool("web_search")
        def web_search_tool(query: str) -> str:
            """General web search (DuckDuckGo) for iPhone information when retrieve_context returns NO_CONTEXT or insufficient details."""
            try:
                return ddg.run(f"{query} site:support.apple.com OR iPhone 17")
            except Exception as e:
                return f"Web search error: {e}"
        
        tools = [retrieve_context_tool, web_search_tool]
        
        system_prompt = (
            "You are an expert on the iPhone 17. "
            "First, ALWAYS call the 'retrieve_context' tool with the user's question. "
            "If it returns 'NO_CONTEXT' or lacks specific details to answer, then call 'web_search'. "
            "Prefer Apple Support pages when available. "
            "When you include information supported by sources, end your answer with a short 'Sources:' list of URLs. "
            "Be accurate and concise."
        )
        
        try:
            agent = create_agent(
                self.llm,
                tools=tools,
                system_prompt=system_prompt
            )
            return agent
        except Exception as e:
            print(f"Warning: Could not create agent, falling back to basic chain. Error: {e}")
            from basic_agent import BasicAgent
            return BasicAgent(self.llm, system_prompt)
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve top passages from Qdrant for iPhone 17 knowledge. Returns 'NO_CONTEXT' if nothing useful is found."""
        try:
            if not self.vector_store:
                return "NO_CONTEXT"
            # Prefer with_score; fall back if unavailable
            try:
                results = self.vector_store.similarity_search_with_score(query, k=4)
            except Exception:
                docs = self.vector_store.similarity_search(query, k=4)
                results = [(d, 0.0) for d in docs]
            chunks: List[str] = []
            sources: List[str] = []
            for doc, _score in results:
                chunks.append(doc.page_content)
                src = (doc.metadata or {}).get("source")
                if src:
                    sources.append(src)
            joined = "\n\n".join(chunks).strip()
            if not joined or len(joined) < 200:
                return "NO_CONTEXT"
            unique_sources = list(dict.fromkeys(sources))[:8]
            if unique_sources:
                return f"{joined}\n\nSources:\n" + "\n".join(unique_sources)
            return joined
        except Exception:
            return "NO_CONTEXT"
    
    def _perform_full_ingestion(self):
        """Crawl Apple Support seed URLs and ingest PDFs from data/."""
        if not self.vector_store:
            return
        try:
            print("Starting full ingestion in background...")
            pages = self._crawl_seed_urls(self.seed_urls)
            if pages:
                self._ingest_pages(pages)
                print(f"Ingested {len(pages)} pages from Apple Support into Qdrant")
            self._ingest_pdf_reports()
            print("Background ingestion finished.")
        except Exception as e:
            print(f"Background ingestion error: {e}")
        finally:
            with self._ingest_lock:
                self._ingest_running = False
    
    def start_background_ingestion(self) -> bool:
        """
        Start ingestion in a background thread. Returns True if started, False if already running.
        """
        with self._ingest_lock:
            if self._ingest_running:
                return False
            self._ingest_running = True
        th = threading.Thread(target=self._perform_full_ingestion, daemon=True)
        th.start()
        return True
    
    def _is_allowed_url(self, url: str, seed_hosts: Set[str], allowed_prefixes: List[str]) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        if parsed.netloc not in seed_hosts:
            return False
        # Only allow staying under the provided seed path prefixes (to keep crawl tight)
        normalized = url.split('#')[0]
        return any(normalized.startswith(prefix) for prefix in allowed_prefixes)

    def _fetch_page(self, url: str) -> Tuple[str, str]:
        """Return (title, text) for a URL or ('', '') on failure."""
        try:
            resp = requests.get(url, timeout=self.request_timeout_sec, headers={"User-Agent": "ai-test-crawler/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else url
            # Remove script/style
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            text = soup.get_text(separator='\n')
            # Normalize whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)
            return title, clean_text
        except Exception:
            return "", ""

    def _extract_links(self, base_url: str, html_text: str) -> List[str]:
        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                abs_url = urljoin(base_url, href)
                links.append(abs_url)
            return links
        except Exception:
            return []

    def _crawl_seed_urls(self, seeds: List[str]) -> List[Dict[str, str]]:
        """Crawl starting from seed URLs up to depth and page limits, constrained to Apple Support."""
        seed_hosts: Set[str] = set(urlparse(u).netloc for u in seeds)
        allowed_prefixes: List[str] = [u.split('#')[0].rstrip('/') for u in seeds]
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(u, 0) for u in seeds]
        pages: List[Dict[str, str]] = []

        while queue and len(pages) < self.crawl_max_pages:
            url, depth = queue.pop(0)
            normalized = url.split('#')[0]
            if normalized in visited:
                continue
            if not self._is_allowed_url(normalized, seed_hosts, allowed_prefixes):
                continue

            visited.add(normalized)
            title, text = self._fetch_page(normalized)
            if title and text:
                pages.append({"url": normalized, "title": title, "text": text})
                # If we can go deeper, extract links from this page
                if depth < self.crawl_max_depth:
                    # Use original HTML again for links for efficiency
                    try:
                        resp = requests.get(normalized, timeout=self.request_timeout_sec, headers={"User-Agent": "ai-test-crawler/1.0"})
                        resp.raise_for_status()
                        child_links = self._extract_links(normalized, resp.text)
                        for link in child_links:
                            if len(pages) + len(queue) >= self.crawl_max_pages:
                                break
                            if self._is_allowed_url(link, seed_hosts, allowed_prefixes) and link.split('#')[0] not in visited:
                                queue.append((link, depth + 1))
                    except Exception:
                        pass
            time.sleep(self.request_delay_sec)

        return pages

    def _ingest_pages(self, pages: List[Dict[str, str]]):
        if not self.vector_store or not pages:
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for page in pages:
            chunks = splitter.split_text(page["text"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({"source": page["url"], "title": page["title"]})
        if texts:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def _ingest_pdf_reports(self):
        """Parse and ingest local Product Environmental Report PDFs under data/"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_dir = os.path.join(base_dir, "data")
            if not os.path.isdir(pdf_dir):
                return
            pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
            if not pdf_paths:
                return
            pages: List[Dict[str, str]] = []
            for pdf_path in pdf_paths:
                try:
                    reader = PdfReader(pdf_path)
                    extracted_parts: List[str] = []
                    for page in reader.pages:
                        try:
                            content = page.extract_text() or ""
                        except Exception:
                            content = ""
                        if content.strip():
                            extracted_parts.append(content)
                    full_text = "\n".join(extracted_parts).strip()
                    if not full_text:
                        continue
                    title = Path(pdf_path).name
                    pages.append({
                        "url": f"file://{pdf_path}",
                        "title": title,
                        "text": full_text
                    })
                except Exception:
                    continue
            if pages:
                print(f"Ingesting {len(pages)} PDF report(s) from data/ into Qdrant")
                self._ingest_pages(pages)
        except Exception:
            # Non-fatal if PDF ingestion fails
            pass
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about iPhone 17
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The answer to the question
        """
        try:
            # Retrieve relevant context from vector store
            context = ""
            sources: List[str] = []
            if self.vector_store:
                try:
                    docs_with_scores = self.vector_store.similarity_search_with_score(question, k=4)
                except Exception:
                    # Fallback if with_score not available
                    docs = self.vector_store.similarity_search(question, k=4)
                    docs_with_scores = [(doc, 0.0) for doc in docs]
                # Build context and track sources
                top_docs: List[str] = []
                for doc, score in docs_with_scores:
                    top_docs.append(doc.page_content)
                    if doc.metadata and "source" in doc.metadata:
                        sources.append(doc.metadata["source"])
                context = "\n".join(top_docs)
            else:
                context = "Limited information available about iPhone 17."
            
            # Use the agent to answer. Agent will decide to call tools.
            try:
                # Some runtimes require messages format
                agent_result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
            except Exception:
                # Fallback: try simple call style
                try:
                    agent_result = self.agent.invoke({"input": question})
                except Exception as e:
                    return f"I apologize, but I encountered an error while processing your question: {str(e)}"
            
            # Normalize output to plain text
            if isinstance(agent_result, str):
                return agent_result
            if isinstance(agent_result, dict):
                # Common keys used by various agent executors
                for key in ["output_text", "content", "output", "final_output", "answer", "result"]:
                    value = agent_result.get(key)
                    if isinstance(value, str) and value.strip():
                        return value

                # LangChain/LangGraph style: { messages: [HumanMessage, AIMessage, ToolMessage, AIMessage, ...] }
                msgs = agent_result.get("messages")
                if isinstance(msgs, list) and msgs:
                    # Walk backwards to find the last non-empty textual content
                    for item in reversed(msgs):
                        # Item can be a dict or an object; prefer attribute first
                        text = None
                        if hasattr(item, "content"):
                            text = getattr(item, "content")
                        elif isinstance(item, dict):
                            maybe_content = item.get("content")
                            text = maybe_content
                        # Some message formats have content as a list of parts
                        if isinstance(text, list):
                            # Extract text fields from parts if any
                            try:
                                parts_text = "".join(
                                    part.get("text", "") if isinstance(part, dict) else ""
                                    for part in text
                                )
                                text = parts_text
                            except Exception:
                                text = ""
                        if isinstance(text, str) and text.strip():
                            return text

                # Fallback: stringify but this includes tool traces; avoid if possible
                # If no clean text was found, return empty string to avoid noisy outputs
                return ""

            # Unknown structure; avoid dumping reprs with tool traces
            return ""
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def add_knowledge(self, text: str):
        """
        Add new knowledge about iPhone 17 to the vector store
        
        Args:
            text (str): New information to add
        """
        if self.vector_store:
            self.vector_store.add_texts([text])
            print(f"Added new knowledge to vector store")
        else:
            print("Vector store not available")

