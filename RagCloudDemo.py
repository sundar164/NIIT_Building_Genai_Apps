"""
GenAI RAG Application - Layered Architecture Demo
A Tkinter application showcasing all 11 architecture layers with sample data
for classroom training purposes.

LAYERS DEMONSTRATED:
1. Presentation Layer (Tkinter GUI)
2. API Layer (FastAPI schemas)
3. Business Logic (RAG Pipeline)
4. Data Access (AWS Adapters)
5. Database (Models)
6-10. Infrastructure/Deployment/Config/Testing/CI-CD (Documentation)
11. Monitoring (CloudWatch simulation)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================================
# LAYER 5: DATABASE MODELS
# ============================================================================

@dataclass
class Document:
    """Database model for documents"""
    id: int
    title: str
    content: str
    embedding_id: str
    score: float


# ============================================================================
# LAYER 4: DATA ACCESS LAYER - AWS ADAPTERS
# ============================================================================

class AWSAdapter:
    """Base AWS adapter for simulating AWS services"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.call_count = 0

    def log_call(self, method: str, params: Dict):
        self.call_count += 1
        return {'service': self.service_name, 'method': method, 'call_number': self.call_count}


class SageMakerAdapter(AWSAdapter):
    """Simulates SageMaker endpoints"""

    def __init__(self):
        super().__init__("SageMaker")
        self.responses = [
            "SageMaker is a fully managed ML service for building, training, and deploying models",
            "Machine learning models are deployed as endpoints for low-latency inference",
            "SageMaker supports real-time and batch inference modes for different use cases",
            "Embeddings are generated using SageMaker's text embedding models"
        ]

    def embed_text(self, text: str) -> List[float]:
        self.log_call("embed_text", {"text": text[:50]})
        return [0.1 * (i % 10) for i in range(1024)]

    def invoke_llm(self, prompt: str) -> str:
        self.log_call("invoke_llm", {"prompt_len": len(prompt)})
        import random
        return random.choice(self.responses)


class KendraAdapter(AWSAdapter):
    """Simulates Amazon Kendra search"""

    def __init__(self):
        super().__init__("Kendra")
        self.documents = [
            {"id": "doc1", "title": "AWS Kendra Documentation",
             "content": "Kendra is an intelligent search service", "score": 0.95},
            {"id": "doc2", "title": "Search Best Practices",
             "content": "Use natural language queries for better results", "score": 0.87},
            {"id": "doc3", "title": "Indexing Documents",
             "content": "Documents are automatically indexed", "score": 0.78},
        ]

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        self.log_call("search", {"query": query, "top_k": top_k})
        return [Document(i, doc["title"], doc["content"], doc["id"], doc["score"])
                for i, doc in enumerate(self.documents[:top_k])]


class OpenSearchAdapter(AWSAdapter):
    """Simulates Amazon OpenSearch"""

    def __init__(self):
        super().__init__("OpenSearch")
        self.vectors = {
            "vec1": Document(1, "Vector Database", "Stores embeddings", "vec1", 0.92),
            "vec2": Document(2, "Similarity Search", "Find similar docs", "vec2", 0.85),
        }

    def search_by_vector(self, embedding: List[float], top_k: int = 3) -> List[Document]:
        self.log_call("search_by_vector", {"embedding_dim": len(embedding)})
        return list(self.vectors.values())[:top_k]


class DynamoDBAdapter(AWSAdapter):
    """Simulates DynamoDB caching"""

    def __init__(self):
        super().__init__("DynamoDB")
        self.cache = {"query_1": {"query": "What is AWS?", "cached_at": datetime.now().isoformat()}}

    def put_item(self, key: str, data: Dict) -> Dict:
        self.log_call("put_item", {"key": key})
        self.cache[key] = data
        return {"status": "success"}

    def get_item(self, key: str) -> Dict:
        self.log_call("get_item", {"key": key})
        return self.cache.get(key, {})


class CloudWatchAdapter(AWSAdapter):
    """Simulates CloudWatch monitoring"""

    def __init__(self):
        super().__init__("CloudWatch")
        self.metrics = []

    def put_metric(self, metric_name: str, value: float) -> Dict:
        self.log_call("put_metric_data", {"metric_name": metric_name})
        self.metrics.append({"metric": metric_name, "value": value, "timestamp": datetime.now().isoformat()})
        return {"status": "success"}

    def get_metrics(self) -> List[Dict]:
        return self.metrics[-10:]


# ============================================================================
# LAYER 3: BUSINESS LOGIC - RAG PIPELINE
# ============================================================================

class QueryProcessor:
    def process(self, query: str) -> Dict:
        return {
            'original': query,
            'cleaned': query.lower().strip(),
            'tokens': query.lower().split(),
            'processing_step': 'Query Preprocessing Complete'
        }


class DocumentRetrieval:
    def __init__(self, kendra: KendraAdapter, opensearch: OpenSearchAdapter):
        self.kendra = kendra
        self.opensearch = opensearch

    def retrieve(self, query: str, embeddings: List[float]) -> List[Document]:
        kendra_results = self.kendra.search(query, top_k=3)
        vector_results = self.opensearch.search_by_vector(embeddings, top_k=3)
        combined = kendra_results + vector_results
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:5]


class ContextAugmentation:
    def augment(self, query: str, documents: List[Document]) -> str:
        context = f"User Query: {query}\n\nRetrieved Context:\n"
        for i, doc in enumerate(documents, 1):
            context += f"{i}. {doc.title} (Score: {doc.score:.2f})\n   {doc.content}\n\n"
        return context


class ResponseGenerator:
    def __init__(self, llm: SageMakerAdapter):
        self.llm = llm

    def generate(self, context: str) -> str:
        return self.llm.invoke_llm(context)


class RAGPipeline:
    def __init__(self, kendra: KendraAdapter, opensearch: OpenSearchAdapter, sagemaker: SageMakerAdapter):
        self.query_processor = QueryProcessor()
        self.retrieval = DocumentRetrieval(kendra, opensearch)
        self.augmentation = ContextAugmentation()
        self.generator = ResponseGenerator(sagemaker)
        self.processing_log = []

    def process(self, query: str) -> Dict:
        start_time = time.time()

        processed = self.query_processor.process(query)
        self.processing_log.append("✓ Step 1: Query Processing Complete")

        embeddings = [0.1] * 1024
        self.processing_log.append("✓ Step 2: Embeddings Generated")

        documents = self.retrieval.retrieve(query, embeddings)
        self.processing_log.append(f"✓ Step 3: Retrieved {len(documents)} Documents")

        context = self.augmentation.augment(query, documents)
        self.processing_log.append("✓ Step 4: Context Augmentation Complete")

        response = self.generator.generate(context)
        self.processing_log.append("✓ Step 5: Response Generated")

        processing_time = time.time() - start_time

        return {
            'query': query,
            'response': response,
            'sources': [{'title': doc.title, 'score': doc.score} for doc in documents],
            'processing_time': processing_time,
            'processing_steps': self.processing_log.copy()
        }


# ============================================================================
# LAYER 1: PRESENTATION LAYER - TKINTER GUI
# ============================================================================

class GenAIArchitectureDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("GenAI RAG Architecture Demo - Classroom Training")
        self.root.geometry("1400x900")

        # AWS Adapters (Layer 4)
        self.sagemaker = SageMakerAdapter()
        self.kendra = KendraAdapter()
        self.opensearch = OpenSearchAdapter()
        self.dynamodb = DynamoDBAdapter()
        self.cloudwatch = CloudWatchAdapter()

        # RAG Pipeline (Layer 3)
        self.rag_pipeline = RAGPipeline(self.kendra, self.opensearch, self.sagemaker)

        self.sample_queries = [
            "What is AWS SageMaker?",
            "How does the RAG pipeline work?",
            "Explain Amazon Kendra search service",
            "What are embeddings in machine learning?",
        ]

        self.setup_ui()

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_query_tab(notebook)
        self.create_architecture_tab(notebook)
        self.create_aws_services_tab(notebook)
        self.create_database_tab(notebook)
        self.create_pipeline_tab(notebook)
        self.create_api_tab(notebook)
        self.create_monitoring_tab(notebook)

    def create_query_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🔍 Query Interface (Layer 1)")

        header = ttk.Label(frame, text="Layer 1: Presentation Layer (Tkinter GUI)",
                           font=("Arial", 12, "bold"))
        header.pack(padx=10, pady=10)

        input_frame = ttk.LabelFrame(frame, text="User Input", padding=10)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Label(input_frame, text="Enter Query:").pack(anchor=tk.W)
        self.query_input = ttk.Combobox(input_frame, values=self.sample_queries, width=80)
        self.query_input.pack(pady=5, fill=tk.X)
        self.query_input.set(self.sample_queries[0])

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Submit Query", command=self.execute_query).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        result_frame = ttk.LabelFrame(frame, text="AI Response", padding=10)
        result_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=20, width=100)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def create_architecture_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🏗️ Architecture Layers")

        title = ttk.Label(frame, text="11 Layered Architecture", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        text = scrolledtext.ScrolledText(frame, height=30, width=140)
        text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        info = """
LAYER 1: PRESENTATION LAYER (Frontend - React/Tkinter)
  • React Components, Redux State, Axios API Client, Tailwind CSS

LAYER 2: API LAYER (FastAPI - Backend)
  • REST Endpoints, JWT Authentication, Pydantic Schemas, Error Handling

LAYER 3: BUSINESS LOGIC (RAG Pipeline)
  • QueryProcessor, DocumentRetrieval, ContextAugmentation, ResponseGenerator

LAYER 4: DATA ACCESS (AWS Adapters)
  • SageMaker, Kendra, OpenSearch, S3, DynamoDB, CloudWatch

LAYER 5: DATABASE (PostgreSQL/RDS)
  • User, Conversation, Query, Document Models

LAYER 6: INFRASTRUCTURE (Terraform)
  • VPC, Lambda, ECS, RDS, S3, CloudFront, SageMaker, Kendra, OpenSearch

LAYER 7: TESTING (Pytest/Jest)
  • Unit Tests, Integration Tests, E2E Tests

LAYER 8: CI/CD (GitHub Actions)
  • Continuous Integration, Deployment, Security Scans

LAYER 9: DEPLOYMENT (Docker/Kubernetes)
  • Docker Images, Kubernetes Manifests, Helm Charts

LAYER 10: CONFIGURATION (.env/Secrets)
  • Environment Variables, AWS Secrets Manager, Config Files

LAYER 11: MONITORING (CloudWatch/Prometheus)
  • Metrics, Logs, Alarms, Dashboards
        """

        text.insert(tk.END, info)
        text.config(state=tk.DISABLED)

    def create_aws_services_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="☁️ AWS Services (Layer 4)")

        title = ttk.Label(frame, text="Data Access Layer - AWS Adapters", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(btn_frame, text="SageMaker", command=self.show_sagemaker).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Kendra", command=self.show_kendra).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="OpenSearch", command=self.show_opensearch).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="DynamoDB", command=self.show_dynamodb).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="CloudWatch", command=self.show_cloudwatch).pack(side=tk.LEFT, padx=5)

        self.aws_text = scrolledtext.ScrolledText(frame, height=25, width=140)
        self.aws_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.show_sagemaker()

    def show_sagemaker(self):
        self.aws_text.config(state=tk.NORMAL)
        self.aws_text.delete(1.0, tk.END)
        info = f"""
AWS SAGEMAKER - LLM & EMBEDDINGS
Location: app/adapters/sagemaker_adapter.py

• Generate text embeddings (1024 dimensions)
• Invoke LLM endpoints for response generation
• API Calls: {self.sagemaker.call_count}

Methods:
  - embed_text(text) → [float]
  - invoke_llm(prompt) → str

Configuration:
  SAGEMAKER_LLM_ENDPOINT: llama2-endpoint
  SAGEMAKER_EMBEDDING_ENDPOINT: embeddings-endpoint
  REGION: us-east-1

Performance:
  Response Time: ~200ms per request
  Cost: $0.00256 per 1K input tokens
        """
        self.aws_text.insert(tk.END, info)
        self.aws_text.config(state=tk.DISABLED)

    def show_kendra(self):
        self.aws_text.config(state=tk.NORMAL)
        self.aws_text.delete(1.0, tk.END)
        docs = "\n  ".join([f"{d['title']} (Score: {d['score']})" for d in self.kendra.documents])
        info = f"""
AMAZON KENDRA - INTELLIGENT SEARCH
Location: app/adapters/kendra_adapter.py

• Index documents for intelligent search
• Perform natural language retrieval
• API Calls: {self.kendra.call_count}

Indexed Documents:
  {docs}

Methods:
  - search(query, top_k) → [Document]
  - index_document(doc_id, content) → bool

Configuration:
  KENDRA_INDEX_ID: genai-prod-index
  REGION: us-east-1

Performance:
  Query Latency: ~100-300ms
  Cost: $0.35 per 1000 queries
        """
        self.aws_text.insert(tk.END, info)
        self.aws_text.config(state=tk.DISABLED)

    def show_opensearch(self):
        self.aws_text.config(state=tk.NORMAL)
        self.aws_text.delete(1.0, tk.END)
        info = f"""
AMAZON OPENSEARCH - VECTOR DATABASE
Location: app/adapters/opensearch_adapter.py

• Store high-dimensional vectors
• Perform vector similarity search
• API Calls: {self.opensearch.call_count}

Vector Storage:
  Dimension: 1024
  Indexed Vectors: {len(self.opensearch.vectors)}
  Algorithm: HNSW (Hierarchical Navigable Small World)

Methods:
  - search_by_vector(embedding, top_k) → [Document]
  - index_document(doc_id, document) → bool

Performance:
  Search Latency: ~50-100ms for 1M vectors
  Indexing: ~10K vectors/second

Features:
  ✓ Semantic understanding
  ✓ Multi-language support
  ✓ Real-time indexing
        """
        self.aws_text.insert(tk.END, info)
        self.aws_text.config(state=tk.DISABLED)

    def show_dynamodb(self):
        self.aws_text.config(state=tk.NORMAL)
        self.aws_text.delete(1.0, tk.END)
        cached = "\n  ".join([f"{k}: {v.get('query', 'N/A')}" for k, v in list(self.dynamodb.cache.items())[:5]])
        info = f"""
AMAZON DYNAMODB - CACHING & METADATA
Location: app/adapters/dynamodb_adapter.py

• Cache query results
• Store conversation metadata
• API Calls: {self.dynamodb.call_count}

Cached Items:
  {cached}

Methods:
  - put_item(key, data) → Dict
  - get_item(key) → Dict
  - query_items(params) → [Dict]

Configuration:
  TABLE_NAME: genai-cache
  BILLING_MODE: PAY_PER_REQUEST
  TTL: 3600 seconds

Performance:
  Read Latency: < 1ms
  Write Latency: < 5ms
  Cost: $1.25/million writes, $0.25/million reads
        """
        self.aws_text.insert(tk.END, info)
        self.aws_text.config(state=tk.DISABLED)

    def show_cloudwatch(self):
        self.aws_text.config(state=tk.NORMAL)
        self.aws_text.delete(1.0, tk.END)
        metrics = "\n  ".join([f"{m['metric']}: {m['value']}" for m in self.cloudwatch.get_metrics()[-5:]])
        info = f"""
AMAZON CLOUDWATCH - MONITORING
Location: app/adapters/cloudwatch_adapter.py

• Collect application metrics
• Aggregate logs
• Create alarms
• API Calls: {self.cloudwatch.call_count}

Recent Metrics:
  {metrics if metrics else 'No metrics yet'}

Key Metrics:
  • Query Processing Time
  • Token Usage
  • Cache Hit Rate
  • Error Rate
  • API Latency

Performance:
  Logs Ingestion: $0.50/GB
  Storage: $0.03/GB/month
  Custom Metrics: $0.30/metric/month
        """
        self.aws_text.insert(tk.END, info)
        self.aws_text.config(state=tk.DISABLED)

    def create_database_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🗄️ Database Models (Layer 5)")

        title = ttk.Label(frame, text="Database Models - PostgreSQL", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        text = scrolledtext.ScrolledText(frame, height=30, width=140)
        text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        schema = """
DATABASE SCHEMA - POSTGRESQL/RDS

USERS TABLE:
  • id (PRIMARY KEY)
  • username (UNIQUE, NOT NULL)
  • email (UNIQUE, NOT NULL)
  • password_hash (NOT NULL)
  • created_at (DEFAULT NOW())
  • updated_at (DEFAULT NOW())
  • is_active (DEFAULT true)

CONVERSATIONS TABLE:
  • id (PRIMARY KEY)
  • user_id (FOREIGN KEY → users)
  • title (NOT NULL)
  • created_at (DEFAULT NOW())
  • updated_at (DEFAULT NOW())
  • is_archived (DEFAULT false)

QUERIES TABLE:
  • id (PRIMARY KEY)
  • conversation_id (FOREIGN KEY → conversations)
  • query_text (NOT NULL)
  • response_text (NOT NULL)
  • processing_time (NOT NULL)
  • token_usage (NOT NULL)
  • sources (JSONB NOT NULL)
  • created_at (DEFAULT NOW())
  • user_rating (DEFAULT NULL)
  • feedback_text (DEFAULT NULL)

DOCUMENTS TABLE:
  • id (PRIMARY KEY)
  • user_id (FOREIGN KEY → users)
  • title (NOT NULL)
  • content (NOT NULL)
  • s3_path (NOT NULL)
  • embedding_id (NOT NULL)
  • document_type (NOT NULL)
  • created_at (DEFAULT NOW())
  • indexed (DEFAULT false)

INDEXES:
  • username_idx (unique)
  • email_idx (unique)
  • user_id_created_at_idx (composite)
  • embedding_id_idx (unique)
  • sources_idx (GIN - JSON indexing)

RELATIONSHIPS:
  USERS (1) ─── (∞) CONVERSATIONS
  USERS (1) ─── (∞) DOCUMENTS
  CONVERSATIONS (1) ─── (∞) QUERIES

Configuration:
  Engine: PostgreSQL 15
  Instance: db.t3.medium
  Storage: 100GB (auto-scaling to 1TB)
  Backups: Automated daily
  Multi-AZ: Enabled
  Encryption: At rest (AWS managed keys)
        """

        text.insert(tk.END, schema)
        text.config(state=tk.DISABLED)

    def create_pipeline_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🔄 RAG Pipeline (Layer 3)")

        title = ttk.Label(frame, text="RAG Pipeline - Business Logic", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(padx=10, pady=5, fill=tk.X)
        ttk.Button(btn_frame, text="Run Pipeline Demo", command=self.run_pipeline_demo).pack(side=tk.LEFT, padx=5)

        self.pipeline_text = scrolledtext.ScrolledText(frame, height=25, width=140)
        self.pipeline_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.show_pipeline_architecture()

    def show_pipeline_architecture(self):
        self.pipeline_text.config(state=tk.NORMAL)
        self.pipeline_text.delete(1.0, tk.END)

        arch = """
RAG PIPELINE - BUSINESS LOGIC LAYER

COMPONENT 1: QueryProcessor
  • Text normalization (lowercase, strip whitespace)
  • Tokenization (split into words)
  • Named Entity Recognition (extract entities)
  • Input: Raw user query
  • Output: Processed query dict

COMPONENT 2: DocumentRetrieval
  • Multi-source search (Kendra + OpenSearch)
  • Result combining and deduplication
  • Ranking by relevance score
  • Input: Processed query + embeddings
  • Output: List of top-K documents

COMPONENT 3: ContextAugmentation
  • Extract relevant passages from documents
  • Rerank documents by relevance
  • Build augmented prompt with context
  • Include conversation history
  • Input: Query + Documents + History
  • Output: Augmented prompt

COMPONENT 4: ResponseGenerator
  • Invoke SageMaker LLM endpoint
  • Handle streaming responses
  • Implement retry logic
  • Track token usage
  • Input: Augmented prompt
  • Output: AI-generated response

COMPONENT 5: RAGPipeline (Orchestrator)
  • Orchestrate all services in sequence
  • Track processing time for each step
  • Handle errors at each stage
  • Log pipeline execution

EXECUTION FLOW:
  1. User Query (Layer 1)
  2. API Endpoint (Layer 2)
  3. QueryProcessor: Clean & tokenize
  4. Generate Embeddings (SageMaker)
  5. DocumentRetrieval: Kendra + OpenSearch
  6. ContextAugmentation: Build prompt
  7. ResponseGenerator: Invoke LLM
  8. Post-Processing: Format & cache
  9. Return Response to User

PERFORMANCE BREAKDOWN:
  • Query Processing: 50ms (4%)
  • Embeddings: 200ms (16%)
  • Retrieval: 300ms (24%)
  • Context Augmentation: 100ms (8%)
  • LLM Generation: 595ms (48%)
  • Total: ~1,245ms

CONFIGURATION:
  max_tokens: 500
  temperature: 0.7 (creativity)
  top_p: 0.9 (diversity)
  repetition_penalty: 1.2
        """

        self.pipeline_text.insert(tk.END, arch)
        self.pipeline_text.config(state=tk.DISABLED)

    def run_pipeline_demo(self):
        query = self.query_input.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query")
            return
        self.execute_query()

    def create_api_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🔌 API Layer (Layer 2)")

        title = ttk.Label(frame, text="REST API Endpoints - FastAPI", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        text = scrolledtext.ScrolledText(frame, height=28, width=140)
        text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        api_info = """
API LAYER - FASTAPI ENDPOINTS

ENDPOINT 1: POST /api/v1/query
  Request: {query, conversation_id, session_id, top_k, stream}
  Response: {answer, sources, processing_time, token_usage}
  Auth: Bearer <JWT_TOKEN>
  Rate Limit: 100 requests/minute

ENDPOINT 2: POST /api/v1/query/stream
  Description: Stream LLM response chunks
  Response Type: Server-Sent Events (SSE)
  Chunk Size: 50 tokens
  Timeout: 60 seconds

ENDPOINT 3: GET /api/v1/query/history
  Params: conversation_id, limit, offset
  Response: {messages, total, limit, offset}
  Auth: Bearer <JWT_TOKEN>
  Cache: 300 seconds

ENDPOINT 4: POST /api/v1/auth/login
  Request: {username, password}
  Response: {access_token, refresh_token, expires_in}
  Rate Limit: 10 attempts/5 minutes
  Security: Bcrypt hashing, JWT tokens

ENDPOINT 5: GET /api/v1/health
  Response: {status, timestamp, services}
  Use: Kubernetes liveness/readiness probes
  Timeout: 5 seconds

MIDDLEWARE LAYERS:

1. CORS MIDDLEWARE:
   ✓ Allowed Origins: frontend, CDN
   ✓ Methods: GET, POST, PUT, DELETE
   ✓ Headers: Authorization, Content-Type

2. AUTHENTICATION MIDDLEWARE:
   ✓ Extract JWT from Authorization header
   ✓ Validate token signature
   ✓ Check expiration
   ✓ Verify permissions

3. LOGGING MIDDLEWARE:
   ✓ Log all requests (method, path, user)
   ✓ Log response time and status
   ✓ Track errors

4. RATE LIMITING:
   ✓ 100 requests/minute per user
   ✓ 1000 requests/minute per IP
   ✓ Returns 429 Too Many Requests

5. ERROR HANDLING:
   ✓ Global exception handler
   ✓ Convert exceptions to HTTP responses
   ✓ Log with full context
   ✓ Return appropriate status codes

REQUEST/RESPONSE FLOW:
  Request → CORS Check → Auth → Logging → Rate Limit
  → Route Handler → Business Logic → Response → Logging
        """

        text.insert(tk.END, api_info)
        text.config(state=tk.DISABLED)

    def create_monitoring_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📊 Monitoring (Layer 11)")

        title = ttk.Label(frame, text="Monitoring & Observability", font=("Arial", 12, "bold"))
        title.pack(padx=10, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(btn_frame, text="Show Metrics", command=self.show_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show Alerts", command=self.show_alerts).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show Logs", command=self.show_logs).pack(side=tk.LEFT, padx=5)

        self.monitoring_text = scrolledtext.ScrolledText(frame, height=25, width=140)
        self.monitoring_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.show_metrics()

    def show_metrics(self):
        self.monitoring_text.config(state=tk.NORMAL)
        self.monitoring_text.delete(1.0, tk.END)

        metrics = f"""
MONITORING & OBSERVABILITY - CLOUDWATCH & PROMETHEUS

KEY METRICS DASHBOARD:

METRIC 1: Query Processing Time
  • Average: 1,245 ms
  • P95: 2,800 ms
  • Target SLA: < 2,000 ms
  • Alert Threshold: > 3,000 ms

METRIC 2: LLM Token Usage
  • Average Input: 1,024 tokens
  • Average Output: 256 tokens
  • Daily Total: ~1.28M tokens
  • Estimated Daily Cost: $6.40

METRIC 3: Cache Hit Rate
  • Current: 67%
  • Target: 75%
  • Cost Savings: $1,350/month

METRIC 4: Error Rate
  • Overall: 0.5%
  • Target: < 1%
  • SageMaker Timeout: 0.15%
  • Kendra Failure: 0.12%

METRIC 5: API Latency by Endpoint
  /api/v1/query: P50 1,100ms, P95 2,800ms
  /api/v1/health: P50 50ms, P95 100ms
  /api/v1/auth/login: P50 500ms, P95 1,200ms

CLOUDWATCH DASHBOARD SECTIONS:

1. OVERVIEW:
   • Total Requests (24h): 240,000
   • Success Rate: 99.5%
   • Avg Processing Time: 1,245ms
   • Daily Cost: $120

2. PERFORMANCE:
   • Query Processing Time (graph)
   • Cache Hit Rate (gauge)
   • API Latency (bar chart)
   • Token Usage (area chart)

3. ERROR TRACKING:
   • Error Rate (time series)
   • Errors by Type (pie chart)
   • Recent Errors (table)

4. RESOURCE MONITORING:
   • Lambda Concurrent Executions
   • RDS CPU Utilization
   • ElastiCache Network I/O
   • CloudFront Cache Hit Ratio

CUSTOM METRICS: {len(self.cloudwatch.get_metrics())} recorded

Recent Metrics:
"""
        for metric in self.cloudwatch.get_metrics()[-5:]:
            metrics += f"  • {metric['metric']}: {metric['value']}\n"

        metrics += """

ALARMS:
  1. HighQueryLatency (> 3,000ms)
  2. HighErrorRate (> 2%)
  3. LowCacheHitRate (< 50%)
  4. ServiceUnavailable (health check fails)
        """

        self.monitoring_text.insert(tk.END, metrics)
        self.monitoring_text.config(state=tk.DISABLED)

    def show_alerts(self):
        self.monitoring_text.config(state=tk.NORMAL)
        self.monitoring_text.delete(1.0, tk.END)

        alerts = """
CONFIGURED ALARMS & ALERTS

CRITICAL (Page On-Call):
  🔴 QueryProcessingTimeout: > 5 seconds
  🔴 ServiceDown: Health check fails 3 times
  🔴 DatabaseConnectionFailure: Connection pool exhausted
  🔴 HighErrorRate: > 5% for 3 minutes

HIGH PRIORITY (Email + Slack):
  🟠 SageMakerEndpointDown: Unavailable
  🟠 KendraIndexingFailure: Index update fails
  🟠 OpenSearchClusterYellow: Status = yellow
  🟠 RDSCPUUtilization: > 80% for 10 minutes

MEDIUM PRIORITY (Slack):
  🟡 CacheHitRateLow: < 50% for 30 minutes
  🟡 APILatencyHigh: P95 > 2.5 seconds
  🟡 TokenUsageHigh: > 2M daily
  🟡 S3StorageLarge: > 500GB

LOW PRIORITY (Digest):
  🔵 HighMemoryUsage: > 70% for 30 minutes
  🔵 DiskSpaceWarning: > 80% utilized

NOTIFICATION CHANNELS:
  • SMS: Critical alerts (page on-call)
  • Slack: All alerts (#ops, #platform)
  • Email: Medium/Low priority
  • PagerDuty: Incident management

HISTORICAL (Last 7 Days):
  Total Alerts: 24
  Critical (paged): 2 ✓ Resolved
  MTTR: 12 minutes (critical)
  False Alarm Rate: 8.3%
        """

        self.monitoring_text.insert(tk.END, alerts)
        self.monitoring_text.config(state=tk.DISABLED)

    def show_logs(self):
        self.monitoring_text.config(state=tk.NORMAL)
        self.monitoring_text.delete(1.0, tk.END)

        logs = """
APPLICATION LOGS & TRACES

CloudWatch Log Groups:
  /aws/genai-rag-app/backend
    ├─ api-server
    ├─ lambda-handler
    └─ background-workers

  /aws/genai-rag-app/database
    ├─ queries
    └─ slow-queries (> 1s)

  /aws/genai-rag-app/ai-services
    ├─ sagemaker-invocations
    ├─ kendra-indexing
    └─ opensearch-operations

LOG ENTRIES (Recent):

[12:34:56.123] INFO [query:001] User: user_123
  Query: "What is AWS SageMaker?"
  Status: PROCESSING

[12:34:56.850] DEBUG [query_processor:001]
  Tokens: ['what', 'is', 'aws', 'sagemaker']

[12:34:57.050] DEBUG [embeddings:001]
  Generated embedding (1024 dims)

[12:34:57.350] DEBUG [retrieval:001]
  Kendra: 3 documents (0.95, 0.87, 0.78)
  OpenSearch: 2 documents (0.92, 0.85)

[12:34:57.450] DEBUG [context:001]
  Context length: 1024 tokens
  Documents: 5

[12:34:58.045] DEBUG [llm:001]
  Latency: 595ms
  Tokens: 1280

[12:34:58.150] INFO [query:001]
  Status: COMPLETED
  Processing time: 1025ms

DISTRIBUTED TRACING (X-Ray):

Trace ID: 1-5e8c8c8c-a1b2c3d4e5f6g7h8i9j0k1l2

Service Map:
  API Gateway (0.5ms) →
  Lambda (1,025ms) →
    ├─ QueryProcessor (50ms)
    ├─ SageMaker (200ms + 595ms)
    ├─ Kendra (150ms)
    └─ OpenSearch (150ms)
  →
  DynamoDB (5ms) →
  Response

Performance Analysis:
  Total Duration: 1,025ms
  Critical Path: LLM Generation (595ms)
  Slowest: SageMaker endpoint
  Optimization: Use faster model, caching

LOG FILTERING:

Find queries with high latency:
  filter @duration > 2000
  stats count() as HighLatencyQueries

Find errors by service:
  filter @message like /ERROR/
  stats count() by @service

Most frequently asked:
  stats count() as frequency by query
  sort frequency desc
        """

        self.monitoring_text.insert(tk.END, logs)
        self.monitoring_text.config(state=tk.DISABLED)

    def execute_query(self):
        query = self.query_input.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query")
            return

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "⏳ Processing query through RAG pipeline...\n\n")
        self.result_text.config(state=tk.DISABLED)
        self.root.update()

        thread = threading.Thread(target=self._process_query, args=(query,))
        thread.daemon = True
        thread.start()

    def _process_query(self, query):
        try:
            result = self.rag_pipeline.process(query)

            # Log metrics
            self.cloudwatch.put_metric("QueryProcessingTime", result['processing_time'] * 1000)
            self.dynamodb.put_item(f"query_{int(time.time())}", {"query": query, "response": result['response']})

            output = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              QUERY RESULT FROM RAG PIPELINE                                                   ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

USER QUERY:
  {query}

AI RESPONSE:
  {result['response']}

PROCESSING STEPS:
"""
            for step in result['processing_steps']:
                output += f"  {step}\n"

            output += f"""
SOURCES RETRIEVED:
"""
            for i, source in enumerate(result['sources'], 1):
                output += f"  {i}. {source['title']} (Confidence: {source['score']:.2%})\n"

            output += f"""
PERFORMANCE METRICS:
  • Processing Time: {result['processing_time']:.2f} seconds
  • Pipeline Efficiency: {'Excellent' if result['processing_time'] < 1.5 else 'Good'}

LAYER EXECUTION SUMMARY:
  ✓ Layer 1 (Presentation): Query received
  ✓ Layer 2 (API): Request routed
  ✓ Layer 3 (Business Logic): RAG pipeline executed
  ✓ Layer 4 (Data Access): AWS adapters called
  ✓ Layer 5 (Database): Results cached
  ✓ Layer 11 (Monitoring): Metrics recorded
"""

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, output)
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
            self.result_text.config(state=tk.DISABLED)

    def clear_results(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = GenAIArchitectureDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
