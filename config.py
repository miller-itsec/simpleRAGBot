# config.py
"""
simpleRAGBot Configuration

Part of the simpleRAGBot project, this configuration script sets up the environment for an advanced Retrieval-Augmented Generation (RAG) Bot. Leveraging cutting-edge NLP models, the bot is designed to generate contextually aware responses from a diverse range of sources, including markdown documents, PDFs, and web content.

Configuration Highlights:
- Uses the Mistral-7B-Instruct-v0.2 model for nuanced response generation.
- Employs efficient text embeddings with sentence-transformers.
- Handles various document types to form a comprehensive knowledge base.
- Adjustable document chunking for effective context capture.
- Flask-based API server for external prompt handling, customizable for different operational needs.
- Implements refined web scraping techniques with adaptable user-agent settings.

Usage Instructions:
- Adjust configuration parameters as needed.
- Run the script to initialize the RAG Bot and optionally activate the web server.
- Interact with the bot through the CLI or API endpoints as per your setup.

(c) 2024 Jan Miller (@miller_itsec) for OPSWAT, Inc. All rights reserved.
"""
import os
current_directory = os.getcwd()

# Flask server configuration
ENABLE_WEBSERVER = True  # Set to False if you don't want to run the web server
WEBSERVER_PORT = 5000  # The port on which the Flask server will run
WEBSERVER_RATE_LIMIT = "1 per minute"
WEBSERVER_MAX_WORKERS = 2
TASK_PROCESSING_TIMEOUT = 180

# User-input on the console
ENABLE_COMMANDLINE = True

# Set up custom data sources
CUSTOM_PDFS = os.path.join(current_directory, "custom_documents")
CUSTOM_MARKDOWN = os.path.join(current_directory, "custom_documents")
CUSTOM_URLS = ["https://www.opswat.com/products/metadefender/sandbox",
                "https://www.opswat.com/blog/getting-started-with-opswat-filescan-sandbox-sandboxing-made-easy",
                "https://www.opswat.com/blog/opswat-filescan-sandbox-v1-9-1-new-updates-and-releases",
                "https://www.opswat.com/blog/metadefender-sandbox-v1-9-2-sharpen-your-threat-response-with-the-latest-enhanced-features",
                "https://www.opswat.com/blog/introducing-opswat-metadefender-sandbox-v1-9-3",
                "https://www.filescan.io/api/docs",
                "https://www.filescan.io/help/faq"]
CUSTOM_GOOGLE_QUERIES = ["MetaDefender Sandbox", "OPSWAT Products", "OPSWAT Blog", "OPSWAT"]
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
PRODUCT_NAMES = {
    "MetaDefender Sandbox": ["Filescan Sandbox", "OPSWAT Sandbox"],
    # Add other products and their synonyms here
    # "Another Product": ["Synonym1", "Synonym2"]
}
# Define prompt template
PROMPT_TEMPLATE = """
### [INST] Product description: 
{product_instructions}

CONTEXT ONLY for questions relating to the product:
{context}

QUESTION:
{question}

In your response, it is okay to ignore the context if it is unrelated, as it was provided just in case the question refers to it.
If you do not include context related data, do not mention that in your response. Omit sentences like 'The given context does not 
provide any information regarding ... Therefore, the response will focus only on the question at hand.'

Please offer clear, concise, and relevant information and summarize the key findings in a list, with a proper intro and outro.

If the question is unrelated to the product, please ask for a product related question listing the product names.

[/INST]
"""

# Set up directories, models and search parameters (advanced)
MODEL_PATH = os.path.join(current_directory, "Mistral-7B-Instruct-v0.2")
MODEL_TEMPERATURE = 0.25  # Lower temperature reduces randomness
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Configure the retriever with Maximal Marginal Relevance (MMR) for a good balance of relevance and diversity.
# Here, 'k' represents the number of documents to retrieve, and 'lambda_mult' adjusts the balance between relevance and diversity.
# Setting e.g. 'lambda_mult' to 0.5 gives equal importance to both, with a lower value corresponding to maximum diversity.
RETRIEVER_FAISS_SEARCH_TYPE = "mmr"
RETRIEVER_FAISS_SEARCH_ARGS = {'k': 10, 'lambda_mult': 0.75}
MAX_OUTPUT_LENGTH = 512
VECTOR_USE_CACHE = True
VECTOR_CACHE_TTL = 60  # Enforce vector db rebuild, if too old (in minutes)
VECTOR_STORAGE_FOLDER = "db_vectors"
VECTOR_INDEX_FILE = os.path.join(VECTOR_STORAGE_FOLDER, "index.faiss")
BM25_CACHE_FILE = "bm25_index.pkl"
ENSEMBLE_RETRIEVER_WEIGHTS = [0.25, 0.75] #[BM25, faiss]
ENABLE_LANGCHAIN_CACHE = False

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.152 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
]
