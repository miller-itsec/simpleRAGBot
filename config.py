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

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
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
DOWNLOAD_CUSTOM_URLS = True
CUSTOM_URLS = ["https://www.opswat.com/products/metadefender/sandbox",
                "https://www.opswat.com/blog/getting-started-with-opswat-filescan-sandbox-sandboxing-made-easy",
                "https://www.opswat.com/blog/opswat-filescan-sandbox-v1-9-1-new-updates-and-releases",
                "https://www.opswat.com/blog/metadefender-sandbox-v1-9-2-sharpen-your-threat-response-with-the-latest-enhanced-features",
                "https://www.opswat.com/blog/introducing-opswat-metadefender-sandbox-v1-9-3",
                "https://www.filescan.io/api/docs",
                "https://www.filescan.io/help/faq"]
PERFORM_CUSTOM_GOOGLE_QUERIES = True
CUSTOM_GOOGLE_QUERIES = ["MetaDefender Sandbox"]
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
# Delete small files in the CUSTOM_* directories
CLEAN_SMALL_CUSTOM_DOCUMENTS = False
CLEAN_SMALL_CUSTOM_DOCUMENTS_THRESHOLD_BYTES = 1024
SUMMARIZE_LARGE_DOCUMENTS = True
SUMMARIZE_LARGE_DOCUMENTS_THRESHOLD = 10 * 1024  # Summarize any document above this threshold
SUMMARIZE_MODEL = "facebook/bart-large-cnn"
SKIP_URL_SUMMARY = True
CUSTOM_URL_SUMMARY_STORAGE = os.path.join(current_directory, "custom_urls")
SUMMARIZE_USE_CACHE = True
COMPANY_NAME = "OPSWAT"
PRODUCT_NAMES = {
    "MetaDefender Sandbox": ["Filescan Sandbox", "OPSWAT Sandbox"],
    # Add other products and their synonyms here
    # "Another Product": ["Synonym1", "Synonym2"]
}
# Define prompt template (make sure it matches the model's expected input format and you use {product_instructions}, {context} and {question})
PROMPT_TEMPLATE = """
### [INST] Product description:
{product_instructions}

CONTEXT ONLY for questions relating to the product:
{context}

QUESTION:
{question}

In your response, provide clear, concise, and relevant information. Summarize the key findings in a list format, with a proper intro and outro.

If the question is unrelated to the product, ask for a product-related question by listing the product names without mentioning the context's relevance or lack thereof.

[/INST]
"""
PROMPT_TOKEN_START = "[INST]"
PROMPT_TOKEN_END = "[/INST]"
RESPONSE_STRIP_CHUNKS = [
    "\n### \n",  # Original text to strip, add more items as needed
]

# Set up directories, models and search parameters (advanced)
MODEL_DIR = "E:\\LLM\\models"
MODEL_PATH = os.path.join(MODEL_DIR, "mistral-7b-instruct-v0.2-bnb-4bit")  # https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit
MODEL_TEMPERATURE = 0.25  # Lower temperature reduces randomness
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Configure the retriever with Maximal Marginal Relevance (MMR) for a good balance of relevance and diversity.
# Here, 'k' represents the number of documents to retrieve, and 'lambda_mult' adjusts the balance between relevance and diversity.
# Setting e.g. 'lambda_mult' to 0.5 gives equal importance to both, with a lower value corresponding to maximum diversity.
RETRIEVER_FAISS_SEARCH_TYPE = "mmr"
RETRIEVER_FAISS_SEARCH_ARGS = {'k': 50, 'lambda_mult': 0.65}
MAX_OUTPUT_LENGTH = 512
VECTOR_USE_CACHE = True
VECTOR_CACHE_TTL = 60  # Enforce vector db rebuild, if too old (in minutes)
VECTOR_STORAGE_FOLDER = "db_vectors"
VECTOR_INDEX_FILE = os.path.join(VECTOR_STORAGE_FOLDER, "index.faiss")
BM25_CACHE_FILE = "bm25_index.pkl"
ENSEMBLE_RETRIEVER_WEIGHTS = [0.25, 0.75] # [BM25, faiss]
RELEVANCE_SCORE_THRESHOLD = 0.95  # Ignore context documents if they do not meet at least this threshold
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

GENERATE_TRAINING_DATA = True
TRAINING_DATA_PROMPT = """
Given the detailed content provided below, identify three key aspects or sections and provide a detailed summary for each. These summaries should include specific facts, data points, and definitions used in the text. For each detailed summary, imagine you are designing a training exercise for an advanced AI. Treat each detailed summary as the desired outcome (output) of separate exercises. For each detailed summary, generate:
1. A question (input) that the AI should answer to arrive at this output.
2. A specific instruction that the AI should follow to ensure it addresses the question correctly and effectively.

Content: {content}

Please provide your output in the following JSON format for each section, and ensure the summary includes exact content from the original text:

```
[
{{
"instruction": "<instruction_1>",
"input": "<input_1>",
"output": "<detailed_summary_1>"
}},
{{
"instruction": "<instruction_2>",
"input": "<input_2>",
"output": "<detailed_summary_2>"
}},
{{
"instruction": "<instruction_3>",
"input": "<input_3>",
"output": "<detailed_summary_3>"
}}
]
```
The summaries must include specific details such as data values, factual content, and precise definitions from the text. Please replace `<instruction_n>`, `<input_n>`, and `<detailed_summary_n>` with text that reflects these requirements.
"""
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-16k')
API_REQUEST_TIMEOUT = int(os.getenv('API_REQUEST_TIMEOUT', 20))  # in seconds
