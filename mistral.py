import os
import sys
import asyncio
import glob
import logging
import json
import time
import torch
import re
import random
import warnings

from tqdm import tqdm
from bs4 import BeautifulSoup

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.docstore.document import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up directories
current_directory = os.getcwd()
MODEL_PATH = os.path.join(current_directory, "Mistral-7B-Instruct-v0.2")
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTOR_USE_CACHE = False
VECTOR_STORAGE_FOLDER = "db_vectors"
VECTOR_INDEX_FILE = os.path.join(VECTOR_STORAGE_FOLDER, "index.faiss")

# Set up custom data sources
CUSTOM_PDFS = os.path.join(current_directory, "custom_documents")
CUSTOM_MARKDOWN = os.path.join(current_directory, "custom_documents")
CUSTOM_URLS = ["https://www.opswat.com/products/metadefender/sandbox",
                "https://www.opswat.com/blog/getting-started-with-opswat-filescan-sandbox-sandboxing-made-easy",
                "https://www.opswat.com/blog/opswat-filescan-sandbox-v1-9-1-new-updates-and-releases",
                "https://www.opswat.com/blog/metadefender-sandbox-v1-9-2-sharpen-your-threat-response-with-the-latest-enhanced-features",
                "https://www.opswat.com/blog/introducing-opswat-metadefender-sandbox-v1-9-3",
                "https://www.filescan.io/api/docs"]
CHUNK_SIZE = 1024  # Example: Increase chunk size
CHUNK_OVERLAP = 100  # Example: Add overlap
MAX_OUTPUT_LENGTH = 2048
# Define prompt template
prompt_template = """
### [INST] Instruction: 
Utilize your knowledge base to provide an accurate and informative response. Consider the potential context provided below to 
better understand the nuances of the question. Assume that 'Filescan Sandbox' is the same as 'MetaDefender Sandbox'.

**Context:**  
{context}

### QUESTION:
{question}

**Note:** 
In your response, ensure that you address specific points mentioned in the context and question.
 Offer clear, concise, and relevant information based on the available data.

[/INST]
 """
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

if len(CUSTOM_URLS) > 0:
    import nest_asyncio
    # For URL scraping, additional setup is needed
    # pip install playwright
    # pip install nest-asyncio
    # pip install html2text
    # playwright install
    # playwright install-deps
    nest_asyncio.apply()


def instantiate_huggingface_model(model_name, use_4bit=False, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4",
                                  use_nested_quant=False, llm_int8_threshold=400.0, device_map="auto", use_cache=True,
                                  trust_remote_code=None, pad_token=None, padding_side="left"):
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=not use_4bit,  # Load in 4-bit if use_4bit is True
            compute_dtype=bnb_4bit_compute_dtype,  # Set compute dtype for 4-bit models
            quant_type=bnb_4bit_quant_type,  # Set quantization type
            use_nested_quant=use_nested_quant,  # Enable/disable nested quantization
            llm_int8_threshold=llm_int8_threshold  # Set the int8 threshold
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=llm_int8_threshold
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        use_cache=use_cache,
        trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    else:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return model, tokenizer


def get_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (f"Trainable model parameters: {trainable_model_params}\nAll model parameters: {all_model_params}\n"
            f"Percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")


def strip_base64_images(content):
    # Regular expression pattern for base64 encoded images
    pattern = r'!\[\]\(data:image\/[a-zA-Z]*;base64,[^)]*\)'
    # Replace the base64 images with an empty string
    return re.sub(pattern, '', content)


def strip_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    # Get text
    text = soup.get_text(separator=' ')
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each and then concatenate lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


class MarkdownDirectoryLoader:
    def __init__(self, directory_path, max_file_size_kb=100):
        self.directory_path = directory_path
        self.max_file_size_kb = max_file_size_kb

    def is_file_size_within_limit(self, file_path):
        file_size_kb = os.path.getsize(file_path) / 1024
        return file_size_kb <= self.max_file_size_kb

    def load(self):
        documents = []
        markdown_files = glob.glob(f"{self.directory_path}/**/*.md", recursive=True)
        for file_path in markdown_files:
            if self.is_file_size_within_limit(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Strip out Markdown links using a regular expression
                    content = re.sub(r'\[([^]]+)]\([^)]+\)', r'\1', content)
                    content = strip_base64_images(content)
                    document = Document(page_content=content, metadata={"source": file_path})
                    documents.append(document)
        return documents


pdf_docs = []
if len(CUSTOM_PDFS) > 0:
    # Load PDFs
    logger.info("Loading custom PDF files ...")
    pdf_loader = PyPDFDirectoryLoader(CUSTOM_PDFS)
    pdf_docs = pdf_loader.load()
    logger.info(f"Number of PDF documents loaded: {len(pdf_docs)}")

markdown_docs = []
if len(CUSTOM_MARKDOWN) > 0:
    # Load Markdown files
    logger.info("Loading custom markdown files ...")
    markdown_loader = MarkdownDirectoryLoader(CUSTOM_MARKDOWN)
    markdown_docs = markdown_loader.load()
    logger.info(f"Number of Markdown documents loaded: {len(markdown_docs)}")

url_docs = []


def is_valid_content(content):
    # List of phrases indicating invalid or error content
    invalid_phrases = ["403 ERROR", "Request blocked"]
    return not any(phrase in content for phrase in invalid_phrases)


class CustomAsyncChromiumLoader(AsyncChromiumLoader):
    def __init__(self, urls, user_agent=None):
        super().__init__(urls)
        self.user_agent = user_agent

    async def ascrape_playwright(self, url: str) -> str:
        from playwright.async_api import async_playwright
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                # Create a new browser context with the custom user agent
                context = await browser.new_context(
                    user_agent=self.user_agent) if self.user_agent else await browser.new_context()
                page = await context.new_page()
                await page.goto(url)
                results = await page.content()
            except Exception as e:
                results = f"Error: {e}"
            await browser.close()
        return results


async def load_and_transform_url(loader, url, delay=0.0):
    await asyncio.sleep(delay)
    html_content = await loader.ascrape_playwright(url)
    html_content = strip_html(html_content)
    html_content = re.sub(r'[\s]+', ' ', html_content)  # Remove excessive whitespace
    logger.info(f"Extracted Text from URL:\n{html_content[:500]}...")  # Print the first 500 characters
    return Document(page_content=html_content, metadata={"source": url})


async def process_urls(urls, delay_between_requests=1):
    user_agent = random.choice(USER_AGENTS)
    loader = CustomAsyncChromiumLoader(urls, user_agent)
    html2text = Html2TextTransformer()
    tasks = []
    for url in urls:
        # Randomly vary the delay a bit to make the requests seem more 'human'
        adjusted_delay = delay_between_requests + random.uniform(-0.5, 0.5)
        tasks.append(
            load_and_transform_url(loader, url, delay=adjusted_delay))
    docs = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(urls), desc="Processing URLs"):
        doc = await future
        # Check if the content is valid
        if is_valid_content(doc.page_content):
            docs.append(doc)
        else:
            logger.info(f"Invalid content detected for URL: {doc.metadata['source']}")
    return html2text.transform_documents(docs)


if len(CUSTOM_URLS) > 0:
    logger.info("Loading custom URLs ...")
    url_docs = asyncio.run(process_urls(CUSTOM_URLS))
    logger.info(f"Number of URLs loaded: {len(url_docs)}")

# Combine PDF and Markdown documents
all_docs = pdf_docs + markdown_docs + url_docs
if len(all_docs) > 0:

    # Split text
    logger.info(f"Splitting documents (chunk size: {CHUNK_SIZE}) ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    document_chunks = text_splitter.split_documents(all_docs)

    # Create embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Create FAISS index from the document chunks with progress bar
    with tqdm(total=len(document_chunks), desc="Ingesting documents") as pbar:
        db = None
        os.makedirs(VECTOR_STORAGE_FOLDER, exist_ok=True)
        if VECTOR_USE_CACHE and os.path.exists(VECTOR_INDEX_FILE):
            db = FAISS.load_local(VECTOR_STORAGE_FOLDER, embeddings_model, allow_dangerous_deserialization=True)
        else:
            for d in document_chunks:
                if db:
                    db.add_documents([d])
                else:
                    db = FAISS.from_documents([d], embeddings_model)
                pbar.update(1)
            db.save_local(VECTOR_STORAGE_FOLDER)
            logger.info(f"Saved vectors to {VECTOR_STORAGE_FOLDER}")
        # Configure the retriever with Maximal Marginal Relevance (MMR) for a good balance of relevance and diversity.
        # Here, 'k' represents the number of documents to retrieve, and 'lambda_mult' adjusts the balance between relevance and diversity.
        # Setting 'lambda_mult' to 0.5 gives equal importance to both. This can be particularly effective for technical and varied datasets.
        retriever = db.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 10, 'lambda_mult': 0.5}
                    )

# Load model
device_str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Loading model {MODEL_PATH} (Torch: {torch.__version__}, Device: {device_str})")
model, tokenizer = instantiate_huggingface_model(MODEL_PATH, device_map=device_str)
logger.info(get_number_of_trainable_model_parameters(model))

# Configure text generation pipeline
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,  # Lower temperature reduces randomness
    repetition_penalty=1.1,  # Discourages repeated phrases
    return_full_text=True,
    max_new_tokens=MAX_OUTPUT_LENGTH,  # Limit on the length of generated text
    do_sample=True,  # Sampling introduces variability in responses
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Set up prompt for RAG
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create LLM chain for RAG
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

# Interaction loop with suppressed UserWarnings
while True:
    user_input = input("\033[34mYou: \033[0m")  # User input prompt
    start_time = time.time()
    
    # Suppress UserWarnings within this block
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        response = rag_chain.invoke(user_input)
    
    # Function to convert Document objects to dictionaries
    def convert_document_to_dict(doc):
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    
    # Function to recursively convert objects to dictionaries
    def convert_to_dict(obj):
        if isinstance(obj, Document):
            return convert_document_to_dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj
    
    # Convert objects to JSON-serializable format
    response_json = convert_to_dict(response)
    
    # Strip everything between [INST] in the text
    if 'text' in response_json:
        response_json['text'] = re.sub(r'\[INST].*?\[/INST]', '', response_json['text'], flags=re.DOTALL)
    
    # Convert the source to relative path
    if 'context' in response_json:
        for item in response_json['context']:
            if 'metadata' in item and 'source' in item['metadata']:
                source_path = item['metadata']['source']
                relative_path = os.path.relpath(source_path, current_directory)
                item['metadata']['source'] = relative_path
    
    # Pretty print the entire JSON response
    pretty_response = json.dumps(response_json, indent=4)
    logger.info(f"\033[32m{pretty_response}\033[0m")
    
    end_time = time.time()
    response_time = end_time - start_time
    logger.info(f"\033[90mOutput generated in {response_time:.2f} seconds\033[0m")
