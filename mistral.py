import asyncio
import os
import glob
import json
import time
import torch
import re
import warnings

from tqdm import tqdm

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

# Set up directories
current_directory = os.getcwd()
MODEL_PATH = os.path.join(current_directory, "Mistral-7B-Instruct-v0.2")
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
CUSTOM_PDFS = os.path.join(current_directory, "custom_documents")
CUSTOM_MARKDOWN = os.path.join(current_directory, "custom_documents")
CUSTOM_URLS = ["https://www.opswat.com/products/metadefender/sandbox",
                "https://www.opswat.com/blog/getting-started-with-opswat-filescan-sandbox-sandboxing-made-easy",
                "https://www.opswat.com/blog/opswat-filescan-sandbox-v1-9-1-new-updates-and-releases",
                "https://www.opswat.com/blog/metadefender-sandbox-v1-9-2-sharpen-your-threat-response-with-the-latest-enhanced-features",
                "https://www.opswat.com/blog/introducing-opswat-metadefender-sandbox-v1-9-3"]
CHUNK_SIZE = 200
MAX_OUTPUT_LENGTH = 512

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


def print_number_of_trainable_model_parameters(model):
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
                    content = re.sub(r'\[([^\]]+)]\([^\)]+\)', r'\1', content)
                    content = strip_base64_images(content)
                    document = Document(page_content=content, metadata={"source": file_path})
                    documents.append(document)
        return documents


pdf_docs = []
if len(CUSTOM_PDFS) > 0:
    # Load PDFs
    print("Loading custom PDF files ...")
    pdf_loader = PyPDFDirectoryLoader(CUSTOM_PDFS)
    pdf_docs = pdf_loader.load()
    print(f"Number of PDF documents loaded: {len(pdf_docs)}")

markdown_docs = []
if len(CUSTOM_MARKDOWN) > 0:
    # Load Markdown files
    print("Loading custom markdown files ...")
    markdown_loader = MarkdownDirectoryLoader(CUSTOM_MARKDOWN)
    markdown_docs = markdown_loader.load()
    print(f"Number of Markdown documents loaded: {len(markdown_docs)}")

url_docs = []


async def load_and_transform_url(loader, html2text, url):
    # Load URL content and return a document
    html_content = await loader.ascrape_playwright(url)
    return Document(page_content=html_content, metadata={"source": url})


async def process_urls(urls):
    loader = AsyncChromiumLoader(urls)
    html2text = Html2TextTransformer()
    tasks = [load_and_transform_url(loader, html2text, url) for url in urls]
    docs = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(urls), desc="Processing URLs"):
        docs.append(await future)
    return html2text.transform_documents(docs)


if len(CUSTOM_URLS) > 0:
    print("Loading custom URLs ...")
    url_docs = asyncio.run(process_urls(CUSTOM_URLS))
    print(f"Number of URLs loaded: {len(url_docs)}")

    # Optionally, print extracted text from each URL
    for doc in url_docs:
        print(f"Extracted Text from URL:\n{doc.page_content[:500]}...")  # Print the first 500 characters


# Combine PDF and Markdown documents
all_docs = pdf_docs + markdown_docs + url_docs
if len(all_docs) > 0:

    # Split text
    print(f"Splitting documents (chunk size: {CHUNK_SIZE}) ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(all_docs)

    # Create embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Create FAISS index from the document chunks with progress bar
    with tqdm(total=len(document_chunks), desc="Ingesting documents") as pbar:
        db = None
        for d in document_chunks:
            if db:
                db.add_documents([d])
            else:
                db = FAISS.from_documents([d], embeddings_model)
            pbar.update(1)
        retriever = db.as_retriever()

# Define prompt template
prompt_template = """
### [INST] Instruction: 
Utilize your knowledge base to provide an accurate and informative response. Consider the context provided below to 
better understand the nuances of the question.

**Context:**  
{context}

### QUESTION:
{question}

**Note:** 
In your response, ensure that you address specific points mentioned in the context and question.
 Offer clear, concise, and relevant information based on the available data.

[/INST]
 """

# Load model
device_str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model {MODEL_PATH} (Torch: {torch.__version__}, Device: {device_str})")
model, tokenizer = instantiate_huggingface_model(MODEL_PATH, device_map=device_str)
print(print_number_of_trainable_model_parameters(model))

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
    print(f"\033[32m{pretty_response}\033[0m")
    
    end_time = time.time()
    response_time = end_time - start_time
    print(f"\033[90mOutput generated in {response_time:.2f} seconds\033[0m")
