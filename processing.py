# processing.py
"""
Processing Module for simpleRAGBot

This module is designed to handle all aspects of document processing for the simpleRAGBot system. It includes functionalities for loading documents, performing advanced natural language processing, and interfacing with the latest machine learning models to facilitate text transformation and summarization. The key components include asynchronous web scraping, document transformation using Playwright, and leveraging Transformer models for context-aware processing.

The module encapsulates:
- Document loading from various sources including local Markdown files and the web via asynchronous scraping.
- Text transformation using custom and built-in document transformers to prepare text for processing.
- Instantiation and management of Hugging Face's transformer models, with optional quantization for efficient operation on diverse hardware.
- Functions to calculate and log model statistics to monitor performance and capabilities.
- Processing web content asynchronously to maximize efficiency and minimize response times.

This module acts as the backend processor for the Flask-based web server defined in the server.py module, providing all necessary NLP functionalities required by the simpleRAGBot's RAG (Retrieval-Augmented Generation) architecture.

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
import sys
import asyncio
import random
import time
import warnings
from tqdm import tqdm

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import *
from common import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Function to instantiate Hugging Face models with optional quantization
def instantiate_huggingface_model(model_name, use_4bit=True, bnb_4bit_compute_dtype="float16",
                                  bnb_4bit_quant_type="nf4",
                                  use_nested_quant=False, llm_int8_threshold=400.0, device_map="auto", use_cache=True,
                                  trust_remote_code=None, pad_token=None, padding_side="left"):
    # Sets configuration for model quantization and device mapping
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

    # Instantiates the model and tokenizer with specified parameters
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


# Functions for model and system diagnostics
def print_model_stats(model):
    stats = get_model_stats(model)
    logger.info(f"Trainable model parameters: {stats['trainable_model_params']}\n"
                f"All model parameters: {stats['all_model_params']}\n"
                f"Percentage of trainable model parameters: {stats['percentage_trainable']}")


def get_model_stats(model):
    # Computes and returns statistics about model parameters
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    percentage_trainable = 100 * trainable_model_params / all_model_params
    return {
        "trainable_model_params": trainable_model_params,
        "all_model_params": all_model_params,
        "percentage_trainable": f"{percentage_trainable:.2f}%"
    }


def invoke_rag(prompt, rag_chain):
    response = rag_chain.invoke(prompt)
    response_json = {}
    if isinstance(response, str):
        response_json['text'] = response
    response_json = convert_to_dict(response)
    response_json['text'] = re.sub(r'\[INST].*?\[/INST]', '', response_json.get('text', ''), flags=re.DOTALL)
    response_json['text'] = response_json['text'].replace("\n### \n", "", 1).strip()
    response_json = convert_to_serializable(response_json)
    return json.dumps(response_json, indent=4, cls=NumpyEncoder)


def get_prettified_prompt_result(prompt, rag_chain, output=True):
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.info("Querying the LLM. Give me some time ...")
        this_pretty_response = invoke_rag(prompt, rag_chain)
    if output:
        logger.info(f"\033[32m{this_pretty_response}\033[0m")
    end_time = time.time()
    response_time = end_time - start_time
    logger.info(f"\033[90mOutput generated in {response_time:.2f} seconds\033[0m")
    return this_pretty_response


# Defines classes for handling different types of document loaders
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


# Custom loader class for asynchronous web scraping using Playwright
class CustomAsyncChromiumLoader(AsyncChromiumLoader):
    def __init__(self, urls, user_agent=None):
        super().__init__(urls)
        self.user_agent = user_agent

    async def scrape_playwright(self, url: str) -> str:
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


# Async function to load and transform URL content
async def load_and_transform_url(loader, url, delay=0.0):
    try:
        await asyncio.sleep(delay)
        html_content = await loader.scrape_playwright(url)
        html_content = strip_html(html_content)
        html_content = re.sub(r'[\s]+', ' ', html_content)  # Remove excessive whitespace
        logger.info(f"Extracted Text from URL:\n{html_content[:500]}...")  # Print the first 500 characters
        return Document(page_content=html_content, metadata={"source": url})
    except Exception as e:
        logger.exception("Failed to load and transform URL: " + url, exc_info=e)
        return None


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
        if doc is None:
            continue
        # Check if the content is valid
        if is_valid_content(doc.page_content):
            docs.append(doc)
        else:
            logger.info(f"Invalid content detected for URL: {doc.metadata['source']}")
    return html2text.transform_documents(docs)
