# main.py
"""
simpleRAGBot Main Script

This script is the heart of simpleRAGBot, a powerful system leveraging advanced natural language processing to generate context-aware responses from a comprehensive document base. It processes various document types such as PDFs, Markdown files, and web URLs. The key steps in the pipeline include document loading, processing, embedding with FAISS, and generating responses through the Retrieval-Augmented Generation chain using Mistral-7B-Instruct-v0.2 model.

Features:
- Asynchronous web scraping for varied content acquisition.
- Effective document segmentation and rapid retrieval via FAISS indexing.
- Utilizes the RAG chain with the Mistral model for enriched response generation.
- Flask-based web server setup for processing API-driven prompts.
- Configurable multi-threaded prompt processing system.

Usage:
Directly execute this script after configuring through 'config.py'. It supports both a command-line interface and a Flask-based API for external prompts.

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
import pickle
import sys
import asyncio
import glob
import json
import logging
import time
import queue
import random
import threading
from flask_cors import CORS

from googlesearch import search
from typing import Dict, Any

import torch
import uuid
import warnings

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

import langchain
from langchain.retrievers import EnsembleRetriever
from langchain_community.cache import InMemoryCache
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input

from config import *
from common import *

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

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cache = None


def instantiate_huggingface_model(model_name, use_4bit=False, bnb_4bit_compute_dtype="float16",
                                  bnb_4bit_quant_type="nf4",
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


def print_model_stats(model):
    stats = get_model_stats(model)
    logger.info(f"Trainable model parameters: {stats['trainable_model_params']}\n"
                f"All model parameters: {stats['all_model_params']}\n"
                f"Percentage of trainable model parameters: {stats['percentage_trainable']}")


def get_model_stats(model):
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    percentage_trainable = 100 * trainable_model_params / all_model_params
    return {
        "trainable_model_params": trainable_model_params,
        "all_model_params": all_model_params,
        "percentage_trainable": f"{percentage_trainable:.2f}%"
    }


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
    try:
        await asyncio.sleep(delay)
        html_content = await loader.ascrape_playwright(url)
        html_content = strip_html(html_content)
        html_content = re.sub(r'[\s]+', ' ', html_content)  # Remove excessive whitespace
        logger.debug(f"Extracted Text from URL:\n{html_content[:500]}...")  # Print the first 500 characters
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


def invoke_rag(prompt, rag_chain):
    response = rag_chain.invoke(prompt)
    response_json = convert_to_dict(response)
    response_json['text'] = re.sub(r'\[INST].*?\[/INST]', '', response_json.get('text', ''), flags=re.DOTALL)
    response_json['text'] = response_json['text'].replace("\n### \n", "", 1).strip()

    if 'context' in response_json:
        for item in response_json['context']:
            if 'metadata' in item and 'source' in item['metadata']:
                source_path = item['metadata']['source']
                relative_path = os.path.relpath(source_path, current_directory)
                item['metadata']['source'] = relative_path

    return json.dumps(response_json, indent=4)


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


def get_product_instructions(question):
    question_lower = question.lower()
    for product, synonyms in PRODUCT_NAMES.items():
        matching_synonyms = [synonym for synonym in synonyms if synonym.lower() in question_lower]
        if matching_synonyms:
            # Use the original case of 'product' in the returned string
            synonyms_text = ', '.join(matching_synonyms)
            return f"Consider the potential context provided below to better understand the nuances of the question, focusing on {product} (including related terms like {synonyms_text})."
    return "Consider the potential context provided below to better understand the nuances of the question."


class DynamicPromptRunnable(Runnable):
    def __init__(self, llm_chain, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm_chain = llm_chain

    def invoke(self, input: Input, config: RunnableConfig | None = None):
        context = input.get('context', '')
        question = input.get('question', '')
        product_instructions = get_product_instructions(question)
        return self.llm_chain.invoke({"product_instructions": product_instructions, "context": context, "question": question})


def save_bm25_index(bm25_retriever):
    with open(BM25_CACHE_FILE, 'wb') as file:
        pickle.dump(bm25_retriever, file)
    return bm25_retriever


def load_bm25_index():
    with open(BM25_CACHE_FILE, 'rb') as file:
        bm25_retriever = pickle.load(file)
    return bm25_retriever


def get_app_name():
    return "simpleRAGBot"


def get_app_version():
    return "1.0.0"


def main():
    logger.info(f"Welcome to {get_app_name()} v{get_app_version()}")

    # Create embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    faiss_db = None
    bm25_retriever = None
    if VECTOR_USE_CACHE and os.path.exists(VECTOR_INDEX_FILE):
        # Calculate the age of the file
        file_mod_time = os.path.getmtime(VECTOR_INDEX_FILE)
        current_time = time.time()
        if (current_time - file_mod_time) > VECTOR_CACHE_TTL * 60:
            logger.info(f"Vector index file is older than {VECTOR_CACHE_TTL} hours, rebuilding database.")
        else:
            faiss_db = FAISS.load_local(VECTOR_STORAGE_FOLDER, embeddings_model, allow_dangerous_deserialization=True)

    # Get ensemble retriever for performance optimization
    if VECTOR_USE_CACHE and os.path.exists(BM25_CACHE_FILE):
        bm25_retriever = load_bm25_index()

    if bm25_retriever is None or faiss_db is None:
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

        google_docs = []

        if len(CUSTOM_GOOGLE_QUERIES) > 0:
            additional_links = []
            for query in CUSTOM_GOOGLE_QUERIES:
                logger.info(f"Querying Google for additional URLs for {query} ...")
                results = search(query, lang="en", num_results=10, timeout=5)
                for result in results:
                    # Only add result if it's not in CUSTOM_URLS or seen before
                    if result not in CUSTOM_URLS and result not in additional_links:
                        additional_links.append(result)
            logger.info(f"Loading {len(additional_links)} additional URLs ...")
            logger.debug(f"{additional_links}")
            google_docs = asyncio.run(process_urls(additional_links))
            logger.info(f"Number of URLs loaded: {len(google_docs)}")

        url_docs = []

        if len(CUSTOM_URLS) > 0:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info(f"Loading {len(CUSTOM_URLS)} custom URLs ...")
            url_docs = asyncio.run(process_urls(CUSTOM_URLS))
            logger.info(f"Number of URLs loaded: {len(url_docs)}")

        # Combine PDF and Markdown documents
        all_docs = pdf_docs + markdown_docs + url_docs + google_docs

        # Build a vector database, if we have custom content
        if len(all_docs) > 0:

            # Split text
            logger.info(f"Splitting documents (chunk size: {CHUNK_SIZE}) ...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            document_chunks = text_splitter.split_documents(all_docs)

            if faiss_db is None:
                # Create FAISS index from the document chunks with progress bar
                with tqdm(total=len(document_chunks), desc="Ingesting documents") as pbar:
                    os.makedirs(VECTOR_STORAGE_FOLDER, exist_ok=True)
                    for d in document_chunks:
                        if faiss_db:
                            faiss_db.add_documents([d])
                        else:
                            faiss_db = FAISS.from_documents([d], embeddings_model)
                        pbar.update(1)
                    faiss_db.save_local(VECTOR_STORAGE_FOLDER)
                    logger.info(f"Saved high density vectors to {VECTOR_STORAGE_FOLDER}")

            if bm25_retriever is None:
                # Create BM25 index (should be much quicker)
                bm25_retriever = BM25Retriever.from_documents(document_chunks)
                bm25_retriever = save_bm25_index(bm25_retriever)
                logger.info(f"Saved low density vectors to {VECTOR_STORAGE_FOLDER}")

    faiss_retriever = faiss_db.as_retriever(
        search_type=RETRIEVER_FAISS_SEARCH_TYPE,
        search_kwargs=RETRIEVER_FAISS_SEARCH_ARGS
    )
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                           weights=ENSEMBLE_RETRIEVER_WEIGHTS)

    # Load model
    device_str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(MODEL_PATH)
    logger.info(f"Loading model {model_name} (Torch: {torch.__version__}, Device: {device_str})")
    model, tokenizer = instantiate_huggingface_model(MODEL_PATH, device_map=device_str)
    print_model_stats(model)

    # Configure text generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=MODEL_TEMPERATURE,  # Lower temperature reduces randomness
        early_stopping=True,
        repetition_penalty=1.1,  # Discourages repeated phrases
        return_full_text=True,
        max_new_tokens=MAX_OUTPUT_LENGTH,  # Limit on the length of generated text
        do_sample=True,  # Sampling introduces variability in responses
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Set up prompt for RAG
    prompt = PromptTemplate(
        input_variables=["product_instructions", "context", "question"],
        template=PROMPT_TEMPLATE,
    )

    # Create LLM chain for RAG
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    if ENABLE_LANGCHAIN_CACHE:
        langchain.llm_cache = InMemoryCache()
    dynamic_prompt_runnable = DynamicPromptRunnable(llm_chain)
    rag_chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | dynamic_prompt_runnable
    )

    keep_running = True
    cli_thread = None
    flask_thread = None
    prompt_queue = queue.Queue()

    if ENABLE_WEBSERVER:
        logger.info(
            f"Starting webserver on localhost:{WEBSERVER_PORT} (Workers: {WEBSERVER_MAX_WORKERS}, Rate limit: {WEBSERVER_RATE_LIMIT})")
        app = Flask(get_app_name())
        CORS(app)
        limiter = Limiter(get_app_name, request_identifier=get_remote_address, default_limits=["1 per minute"])
        executor = ThreadPoolExecutor(max_workers=WEBSERVER_MAX_WORKERS)  # Adjust max_workers as needed
        prompt_status = {}
        prompt_responses = {}

        def process_prompt_task(prompt_id, this_prompt):
            prompt_status[prompt_id] = "IN_PROGRESS"
            try:
                prompt_responses[prompt_id] = get_prettified_prompt_result(this_prompt, rag_chain)
                prompt_status[prompt_id] = "SUCCESS"
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_id}: {e}")
                prompt_status[prompt_id] = "FAILED"

        @app.route('/prompt', methods=['POST'])
        @limiter.limit(WEBSERVER_RATE_LIMIT)
        def process_prompt():
            data = request.json
            prompt_id = str(uuid.uuid4())
            prompt_queue.put((prompt_id, data['prompt']))
            prompt_status[prompt_id] = "IN_QUEUE"
            return jsonify({"message": "Prompt received", "prompt_id": prompt_id}), 200

        @app.route('/result', methods=['GET'])
        def get_result():
            prompt_id = request.args.get('prompt_id')
            status = prompt_status.get(prompt_id, "UNKNOWN")
            response = prompt_responses.get(prompt_id, "") if status == "SUCCESS" else ""
            return jsonify({"prompt_id": prompt_id, "status": status, "response": response})

        @app.route('/system', methods=['GET'])
        def system_info():
            model_stats = get_model_stats(model)
            return jsonify({
                'app_name': get_app_name(),
                'app_version': get_app_version(),
                'model_name': model_name,
                'product_names': PRODUCT_NAMES,
                'model_stats': model_stats
            })

        def process_queue():
            while keep_running:
                try:
                    prompt_id, this_prompt = prompt_queue.get(block=True, timeout=1.0)
                    future = executor.submit(process_prompt_task, prompt_id, this_prompt)

                    try:
                        # Wait for the task to complete, with a timeout
                        for _ in as_completed([future], timeout=TASK_PROCESSING_TIMEOUT):
                            future.result()  # This will either return the result or raise an exception if the task failed
                    except TimeoutError:
                        logger.error(
                            f'Task processing timed out after {TASK_PROCESSING_TIMEOUT} seconds for prompt_id: {prompt_id}')
                        prompt_status[prompt_id] = "TIMEOUT"
                        prompt_responses[prompt_id] = "Processing timed out."
                    except Exception as e:
                        logger.error(f"Exception during processing prompt {prompt_id}: {e}")

                except queue.Empty:
                    pass

        def run_flask_app():
            threading.Thread(target=process_queue, daemon=True).start()
            app.run(port=WEBSERVER_PORT)

        flask_thread = Thread(target=run_flask_app)
        flask_thread.start()

    if ENABLE_COMMANDLINE:
        def run_command_line_interface():
            time.sleep(2) # Wait two seconds
            # Command line interaction loop
            while True:
                try:
                    user_input = input("\u001b[35mYour prompt: \033[0m")
                    get_prettified_prompt_result(user_input, rag_chain)
                except EOFError:
                    break  # Break the loop if EOFError occurs
                except KeyboardInterrupt:
                    break  # Handle Ctrl+C interruption here
            logger.info("Shutting down commandline ...")

        cli_thread = Thread(target=run_command_line_interface)
        cli_thread.start()

    try:
        # Wait for threads to complete or handle keyboard interrupt for graceful shutdown
        if cli_thread is not None:
            logger.info("Waiting for the commandline to quit")
            cli_thread.join()
        if flask_thread is not None:
            logger.info("Waiting for the webserver to quit")
            flask_thread.join()
        keep_running = False
        time.sleep(5)
        logger.info("Goodbye")
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    os._exit(0)


if __name__ == '__main__':
    main()
