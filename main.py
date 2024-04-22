# main.py
"""
simpleRAGBot Main Script

This script is the core of simpleRAGBot, a robust system utilizing advanced natural language processing technologies.
It is designed to generate context-aware responses from a comprehensive document base, which includes various document
types such as PDFs, Markdown files, and web URLs. The pipeline integrates several advanced features, including document
loading, processing, segmentation, summarization of long documents using the BART model, embedding with FAISS, and
generating responses through the Retrieval-Augmented Generation (RAG) chain using the Mistral-7B-Instruct-v0.2 model by default.

Features:
- Asynchronous web scraping for diverse content acquisition.
- Summarization of long documents using the BART model, tailored for extensive content.
- Effective document segmentation and rapid retrieval via FAISS indexing.
- Utilizes the RAG chain with the Mistral model for enriched response generation.
- Flask-based web server setup for processing API-driven prompts.
- Configurable multi-threaded prompt processing system, optimized for performance and scalability.

Usage:
Directly execute this script after configuring through 'config.py'. It supports both a command-line interface and a
Flask-based API for external prompts, allowing for flexible interaction modes depending on user preferences.

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
from __future__ import with_statement
import contextlib
import pickle

from googlesearch import search
from typing import Any

import torch

from threading import Thread

import langchain
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cache import InMemoryCache
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input

from config import *
from processing import *

from tqdm import tqdm

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

from server import run_flask_app, abort_flask_app
from training import generate_training_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_app_name():
    return "simpleRAGBot"


def get_app_version():
    return "1.0.0"


def get_product_instructions(question):
    question_lower = question.lower()
    for product, synonyms in PRODUCT_NAMES.items():
        matching_synonyms = [synonym for synonym in synonyms if synonym.lower() in question_lower]
        if matching_synonyms:
            # Use the original case of 'product' in the returned string
            synonyms_text = ', '.join(matching_synonyms)
            return (f"Consider the potential context provided below to better understand the nuances of the question, "
                    f"focusing on {product} (including related terms like {synonyms_text}) by {COMPANY_NAME}.")
    return (f"Consider the potential context provided below to better understand the nuances of the question regarding "
            f"{PRODUCT_NAMES.keys()} by {COMPANY_NAME}.")


class DynamicPromptRunnable(Runnable):
    def __init__(self, llm_chain, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm_chain = llm_chain

    def invoke(self, input: Input, config: RunnableConfig | None = None):
        context = input.get('context', [])  # list of Documents
        pretty_print_documents(context)
        # Filter documents based on the relevance score
        filtered_context = [doc for doc in context if
                            doc.metadata.get('relevance_score', 0) >= RELEVANCE_SCORE_THRESHOLD]
        question = input.get('question', '')
        product_instructions = get_product_instructions(question) if len(filtered_context) > 0 else ''
        payload = {"product_instructions": product_instructions, "context": filtered_context, "question": question}
        return self.llm_chain.invoke(payload)


def save_bm25_index(bm25_retriever):
    with open(BM25_CACHE_FILE, 'wb') as file:
        pickle.dump(bm25_retriever, file)
    return bm25_retriever


def load_bm25_index():
    with open(BM25_CACHE_FILE, 'rb') as file:
        bm25_retriever = pickle.load(file)
    return bm25_retriever


# Processes documents by summarizing content that exceeds a specified size threshold, using the BART model for summarization.
# This function also implements a caching system, storing summaries on disk to avoid redundant processing.
def prepare_documents(all_docs, summarize_large_documents=SUMMARIZE_LARGE_DOCUMENTS, summarize_model=SUMMARIZE_MODEL, use_cache=SUMMARIZE_USE_CACHE):
    if summarize_large_documents:
        os.makedirs(CUSTOM_URL_SUMMARY_STORAGE, exist_ok=True)
        tokenizer = BartTokenizer.from_pretrained(summarize_model)
        model = BartForConditionalGeneration.from_pretrained(summarize_model)

    def summarize_text(text):
        # Tokenize the input text
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    for document in all_docs:
        if isinstance(document.metadata, dict) and 'source' in document.metadata:
            source_path = document.metadata['source']
            if source_path.startswith("http"):
                if SKIP_URL_SUMMARY:
                    continue
                source_path = os.path.join(CUSTOM_URL_SUMMARY_STORAGE, sanitize_filename(source_path))
            summary_path = source_path + ".summary"
            # Check if the summary file exists and load it if present
            if use_cache and os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as file:
                    document.page_content = file.read()
                logger.info(f"Loaded summary from {summary_path}")
            else:
                # Perform summarization if the document is large enough
                document_length = len(document.page_content.encode('utf-8'))
                if summarize_large_documents and document_length > SUMMARIZE_LARGE_DOCUMENTS_THRESHOLD:
                    # Summarize the content
                    logger.info(
                        f"Summarizing {document.metadata['source']} as its content size ({document_length}) exceeds {SUMMARIZE_LARGE_DOCUMENTS_THRESHOLD}")
                    start_time = time.time()
                    document.page_content = summarize_text(document.page_content)
                    end_time = time.time()
                    response_time = end_time - start_time
                    logger.debug(f"{document.page_content}")
                    logger.info(
                        f"\033[90mCompleted summary in {response_time:.2f} seconds. New content length is {len(document.page_content)}\033[0m")
                    if use_cache:
                        # Save the summary to disk
                        with open(summary_path, 'w', encoding='utf-8') as file:
                            file.write(document.page_content)
                        logger.info(f"Saved summary to {summary_path}")
            document.metadata['source'] = clean_source_path(document.metadata['source'], [CUSTOM_PDFS, CUSTOM_MARKDOWN])


def load_custom_data():
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

        if PERFORM_CUSTOM_GOOGLE_QUERIES and len(CUSTOM_GOOGLE_QUERIES) > 0:
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

        if DOWNLOAD_CUSTOM_URLS and len(CUSTOM_URLS) > 0:
            logger.info(f"Loading {len(CUSTOM_URLS)} custom URLs ...")
            url_docs = asyncio.run(process_urls(CUSTOM_URLS))
            logger.info(f"Number of URLs loaded: {len(url_docs)}")

        # Combine PDF and Markdown documents
        all_docs = pdf_docs + markdown_docs + url_docs + google_docs

        # Build a vector database, if we have custom content
        if len(all_docs) > 0:

            # Clean up the metadata "source" fields in the context to show only the relative pathway
            prepare_documents(all_docs)

            # Use only if fine-tuning the LLM model is desired
            if GENERATE_TRAINING_DATA:
                generate_training_data(all_docs)
                logger.info("Completed training data generation. Please fine-tune the model and disable 'GENERATE_TRAINING_DATA'")
                sys.exit(0)

            # Split text
            logger.info(f"Splitting documents (chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}) ...")
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

    return faiss_db, bm25_retriever


# Main function to manage the document processing and LLM chain execution
def main():
    logger.info(f"Welcome to {get_app_name()} v{get_app_version()}")

    # Clean/remove files with too little content
    if CLEAN_SMALL_CUSTOM_DOCUMENTS:
        clean_small_documents(CUSTOM_PDFS, "*.pdf", CLEAN_SMALL_CUSTOM_DOCUMENTS_THRESHOLD_BYTES)
        clean_small_documents(CUSTOM_MARKDOWN, "*.md", CLEAN_SMALL_CUSTOM_DOCUMENTS_THRESHOLD_BYTES)

    # Get the embeddings vector databases
    faiss_db, bm25_retriever = load_custom_data()

    faiss_retriever = faiss_db.as_retriever(
        search_type=RETRIEVER_FAISS_SEARCH_TYPE,
        search_kwargs=RETRIEVER_FAISS_SEARCH_ARGS
    )
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                           weights=ENSEMBLE_RETRIEVER_WEIGHTS)

    # Setup re-rank
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

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
            {"context": compression_retriever, "question": RunnablePassthrough()}
            | dynamic_prompt_runnable
    )

    cli_thread = None
    flask_thread = None

    if ENABLE_WEBSERVER:
        flask_thread = Thread(target=run_flask_app, args=(get_app_name(), get_app_version(), model, model_name, rag_chain))
        flask_thread.start()

    if ENABLE_COMMANDLINE:
        cli_thread = Thread(target=run_command_line_interface, args=(rag_chain,))
        cli_thread.start()

    # Wait for threads to complete or handle keyboard interrupt for graceful shutdown
    try:
        if cli_thread is not None:
            logger.info("Waiting for the commandline to quit")
            cli_thread.join()
        if flask_thread is not None:
            logger.info("Waiting for the webserver to quit")
            flask_thread.join()
    except KeyboardInterrupt:
        pass
    time.sleep(2)
    logger.info("Goodbye")
    os._exit(0)


@contextlib.contextmanager
def handler():
    try:
        yield
    except Exception as e:
        abort_flask_app()


def run_command_line_interface(rag_chain):
    with handler():
        if ENABLE_WEBSERVER:
            time.sleep(2)  # Wait two seconds if the webserver started, to have a bit cleaner output
        # Command line interaction loop
        while True:
            try:
                user_input = input("\u001b[35mYour prompt: \033[0m")
                get_prettified_prompt_result(user_input, rag_chain, invoke_rag)
            except EOFError:
                break  # Break the loop if EOFError occurs
            except KeyboardInterrupt:
                break  # Handle Ctrl+C interruption here
    logger.info("Shutting down commandline ...")


if __name__ == '__main__':
    main()
