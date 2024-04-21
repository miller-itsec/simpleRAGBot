# server.py
"""
Server Module for simpleRAGBot

This module establishes the Flask-based web server for the simpleRAGBot system, handling all web-based interactions. It provides endpoints for processing prompts, retrieving results, and obtaining system information. It integrates a ThreadPoolExecutor for managing asynchronous task execution of NLP processing requests and utilizes Flask's CORS and Limiter for security and rate limiting.

Key functionalities include:
- Setting up a RESTful API for submitting prompts and fetching their processed results.
- Managing a queue of prompts to be processed by the system using a concurrent task execution model.
- Dynamically handling incoming requests and distributing them across available resources to optimize throughput.
- Serving as the communication layer between the web interface and the core processing logic encapsulated in the RAGServer class.

This module leverages the Flask framework to route HTTP requests and to format HTTP responses, providing a reliable and scalable server setup for processing complex NLP tasks in real-time.

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
import queue
import threading
from flask_cors import CORS
import uuid

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from concurrent.futures import ThreadPoolExecutor, as_completed

from processing import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global server


class RAGServer:
    def __init__(self, app_name, app_version, model, model_name, rag_chain):
        self.app_name = app_name
        self.app_version = app_version
        self.model = model
        self.model_name = model_name
        self.rag_chain = rag_chain
        self.prompt_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=WEBSERVER_MAX_WORKERS)
        self.prompt_status = {}
        self.prompt_responses = {}
        self.keep_running = True

    def process_prompt_task(self, prompt_id, this_prompt):
        self.prompt_status[prompt_id] = "IN_PROGRESS"
        try:
            self.prompt_responses[prompt_id] = get_prettified_prompt_result(this_prompt, self.rag_chain)
            self.prompt_status[prompt_id] = "SUCCESS"
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {e}")
            self.prompt_status[prompt_id] = "FAILED"

    def process_queue(self):
        while self.keep_running:
            try:
                prompt_id, this_prompt = self.prompt_queue.get(block=True, timeout=1.0)
                future = self.executor.submit(self.process_prompt_task, prompt_id, this_prompt)

                try:
                    # Wait for the task to complete, with a timeout
                    for _ in as_completed([future], timeout=TASK_PROCESSING_TIMEOUT):
                        future.result()  # This will either return the result or raise an exception if the task failed
                except TimeoutError:
                    logger.error(
                        f'Task processing timed out after {TASK_PROCESSING_TIMEOUT} seconds for prompt_id: {prompt_id}')
                    self.prompt_status[prompt_id] = "TIMEOUT"
                    self.prompt_responses[prompt_id] = "Processing timed out."
                except Exception as e:
                    logger.error(f"Exception during processing prompt {prompt_id}: {e}")

            except queue.Empty:
                pass

    def abort(self):
        self.keep_running = False


def run_flask_app(app_name, app_version, model, model_name, rag_chain):
    global server
    app = Flask(__name__)
    CORS(app)
    limiter = Limiter(app, default_limits=[WEBSERVER_RATE_LIMIT])

    @app.route('/prompt', methods=['POST'])
    @limiter.limit(WEBSERVER_RATE_LIMIT)
    def process_prompt():
        global server
        data = request.json
        prompt_id = str(uuid.uuid4())
        server.prompt_queue.put((prompt_id, data['prompt']))
        server.prompt_status[prompt_id] = "IN_QUEUE"
        return jsonify({"message": "Prompt received", "prompt_id": prompt_id}), 200

    @app.route('/result', methods=['GET'])
    def get_result():
        global server
        prompt_id = request.args.get('prompt_id')
        status = server.prompt_status.get(prompt_id, "UNKNOWN")
        response = server.prompt_responses.get(prompt_id, "") if status == "SUCCESS" else ""
        return jsonify({"prompt_id": prompt_id, "status": status, "response": response})

    @app.route('/system', methods=['GET'])
    def system_info():
        try:
            global server
            model_stats = get_model_stats(server.model)
            return jsonify({
                'app_name': server.app_name,
                'app_version': server.app_version,
                'model_name': server.model_name,
                'product_names': PRODUCT_NAMES,
                'model_stats': model_stats
            })
        except Exception as e:
            logger.error("Exception", e)

    logger.info(
        f"Starting webserver on localhost:{WEBSERVER_PORT} (Workers: {WEBSERVER_MAX_WORKERS}, Rate limit: {WEBSERVER_RATE_LIMIT})")
    server = RAGServer(app_name, app_version, model, model_name, rag_chain)
    threading.Thread(target=server.process_queue, daemon=True).start()
    app.run(port=WEBSERVER_PORT)


def abort_flask_app():
    global server
    server.abort()
