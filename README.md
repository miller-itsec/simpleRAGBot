# simpleRAGBot: An Advanced NLP Chatbot

## Overview
`simpleRAGBot` is a state-of-the-art Natural Language Processing (NLP) chatbot that leverages the Retrieval-Augmented Generation (RAG) methodology to provide context-aware, accurate, and relevant responses. Utilizing advanced models and a comprehensive document base, the chatbot processes various document types, including PDFs, Markdown files, and web URLs.

### What is RAG?
Retrieval-Augmented Generation (RAG) combines the benefits of both retrieval-based and generative NLP models. It retrieves relevant information from a document collection and uses it to generate informed and context-aware responses. This approach allows for enhanced accuracy and relevance in natural language understanding and generation.

## Features
- **Asynchronous Web Scraping**: For dynamic and varied content acquisition.
- **Document Processing and Indexing**: Using FAISS (Facebook AI Similarity Search) for rapid retrieval and effective segmentation.
- **RAG Chain with Mistral-7B-Instruct-v0.2 Model**: Enriched response generation leveraging a powerful generative model.
- **Web Server and CLI Support**: Offers both a Flask-based API for web interaction and a command-line interface for direct usage.
- **Configurable Multi-threaded System**: Ensures efficient prompt processing and response generation.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip (Python package installer)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/simpleRAGBot.git
   ```
2. Navigate to the cloned directory:
   ```sh
   cd simpleRAGBot
   ```
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface (CLI)
To use `simpleRAGBot` via the command line:
1. Run the main script:
   ```sh
   python main.py
   ```
2. Follow the on-screen prompts to input your queries.

### Web Interface
`simpleRAGBot` also offers a web interface powered by Flask:
1. Start the Flask server:
   ```sh
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:5000` (or the configured port).

### Endpoints
- **/prompt** (POST): Submit a prompt/query.
- **/result** (GET): Fetch the result of a submitted prompt. Requires `prompt_id` as a parameter.
- **/system** (GET): Get system information, including the model used, app version, and available product names.

### Default Ports
- Flask server: Port 5000 (configurable in `config.py`)

### Configurations
Modify `config.py` to adjust settings like ports, model parameters, document paths, etc.

## Packages and Libraries
This project relies on several key Python libraries:
- Flask: For the web server and API endpoints.
- PyTorch: For handling machine learning models.
- Transformers: From Hugging Face, for pretrained NLP models.
- FAISS: For efficient similarity search and clustering of dense vectors.
- TQDM: For progress bars in loops and console output.
- Playwright: For asynchronous web scraping.

## Contributing
Contributions to `simpleRAGBot` are welcome! Please read `CONTRIBUTING.md` for guidelines on how to contribute.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Jan Miller - [@miller_itsec](https://twitter.com/miller_itsec)

### Example:

![Example](example.png?raw=true "Example output")
