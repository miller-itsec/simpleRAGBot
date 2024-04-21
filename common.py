# common.py
"""
Common utility functions used in simpleRAGBot

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
import glob
import json
import logging
import os
import re

import numpy as np
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


# Function to convert Document objects to dictionaries
def convert_document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }


def pretty_print_documents(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i + 1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )


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


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert array to list
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int to Python int
        return json.JSONEncoder.default(self, obj)


def convert_to_serializable(obj):
    """ Convert non-serializable objects to serializable format. """
    if isinstance(obj, (float, int)):  # This will cover Python floats and numpy.float32 alike
        return float(obj)
    elif isinstance(obj, dict):  # Recursively apply to dictionaries
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # Apply to each item in a list
        return [convert_to_serializable(i) for i in obj]
    return obj


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


def is_valid_content(content):
    if len(content) == 0:
        return False
    # List of phrases indicating invalid or error content
    invalid_phrases = ["403 ERROR", "Request blocked"]
    return not any(phrase in content for phrase in invalid_phrases)


def clean_source_path(source, custom_paths):
    # Strip the custom directory base from the source path
    for path in custom_paths:
        if source.startswith(path):
            # Remove the base path and return the relative path
            return source[len(path):].lstrip("\\/")
    return source


def clean_small_documents(directory_path, pattern, size_threshold_bytes):
    # Use glob to find all files recursively
    all_files = glob.glob(os.path.join(directory_path, '**', pattern), recursive=True)
    # Initialize a counter for deleted files
    deleted_files_count = 0
    # Check each file
    for file_path in all_files:
        # Ensure it's a file
        if os.path.isfile(file_path):
            # Check file size
            if os.path.getsize(file_path) < size_threshold_bytes:
                # Remove file
                os.remove(file_path)
                deleted_files_count += 1
                logging.info(f"Deleted {file_path}")  # Optional: Comment out if you don't want to log
    logging.info(f"Total deleted files: {deleted_files_count}")  # Report the number of deleted files


def sanitize_filename(url):
    """
    Converts a URL into a sanitized filename by removing non-filename characters and optionally hashing.
    """
    # Remove http:// or https://
    name = re.sub(r'^https?://', '', url)
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    # Trim to avoid extremely long filenames
    return name[:255]  # Limit based on common filesystem max filename length constraints
