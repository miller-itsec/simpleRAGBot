# common.py
"""
Common functions used in simpleRAGBot

(c) Jan Miller (@miller_itsec) for OPSWAT, Inc.
"""
import re
from bs4 import BeautifulSoup
from langchain.docstore.document import Document


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
