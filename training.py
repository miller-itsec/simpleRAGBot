import datetime
import json
import logging
import sys

from openai import OpenAI
from tqdm import tqdm

from config import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_openai_client(api_key):
    return OpenAI(api_key=api_key)


def generate_training_data(docs):
    api_key = OPENAI_API_KEY if OPENAI_API_KEY else os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not found. Please set the API key.")
        return

    client = create_openai_client(api_key)
    # Prepare a file to store the training data
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    training_data_file = f"training_data_{timestamp}.json"
    all_data = []  # This list will store all the JSON objects

    with open(training_data_file, 'w') as outfile, tqdm(total=len(docs), desc="Creating training data") as pbar:
        for doc in docs:
            prompt = TRAINING_DATA_PROMPT.format(content=doc.page_content)
            try:
                metadata = doc.metadata
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=API_REQUEST_TIMEOUT
                )
                if response.choices:
                    print(response.choices)
                    json_data = response.choices[0].message.content.strip()
                    logger.info(f"Response: {json_data}")
                    # Extend all_data list with json_objects if it's a list, else append
                    json_objects = json.loads(json_data)
                    if isinstance(json_objects, list):
                        for obj in json_objects:
                            # Merge each object with metadata
                            obj.update(metadata)
                            all_data.append(obj)
                    else:
                        json_objects.update(metadata)
                        all_data.append(json_objects)
                else:
                    logger.error("No suitable response received from the API.")
            except Exception as e:
                logger.error(f"Error generating training data: {e}")
            pbar.update(1)

    # Write all collected data to a single JSON file
    with open(training_data_file, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)  # Serialize the list of dictionaries as a JSON array

    logger.info(f"Training data has been written to {training_data_file}")
