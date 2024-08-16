*llm_toolkit_api: A API for Enhancing Large Language Models*

This repository provides a  API designed to extend the capabilities of large language models (LLMs) like vision models. It offers functionalities such as image captioning and text embedding. The API is built using FastAPI and is designed to be easily deployable and customizable.

New Features:

Website parser: The API now includes a website parser that can extract text from a given URL. This feature is useful for generating text embeddings from web content.

Key Features:

Image Captioning: Leverages the power of Florence-2 to generate captions for images. You can provide either a base64 or a URL to an image. Other abaliables such as optical character recognition (OCR) of Florence-2 are also supported.
Text Embedding: (Optional) If you provide an embedding model path, the API can generate text embeddings using the Sentence Transformers library.

Customization: Configure the API using command-line arguments to specify:
* Port and host for the server

* Model path (e.g., "microsoft/Florence-2-large-ft")

* Data type (float16, float32, bfloat16)

* Device (cuda or cpu)

Installation:

Ensure you have Python and pip installed.
Create a virtual environment (recommended):
   python3 -m venv env
   source env/bin/activate


Install the required dependencies:
   pip install -r requirements.txt


Running the API:
   python main.py

You can customize the port and other settings using command-line arguments (see the code for details).

Usage:

The API exposes two endpoints:

/generate/ (POST):
   {
     "prompt": "<CAPTION>",
     "task_type": "<CAPTION>",
     "file_or_url": "https://example.com/image.jpg"
   }


/embed/ (POST):
   {
     "text": "Hello, world!"
   }


Note:

The /embed/ endpoint requires you to provide an embedding model path when running the API.
Refer to the code comments for more detailed information on the input parameters and expected responses.
