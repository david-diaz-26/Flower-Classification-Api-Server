Flower Classification and Description API

This FastAPI server accepts flower images, classifies the flower species using a TensorFlow Keras model, and then generates a detailed description using the Ollama LLM.


Features

Image Upload: Accepts an image file of a flower.

Classification: Uses a pre-trained TensorFlow Keras model to classify flower species.

Description Generation: Sends a custom prompt to the Ollama language model to generate a rich description including history, traits, and cultural info.

JSON API: Returns JSON response containing the predicted flower name and the LLM-generated description.


How to Run Locally:

Prerequisites
Python 3.8+

TensorFlow (with Keras)

FastAPI

Ollama CLI installed and configured

The llama3.2 model installed in Ollama (can be modified for desired model)

Installation

pip install -r requirements.txt

Install Ollama

ollama install llama3.2

ngrok

Running the Server
IMPORTANT: should have ngrok tunnel on the same port as FastAPI

ngrok http 8000

uvicorn main:app --host 0.0.0.0 --port 8000


How to run on Docker:

Install Docker

Have Docker Daemon running

docker build -t flower-api .

docker run -p 8000:8000 flower-api

