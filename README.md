# Flower Classification and Description API

This FastAPI server accepts flower images, classifies the flower species using a pre-trained TensorFlow Keras model, and generates a detailed description using the Ollama LLM. There is a baseline model which is just a CNN model, and there will be 2 transer learning models...

## Definitions

- ngrok: A secure tunneling service that exposes a local server to the internet by creating a public URL forwarding to a local port. Useful for testing and sharing locally hosted applications without deploying them.

- FastAPI: A modern, high-performance Python web framework for building APIs. It is based on standard Python type hints, enabling automatic interactive API documentation and fast development.

- Docker: A platform that uses containerization to package applications and their dependencies into lightweight, portable containers. Containers run consistently across different computing environments. Super efficient for users sharing evironments.

- Ollama: A local Large Language Model (LLM) runtime environment that enables running and interacting with models like llama3.2 on your own machine, without relying on cloud services.

- Uvicorn: A lightning-fast ASGI server implementation, often used to serve FastAPI applications. It is asynchronous and built on top of the uvloop and httptools libraries, offering great performance.

## Features

- **Image Upload:** Accepts an image file of a flower.
- **Classification:** Uses a TensorFlow Keras model to classify flower species.
- **Description Generation:** Generates rich descriptive text including history, traits, and cultural information via the Ollama language model.
- **JSON API:** Returns a JSON response containing the predicted flower name and the LLM-generated description.

## Prerequisites

- Python 3.8 or higher
- TensorFlow (with Keras)
- FastAPI
- Ollama CLI installed and configured
- `llama3.2` (or preferred model) installed in Ollama
- ngrok (for tunneling when exposing the local server)
- Docker (optional, for containerized deployment)

---

## Installation

1. Clone this repository

2. Install Ollama (find online) and install llama3.2 model (or preferred, will have to update code in main.py)
```bash
ollama install llama3.2
```

3. Install ngrok (find online) and configure (can use another tunneling service)
  
4. (Optional) If running locally, have to import dependences (docker handles this)
```bash
pip install -r requirements.txt
```
5. (Optional) Install docker (find online) and run daemon (only for when running on Docker)

## Running the Server Locally
IMPORTANT: should have ngrok tunnel on the same port as FastAPI

1. Start ngrok
```bash
ngrok http 8000
```
2. Run the FastAPI app with uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Running with Docker (RECOMMEND)
IMPORTANT: should have ngrock tunnel on the same port as docker too

1. Build the Docker Image
```bash
docker build -t flower-classification-api .
```
2. Run the Docker Container
```bash
docker run -p 8000:8000 flower-classification-api
```
3. Start ngrok
```bash
ngrok http 8000
```

```bash
ngrok http 8000
```


