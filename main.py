from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from click import File
from fastapi import Form
from io import BytesIO
from PIL import Image

import requests
import numpy as np 
import subprocess
import logging

# Logging
logging.basicConfig(
    filename="plant_llm.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("plant_llm")

# Initiate Fast Api
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all incoming requests
    allow_methods=["*"],  # GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Content-Type, Authorization, etc.
)

# Load model and class list once
# flower_model = keras.models.load_model("baseline_model.keras")
flower_model = keras.models.load_model("cnn_baseline_model.h5")

efficient_net_model = keras.models.load_model("EfficientNetB0.h5")

# The list of class names that I found (need to verify)
classes = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

class FlowerRequest(BaseModel):
    flower_name: str

@app.post("/flower-info/by-image")
async def flower_info_by_image(
    file: UploadFile = File(...),
    model: str = Form("baseline")  # Default to "baseline"
):
    try:
        print("✅ Received request to /flower-info/by-image")
        print(f"🎯 Selected model: {model}") 

        # Step 1: Read and preprocess image
        contents = await file.read()
        print("📥 Image bytes read")

        img = Image.open(BytesIO(contents)).convert("RGB")
        print(f"🖼️ Image opened and converted to RGB, original size: {img.size}")

        # Step 2: Predict
        if model == "baseline":
            img = img.resize((224, 224))
            print(f"📏 Image resized to: {img.size}")

            img_array = image.img_to_array(img)
            print(f"🔢 Image converted to array, shape: {img_array.shape}")

            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(f"📦 Final model input shape: {img_array.shape}")
            prediction = flower_model.predict(img_array)
        elif model == "Efficient Net B0":
            # Resize to 224x224, keep pixel values [0-255], no normalization
            img = img.resize((224, 224))
            print(f"📏 Image resized to: {img.size}")

            img_array = image.img_to_array(img).astype(np.float32)  # keep [0-255]
            print(f"🔢 Image converted to array, shape: {img_array.shape}")

            img_array = np.expand_dims(img_array, axis=0)
            print(f"📦 Final model input shape: {img_array.shape}")

            prediction = efficient_net_model.predict(img_array)

        else:
            print("Testing: Model 3 has be asked for.")
            # prediction = model3.predict(img_array)
        
        print(f"🤖 Prediction raw output: {prediction}")

        predicted_index = int(np.argmax(prediction, axis=1)[0])
        predicted_name = classes[predicted_index]
        print(f"🌸 Predicted flower: {predicted_name} (index {predicted_index})")

        # Step 3: Format LLM prompt
        prompt = (
            f"Name: {predicted_name}\n\n"
            "History:\n"
            "1. Write a short history of this flower.\n"
            "2. Include origin or discovery context.\n"
            "3. Mention how it became known/popular.\n\n"
            "Important facts:\n"
            "- List one unique physical trait.\n"
            "- List one care or habitat tip.\n"
            "- Mention any symbolism or cultural use."
        )
        print(f"📝 Prompt to LLM:\n{prompt}")

        # Step 4: Run LLM (locally)
        # result = subprocess.run(
        #     ["/usr/local/bin/ollama", "run", "llama3.2", prompt],
        #     capture_output=True,
        #     text=True,
        #     check=True
        # )
        # description = result.stdout.strip()

        # With Docker
        description = query_ollama(prompt)
        print(f"📄 LLM response:\n{description}")

        return JSONResponse({
            "prediction": predicted_name,
            "description": description
        })

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return JSONResponse(
            {"error": "An error occurred while processing the image."},
            status_code=500
        )
    
def query_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        logger.error(f"Failed to get response from Ollama: {e}")
        return "LLM failed to generate description."